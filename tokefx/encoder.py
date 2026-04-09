from pathlib import Path
import random

from transformers import AutoTokenizer
import torch

from tokefx.data import PUDSequence, PUD_Data, PUDCandidate
from tokefx.data import iter_candidates_jsonl

from icecream import ic

INVALID_PREFIXES = {"’", "-"}


class UDEncoder:
    def __init__(
        self, tokenizer_spec: str, device: str = "cpu", seed: int = 42, **kwargs
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_spec)
        self.is_fast = bool(getattr(self.tokenizer, "is_fast", False))
        self.device = torch.device(device)
        self.seed = seed
        self.add_special_tokens = kwargs.get("add_special_tokens", False)
        self.min_context = kwargs.get("min_context", 0)
        self.min_bytes_per_token = kwargs.get("min_bytes_per_token", 0)
        self.max_tokens_per_sequence = kwargs.get("max_tokens_per_sequence", 0)
        self.ignored_pos = set(kwargs.get("ignored_pos", []))

    def _create_spans(self, seq, text, enc):
        """Yes, very annoying to have to do this type of offsetting, but since
        I need to compare BPE to UniGram tokenizers, I figured might as well do
        the more fair thing and just tokenize entire sequences instead..."""
        if self.is_fast:
            return self._create_spans_fast(seq, text, enc)
        return self._create_spans_byt5(seq, text, enc)

    def _ud_char_spans(self, seq):
        spans = []
        pos = 0
        for i, tok in enumerate(seq):
            gap = " " if (i > 0 and seq[i - 1].space_after) else ""
            token_text = gap + tok.form
            spans.append((pos, pos + len(token_text)))
            pos += len(token_text)
        return spans

    def _charspan_to_tokspan(self, offsets, special_mask, target):
        """given offsets returns the token indices overlapping with target character spans"""
        # target = (a, b) in character space
        # offsets[i] = (s, e) in character space for model token i
        a, b = target
        start_tok = None
        end_tok = None
        for i, (s, e) in enumerate(offsets):
            if special_mask[i]:
                continue
            if s == e:
                continue
            if e > a and s < b:  # overlap
                if start_tok is None:
                    start_tok = i
                end_tok = i + 1
        return (start_tok, end_tok)

    def _create_spans_fast(self, seq, text, enc):
        rebuilt = ""
        for i, tok in enumerate(seq):
            gap = " " if (i > 0 and seq[i - 1].space_after) else ""
            rebuilt += gap + tok.form
        offsets = [tuple(x) for x in enc["offset_mapping"][0].tolist()]
        special_mask = [bool(x) for x in enc["special_tokens_mask"][0].tolist()]
        ud_char_spans = self._ud_char_spans(seq)
        return [
            self._charspan_to_tokspan(offsets, special_mask, cs) for cs in ud_char_spans
        ]

    def _create_spans_byt5(self, seq, text, enc):
        """byt5 doesn't return offsets, but it's easy to recreate them"""
        assert not self.add_special_tokens, "not currently implemented for byt5"
        spans = []
        start = 0
        prev_token_space_after = False
        for token in seq:
            tok_len = len([x for x in token.form.encode("utf-8")])
            if prev_token_space_after:
                tok_len += 1
            spans.append((start, start + tok_len))
            start += tok_len
            if token.space_after:
                prev_token_space_after = True
            else:
                prev_token_space_after = False
        return spans

    def _encode(self, text):
        enc = self.tokenizer(
            text,
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt",
            return_offsets_mapping=self.is_fast,
            return_special_tokens_mask=self.is_fast,
        )
        model_inp = {"input_ids": enc["input_ids"].to(self.device)}
        if "attention_mask" in enc:
            model_inp["attention_mask"] = enc["attention_mask"].to(self.device)
        return enc, model_inp

    def get_encode_spans_candidates(
        self,
        seq: PUDSequence,
        ignored_pos: set = set(),
        tokens_per_sequence: int = 0,
        n_sequences: int = 0,
        min_context: int = 0,
        min_bytes_per_token: int = 1,
    ) -> tuple:
        # same as self.__call__ but also returns token candidates for sequences
        enc, model_inp = self._encode(seq.full_text())
        spans = self._create_spans_fast(seq, seq.full_text(), enc)

        in_candidates, out_candidates = [], []
        for tok in seq:
            if tok.idx == 0 or tok.idx < min_context:
                if min_context != 0:
                    continue
            if not tok.upos or tok.upos in ignored_pos:
                continue

            tok_bytes = len(tok.form.encode("utf-8"))
            if tok_bytes < min_bytes_per_token:
                continue
            start, end = spans[tok.idx]
            if end - start >= 2:
                in_candidates.append((end - 1, [end - 2], tok.form))

            prev_start, prev_end = spans[tok.idx - 1]
            prev_tok = seq[tok.idx - 1]
            if not prev_tok.upos or prev_tok.upos in ignored_pos:
                continue
            prev_tok_bytes = len(prev_tok.form.encode("utf-8"))
            if prev_tok_bytes < min_bytes_per_token:
                continue
            out_candidates.append((start, [start - 1], tok.form))
        candidates = []

        random.seed(self.seed)
        random.shuffle(in_candidates)
        random.shuffle(out_candidates)
        min_candidate_len = min(len(in_candidates), len(out_candidates))
        n_tokens = min(
            min_candidate_len if tokens_per_sequence == 0 else tokens_per_sequence,
            min_candidate_len,
        )
        for i in range(n_tokens):
            candidates.append(("in_boundary", in_candidates[i]))
            candidates.append(("out_boundary", out_candidates[i]))
        return enc, model_inp, spans, candidates

    def _valid_token(self, seq: PUDSequence, tgt_idx: int) -> bool:
        tok = seq[tgt_idx]
        if tok.idx == 0:
            return False
        if tok.idx < self.min_context:
            if self.min_context != 0:
                return False
        if not tok.upos or tok.upos in self.ignored_pos:
            return False
        for forbidden in INVALID_PREFIXES:
            if tok.form.startswith(forbidden):
                return False

        tok_bytes = len(tok.form.encode("utf-8"))
        if tok_bytes < self.min_bytes_per_token:
            return False

        return True

    def get_candidates(self, data: PUD_Data, condition: str):
        processed_tokens = set()
        for seq in data:
            valid_tokens = []
            enc, model_inp = self._encode(seq.full_text())
            spans = self._create_spans(seq, seq.full_text(), enc)
            for token in seq:
                if token.form.lower() in processed_tokens:
                    continue
                if not self._valid_token(seq, token.idx):
                    continue
                start, end = spans[token.idx]

                if condition == "in_boundary":
                    if not end - start >= 2:
                        continue
                    end_idx = end - 1
                    src_indices = [end - 2]
                elif condition == "out_boundary":
                    if token.idx == 0:
                        continue
                    if not self._valid_token(seq, token.idx - 1):
                        continue
                    end_idx = start
                    src_indices = [start - 1]
                else:
                    raise ValueError("unspecified candidate condition")

                valid_tokens.append(
                    {
                        "token": token,
                        "seq": seq,
                        "src_indices": src_indices,
                        "end": end_idx,
                    }
                )

            if not valid_tokens:
                continue

            random.seed(self.seed)
            random.shuffle(valid_tokens)

            counter = 0
            for i in range(min(self.max_tokens_per_sequence, len(valid_tokens))):
                if self.max_tokens_per_sequence and i > counter:
                    break
                cropped_inp = {
                    k: v[:, : valid_tokens[i]["end"] + 1].to(self.device)
                    for k, v in model_inp.items()
                }
                processed_tokens.add(valid_tokens[i]["token"].form.lower())
                counter += 1
                valid_tokens[i]["model_inp"] = cropped_inp
                yield valid_tokens[i]

    def __call__(self, seq: PUDSequence) -> tuple:
        # returns (enc, model_inp, spans)
        enc, model_inp = self._encode(seq.full_text())
        spans = self._create_spans_fast(seq, seq.full_text(), enc)
        return enc, model_inp, spans


def get_embed_attn_in_candidates(
    jsonl_fp: Path,
    tokenizer_spec: str,
    model_spec: str,
    lang: str,
    device: torch.device,
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_spec)
    in_candidates = []
    processed_tokens = set()
    for i, cand in enumerate(iter_candidates_jsonl(jsonl_fp, model_spec, lang)):
        seq = cand.seq
        if cand.token.form.lower() in processed_tokens:
            continue
        enc = tokenizer(seq.text_til_i(cand.token.idx), return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        in_candidates.append(
            {
                "seq": seq,
                "token": cand.token,
                "model_inp": enc,
                "src_indices": [-2],
                "end": -1,
            }
        )
    del tokenizer, processed_tokens
    return in_candidates
