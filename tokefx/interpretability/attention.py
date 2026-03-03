from datetime import datetime
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class AttentionAnalyzer:
    def __init__(
        self,
        model_spec,
        tokenizer_spec,
        ignored_pos: set | None = None,
        add_special_tokens=False,
        device="cpu",
    ):
        self.model_spec = model_spec
        self.tokenizer_spec = tokenizer_spec
        self.add_special_tokens = add_special_tokens
        self.ignored_pos = ignored_pos or set()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_spec,
            attn_implementation="eager",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_spec)
        self.is_fast = bool(getattr(self.tokenizer, "is_fast", False))
        self.device = torch.device(device)
        self.model.to(self.device).eval()

    def _get_attention(self, model_inp):
        with torch.no_grad():
            out = self.model(**model_inp, output_attentions=True, return_dict=True)
        return out.attentions if out.attentions is not None else out.decoder_attentions

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

    def _create_spans(self, seq, text, enc):
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

    def _score(self, attn, q, source_indices):
        scores = []
        head_rows = []

        for layer_i, atm in enumerate(attn, start=1):
            head_x_src = atm[0, :, q, source_indices]  # [heads, |src|]
            head_scores = head_x_src.mean(dim=-1)  # [heads]
            scores.append(float(head_scores.mean().item()))

            for head_i, v in enumerate(head_scores.tolist()):
                head_rows.append((layer_i, head_i, float(v)))

        return scores, head_rows

    def _format_scores(self, score, heads, sent_id, tok_id, mode, decoded, forms):
        fmt_head_rows = []
        for layer_i, head_i, v in heads:
            fmt_head_rows.append(
                {
                    "mode": mode,
                    "sentence_id": sent_id,
                    "token_id": tok_id,
                    "layer": layer_i,
                    "head": head_i,
                    "score": v,
                }
            )
        fmt_score_row = {f"layer_{i + 1:02d}": score[i] for i in range(len(score))}
        fmt_score_row["sentence_id"] = sent_id
        fmt_score_row["tok_id"] = tok_id
        fmt_score_row["mode"] = mode
        fmt_score_row["prev_token_form"] = forms["prev"]
        fmt_score_row["token_form"] = forms["curr"]
        fmt_score_row["src_decoded"] = decoded["source"]
        fmt_score_row["tgt_decoded"] = decoded["target"]
        return fmt_score_row, fmt_head_rows

    def _iter_compound_pairs(self, seq):
        for dep in seq:
            if not dep.deprel:
                continue
            if not dep.deprel == "compound":
                continue
            head_idx = dep.head
            if head_idx is None or head_idx < 0 or head_idx >= len(seq):
                continue
            if head_idx < dep.idx:
                continue
            head = seq[head_idx]
            if (not dep.upos) or (dep.upos in self.ignored_pos):
                continue
            if (not head.upos) or (head.upos in self.ignored_pos):
                continue
            yield dep, head

    def _compound_q_src(self, spans, dep_tok, head_tok, source_aggregation=None):
        d_start, d_end = spans[dep_tok.idx]
        h_start, h_end = spans[head_tok.idx]
        if d_start is None or d_end is None or h_start is None or h_end is None:
            return None, None
        if (d_end - d_start) < 1 or (h_end - h_start) < 1:
            return None, None
        q = h_start  # first subtoken of head (second part)
        source_indices = [d_end - 1]  # last subtoken of dep
        if source_aggregation == "mean":
            source_indices = list(range(d_start, d_end))
        return q, source_indices

    def analyze(self, data, goal, **kwargs):
        if kwargs.get("mode") == "boundary":
            print(f"NOTE: boundary is selected so {goal=} is doubled (in/out boundary)")
            goal = int(goal * 2)
        rows = []
        heads = []

        print(f"{datetime.now()} {len(rows)}/{goal}")
        for seq in data:
            if goal and len(rows) >= goal:
                break

            text = seq.full_text()
            enc, model_inp = self._encode(text)
            attn = self._get_attention(model_inp)
            spans = self._create_spans(seq, text, enc)

            input_ids = enc["input_ids"][0]
            if kwargs.get("mode") == "compound":
                # we assume only two parts in compound
                for dep, head in self._iter_compound_pairs(seq):
                    if goal and len(rows) >= goal:
                        break
                    if head.idx < kwargs["min_context"]:
                        continue
                    q, source_indices = self._compound_q_src(
                        spans,
                        dep,
                        head,
                        source_aggregation=kwargs.get("source_aggregation"),
                    )
                    if q is None:
                        continue

                    decoded = {
                        "source": self.tokenizer.decode(
                            [input_ids[i].item() for i in source_indices]
                        ).replace(" ", "Ġ"),
                        "target": self.tokenizer.decode([input_ids[q].item()]).replace(
                            " ", "Ġ"
                        ),
                    }
                    sc_row, h_rows = self._format_scores(
                        *self._score(attn, q, source_indices),
                        seq.sentence_id,
                        dep.idx,
                        "compound",
                        decoded,
                        {"prev": dep.form, "curr": head.form},
                    )
                    rows.append(sc_row)
                    heads += h_rows
                    print(f"{datetime.now()} {len(rows)}/{goal}")
                    continue
            elif kwargs.get("mode") == "boundary":
                for tok in seq:
                    if goal and len(rows) >= goal:
                        break
                    if tok.idx < kwargs["min_context"]:
                        continue
                    if not tok.upos or tok.upos in self.ignored_pos:
                        continue
                    token_bytes = len(tok.form.encode("utf-8"))
                    assert token_bytes > 0

                    start, end = spans[tok.idx]

                    if tok.idx == 0:
                        continue
                    prev_start, prev_end = spans[tok.idx - 1]
                    if seq[tok.idx - 1].upos in self.ignored_pos:
                        continue
                    if end - start < 2:
                        continue
                    pairs = [
                        ("in_boundary", (end - 1, [end - 2])),
                        ("out_boundary", (start, [start - 1])),
                    ]
                    # mean aggregation takes the mean attention to the entire
                    # span of either current word (for in_boundary) or
                    # span of previous ud token (for out_boundary)
                    if kwargs.get("source_aggregation") == "mean":
                        in_src = list(range(start, end - 1))
                        out_src = list(range(prev_start, prev_end))
                        if len(in_src) == 0 or len(out_src) == 0:
                            continue
                        pairs = [
                            ("in_boundary (mean)", (end - 1, in_src)),
                            ("out_boundary (mean)", (start, out_src)),
                        ]
                    for mode_label, (q, source_indices) in pairs:
                        decoded = {
                            "source": self.tokenizer.decode(
                                [input_ids[i].item() for i in source_indices]
                            ).replace(" ", "Ġ"),
                            "target": self.tokenizer.decode(
                                [input_ids[q].item()]
                            ).replace(" ", "Ġ"),
                        }
                        sc_row, h_rows = self._format_scores(
                            *self._score(attn, q, source_indices),
                            seq.sentence_id,
                            tok.idx,
                            mode_label,
                            decoded,
                            {"prev": seq[tok.idx - 1].form, "curr": tok.form},
                        )
                        rows.append(sc_row)
                        heads += h_rows
                        print(f"{datetime.now()} {len(rows)}/{goal}")
                    continue
        return rows, heads
