from collections import defaultdict
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tokefx.data import PUD_Data, PUDSequence
from tokefx.utils import tokenize, token_len

from icecream import ic


class AttentionAnalyzer:
    def __init__(
        self,
        model_spec: str,
        tokenizer_spec: str,
        context_window: int = 4096,
        add_special_tokens: bool = False,
        device: str = "cpu",
    ):
        self.model_spec = model_spec
        self.model = AutoModelForCausalLM.from_pretrained(
            model_spec,
            attn_implementation="eager",  # needed to reliably return attentions
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_spec)
        self.context_window = context_window
        self.add_special_tokens = add_special_tokens
        self.device = torch.device(device)
        self.model.to(self.device).eval()

    def _get_attention(self, inp) -> tuple[torch.Tensor]:
        with torch.no_grad():
            out = self.model(**inp, output_attentions=True, return_dict=True)
        attn = out.attentions if out.attentions is not None else out.decoder_attentions
        return attn

    def _get_spans(self, seq: PUDSequence) -> list[tuple[int, int]]:
        """map each UD token to [start,end] model-token spans via prefix tokenization."""
        # NOTE: I don't like this function but I don't see any other way if I
        # want to use SentencePiece Unigram tokenizers
        spans: list[tuple[int, int]] = []
        for i in range(len(seq)):
            if i == 0:
                start = 0
            else:
                start = token_len(
                    tokenizer=self.tokenizer,
                    text=seq.text_until_token(i - 1),
                    context_window=self.context_window,
                    add_special_tokens=self.add_special_tokens,
                )
            end = token_len(
                tokenizer=self.tokenizer,
                text=seq.text_until_token(i),
                context_window=self.context_window,
                add_special_tokens=self.add_special_tokens,
            )
            spans.append((start, end))
        return spans

    def analyze(
        self,
        data: PUD_Data,
        goal: int,  # how many rows to return
        **kwargs,  # see /scripts/attention for example kwargs
    ) -> list[dict]:
        rows: list[dict] = []
        heads: list[dict] = []
        print(f"{datetime.now()} {len(rows)}/{goal}")

        for seq in data:
            if len(rows) >= goal and goal != 0:
                break

            text = seq.full_text()
            inp = tokenize(
                tokenizer=self.tokenizer,
                text=text,
                context_window=self.context_window,
                add_special_tokens=self.add_special_tokens,
                device=self.device,
            )
            attn = self._get_attention(inp)
            spans = self._get_spans(seq)
            seq_len = int(inp["input_ids"].shape[1])

            if kwargs["mode"] == "prev_subtokens":

                def is_valid(token) -> bool:
                    """since we do these checks twice, for current and prev token"""
                    if not kwargs["allow_foreign_words"] and tok.upos == "FW":
                        return False
                    if kwargs["skip_punct"] and tok.upos == "PUNCT":
                        return False
                    if kwargs["skip_propn"] and tok.upos == "PROPN":
                        return False
                    if kwargs["skip_part"] and tok.upos == "PART":
                        return False
                    return True

                tgt_len = kwargs["tgt_len"]
                for tok in seq:
                    if len(rows) >= goal and goal != 0:
                        break
                    if tok.idx < kwargs["min_context"]:
                        continue
                    if not is_valid(tok):
                        continue

                    start, end = spans[tok.idx]
                    num_subtokens = end - start

                    if num_subtokens == 0:
                        continue
                    token_bytes = len(tok.form.encode("utf-8"))
                    if token_bytes == 0:
                        continue

                    # NOTE: this is a naive implementation and experimentation
                    # with normalization is probably a good idea
                    if num_subtokens != tgt_len:
                        continue

                    q = end - 1
                    if num_subtokens == 1:
                        # single-token word -> previous token span
                        if tok.idx == 0:
                            continue
                        # we don't want to compare single-token words to groups
                        # which could add noise to the result, e.g. PUNCT
                        if not is_valid(seq[tok.idx - 1]):
                            continue
                        p_start, p_end = spans[tok.idx - 1]
                        source_indices = list(range(p_start, p_end))
                    else:
                        # multi-token word -> previous subtokens inside current word
                        source_indices = list(range(start, end - 1))

                    if not source_indices:
                        print(f"WARN: source_indices empty for {seq} {tok.form}")
                        continue
                    # NOTE: unsure if one should do mean here or just the last
                    # subtoken for each udtoken. currently just take the last
                    # one. hypothesis being that while more noisy it ensures a
                    # balanced comparison between single and two-token words
                    #
                    # removing [-1] here would instead do the mean (see when
                    # scores is appended)
                    source_indices = [
                        [i for i in source_indices if 0 <= i < seq_len][-1]
                    ]

                    scores: list[float] = []
                    for layer_i, atm in enumerate(attn, start=1):
                        head_x_src = atm[0, :, q, source_indices]
                        head_scores = head_x_src.sum(dim=-1)
                        scores.append(float(head_scores.mean().item()))
                        for head_i, v in enumerate(head_scores.tolist()):
                            heads.append(
                                {
                                    "mode": kwargs["mode"],
                                    "sentence_id": seq.sentence_id,
                                    "token_id": tok.idx,
                                    "tgt_len": kwargs["tgt_len"],
                                    "query_index": q,
                                    "source_index": int(source_indices[0]),
                                    "layer": layer_i,
                                    "head": head_i,
                                    "score": float(v),
                                }
                            )

                    row = {f"layer_{i + 1:02d}": scores[i] for i in range(len(scores))}
                    row["sentence_id"] = seq.sentence_id
                    row["token_form"] = tok.form
                    row["source_tokens_decoded"] = [
                        self.tokenizer.decode([int(inp["input_ids"][0][x].item())])
                        for x in source_indices
                    ]
                    row["query_token_decoded"] = [
                        self.tokenizer.decode([int(inp["input_ids"][0][q].item())])
                    ]
                    row["token_id"] = tok.idx
                    row["token_upos"] = tok.upos
                    row["num_subtokens"] = num_subtokens
                    row["n_raw"] = num_subtokens
                    row["token_bytes"] = token_bytes
                    row["subtokens_per_byte"] = num_subtokens / token_bytes
                    row["query_index"] = q
                    row["source_indices"] = source_indices
                    rows.append(row)
                    print(f"{datetime.now()} {len(rows)}/{goal}")
            else:
                raise ValueError(f"invalid mode {kwargs['mode']}")

        return rows, heads
