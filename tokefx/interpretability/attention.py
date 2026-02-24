from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from tokefx.data import PUD_Data, PUDSequence

ATTN_MODES = {"prev_subtokens", "prev_udtokens"}
ATTN_RED = {"mean", "sum"}


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
            attn_implementation="eager",  # most models don't return it otherwise
            # NOTE: eager attention is very costly though so be warned!
        )

        if getattr(self.model.config, "n_layer", False):
            self.n_layer = self.model.config.n_layer
        elif getattr(self.model.config, "num_hidden_layers", False):
            self.n_layer = self.model.config.num_hidden_layers
        else:
            raise ValueError("model config doesn't contain layer spec")

        if getattr(self.model.config, "n_heads", False):
            self.n_heads = self.model.config.n_heads
        elif getattr(self.model.config, "num_attention_heads", False):
            self.n_heads = self.model.config.num_attention_heads
        else:
            raise ValueError("model config doesn't contain attention head spec")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_spec)
        self.context_window = context_window
        self.add_special_tokens = add_special_tokens
        self.device = torch.device(device)

        self.model.to(self.device).eval()

    def _tokenize(self, text: str, tensors: bool = False):
        if tensors:
            enc = self.tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=self.add_special_tokens,
                truncation=True,
                max_length=self.context_window,
            )
            return {k: v.to(self.device) for k, v in enc.items()}
        return self.tokenizer(
            text,
            add_special_tokens=self.add_special_tokens,
            truncation=True,
            max_length=self.context_window,
        )["input_ids"]

    def _analyze_prevsubtokens(
        self, seq: PUDSequence, idx: int, goal: int
    ) -> tuple[float]:
        text = seq.get_text_til_i(idx)
        inp = self._tokenize(text, tensors=True)
        with torch.no_grad():
            out = self.model(**inp, output_attentions=True, return_dict=True)

        # NOTE: attn is a tuple of Tensors, but effectively this shape:
        # (Layers, BatchSize, Heads, Query (SeqLen), Key (SeqLen)
        # NOTE: though you'll always have to subscript Layers though
        attn = out.attentions
        if len(attn) != self.n_layer:
            raise ValueError("ERR: attention tuple is not the same size as n layers")
        if attn[0].shape[1] != self.n_heads:
            raise ValueError("ERR: attn heads in matrix is different to model config")

        # get attention scores for each layer from -n to -1
        layer_scores = []
        for lidx, atm in enumerate(attn):
            target_attn = atm[:, :, -1, -1 * goal]
            score = float(target_attn.sum().item()) / (goal - 1)
            layer_scores.append(score)
        return tuple(layer_scores)

    def analyze(
        self,
        data: PUD_Data,
        mode: str,
        min_context: int = 2,  # left-context tokens before match
        min_subtoken: int = 1,
        max_subtoken: int = 1,
        skip_punct: bool = True,
    ) -> dict:
        if mode == "prev_subtoken" and min_subtoken != max_subtoken:
            raise ValueError(
                "ERR: min_subtoken is different from max_subtoken",
                " with mode {mode} (should be equal)",
            )

        rows: list[dict] = []  # contains raw rows for each token
        for seq in tqdm(data, bar_format="{n}/{total} | {rate_fmt} | ETA: {remaining}"):
            # =================================================================
            # approximately the same as in kaplan2025
            # we attend only within the borders of a UDToken
            # makes all relevant checks before calling the parsing function
            # =================================================================
            if mode == "prev_subtokens":
                for tok in seq:
                    if len(tok.input_ids) != min_subtoken:
                        continue
                    if tok.idx < min_context:
                        continue
                    if skip_punct and tok.upos == "PUNCT":
                        continue
                    scores = self._analyze_prevsubtokens(seq, tok.idx, min_subtoken)
                    row = {f"layer_{i + 1:02d}": scores[i] for i in range(self.n_layer)}
                    row["sentence_id"] = seq.sentence_id
                    row["token_id"] = tok.idx
                    rows.append(row)
            # =================================================================
            #
            # =================================================================
            if mode == "prev_udtokens":
                raise NotImplementedError
            # =================================================================
            else:
                raise ValueError(f"ERR: mode must be {ATTN_MODES} (got {mode})")
            # =================================================================
        return pd.DataFrame(rows)
