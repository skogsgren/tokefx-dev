from datetime import datetime
import json
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

from tokefx.utils import log

from icecream import ic


class AttentionAnalyzer:
    def __init__(
        self,
        model_spec,
        tokenizer_spec,
        ignored_pos: set | None = None,
        add_special_tokens=False,
        device="cpu",
        seed: int = 42,
        **kwargs,
    ):
        self.model_spec = model_spec
        self.tokenizer_spec = tokenizer_spec
        self.model = AutoModelForCausalLM.from_pretrained(
            model_spec,
            attn_implementation="eager",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_spec)
        self.device = torch.device(device)
        self.model.to(self.device).eval()

        cfg = self.model.config
        self.n_hidden = getattr(cfg, "hidden_size")
        self.n_heads = getattr(cfg, "num_attention_heads")
        self.head_dim = self.n_hidden // self.n_heads
        assert self.n_hidden % self.n_heads == 0

        self.seed = seed

    def _get_attention(self, model_inp):
        hooks = []
        blocks = self.model.model.layers
        oproj_collection = {layer_idx: [] for layer_idx, _ in enumerate(blocks)}
        for layer_idx, block in enumerate(blocks):
            o_proj = blocks[layer_idx].self_attn.o_proj

            def oproj_hook(module, inputs, layer_idx=layer_idx):
                x = inputs[0]  # [B, S, width]
                x0 = x[0].detach()
                seq_len, width = x0.shape
                head_dim = getattr(block.self_attn, "head_dim", None)
                if head_dim is None:
                    head_dim = getattr(block.self_attn.config, "head_dim", None)
                if head_dim is None:
                    n_heads = getattr(block.self_attn.config, "num_attention_heads")
                    head_dim = width // n_heads
                else:
                    if width % head_dim != 0:
                        raise RuntimeError(
                            f"layer {layer_idx}: width={width} not divisible by head_dim={head_dim}"
                        )
                    n_heads = width // head_dim
                oproj_collection[layer_idx] = x0.view(seq_len, n_heads, head_dim).cpu()
                """
                x = inputs[0]  # [B, S, hidden]
                x0 = x[0].detach()
                seq_len = x0.shape[0]
                oproj_collection[layer_idx] = x0.view(
                    seq_len, self.n_heads, self.head_dim
                ).cpu()
                oproj_collection[layer_idx] = x0.view(seq_len, n_heads, head_dim).cpu()
                """

            hooks.append(o_proj.register_forward_pre_hook(oproj_hook))

        with torch.no_grad():
            out = self.model(**model_inp, output_attentions=True, return_dict=True)

        attn = out.attentions if out.attentions is not None else out.decoder_attentions

        for hook in hooks:
            hook.remove()

        return attn, oproj_collection

    def _score(self, attn, oproj, q, source_indices):
        scores = []
        head_rows = []

        # calculate norms
        oproj_layer_norms = []
        for layer_idx, layer_tensor in oproj.items():
            # layer_tensor: [S, n_heads, head_dim]
            head_vecs = layer_tensor[q]  # [n_heads, head_dim]
            norm = head_vecs.norm(dim=-1).tolist()
            oproj_layer_norms.append(norm)  # [n_heads]

        # get individual and mean attn head scores
        for layer_i, atm in enumerate(attn):
            head_x_src = atm[0, :, q, source_indices]  # [heads, |src|]
            head_scores = head_x_src.mean(dim=-1)  # [heads]
            scores.append(float(head_scores.mean().item()))

            for head_i, v in enumerate(head_scores.tolist()):
                head_rows.append(
                    (layer_i, head_i, float(v), oproj_layer_norms[layer_i][head_i])
                )

        return scores, head_rows

    def __call__(self, candidates, goal):
        layer_rows = []
        head_rows = []
        for cand in candidates:
            model_inp = cand["model_inp"]
            attn, oproj = self._get_attention(model_inp)
            sc_row, h_rows = self._score(attn, oproj, -1, [-2])
            for layer_i, head_i, v, oproj_norm in h_rows:
                head_rows.append(
                    {
                        "layer": layer_i,
                        "head": head_i,
                        "score": v,
                        "head_norm": oproj_norm,
                    }
                )
            sc_row = {f"layer_{i:02d}": sc_row[i] for i in range(len(sc_row))}

            tgt_token = self.tokenizer.decode(model_inp["input_ids"][0][-1])
            src_token = self.tokenizer.decode(model_inp["input_ids"][0][-2])

            sc_row["tgt"] = tgt_token
            sc_row["src"] = src_token
            layer_rows.append(sc_row)
            log(f"{len(layer_rows)}/{goal} ({src_token}) <- ({tgt_token})")
        return layer_rows, head_rows


def get_ablation_map(
    parquet_fp: Path,
    top_k: int | None = None,
    mode_a: str = "in_boundary",
    mode_b: str = "out_boundary",
    rank_by: str = "min_z",
    min_layer: int = 0,
    max_layer: int = 0,
) -> dict[str, dict[str, pd.DataFrame]]:
    df = pd.read_parquet(parquet_fp)
    df["layer"] = pd.to_numeric(df["layer"], errors="coerce")
    df["head"] = pd.to_numeric(df["head"], errors="coerce")

    if min_layer:
        df = df[df["layer"] >= int(min_layer)].copy()
    if max_layer:
        df = df[df["layer"] <= int(max_layer)].copy()

    g = df.groupby(["lang", "model", "mode", "layer", "head"], as_index=False).agg(
        mean_score=("score", "mean"),
        mean_norm=("head_norm", "mean"),
    )

    a = (
        g[g["mode"] == mode_a]
        .drop(columns=["mode"])
        .rename(
            columns={
                "mean_score": "a_score",
                "mean_norm": "a_norm",
            }
        )
    )
    b = (
        g[g["mode"] == mode_b]
        .drop(columns=["mode"])
        .rename(
            columns={
                "mean_score": "b_score",
                "mean_norm": "b_norm",
            }
        )
    )

    merged = a.merge(
        b,
        on=["lang", "model", "layer", "head"],
        how="inner",
    )

    merged["score_delta"] = merged["a_score"] - merged["b_score"]
    merged["norm_delta"] = merged["a_norm"] - merged["b_norm"]

    out: dict[str, dict[str, pd.DataFrame]] = {}
    for lang, df_lang in merged.groupby("lang", sort=True):
        out[lang] = {}
        for model, df_model in df_lang.groupby("model", sort=True):
            d = df_model.copy()
            d = d[(d["score_delta"] > 0) & (d["norm_delta"] > 0)].copy()

            d["score_z"] = (d["score_delta"] - d["score_delta"].mean()) / (
                d["score_delta"].std(ddof=0) + 1e-9
            )
            d["norm_z"] = (d["norm_delta"] - d["norm_delta"].mean()) / (
                d["norm_delta"].std(ddof=0) + 1e-9
            )

            d["sum_z"] = d["score_z"] + d["norm_z"]
            d["min_z"] = d[["score_z", "norm_z"]].min(axis=1)
            d["product_z"] = d["score_z"] * d["norm_z"]

            ranked = (
                d[
                    [
                        "layer",
                        "head",
                        "a_score",
                        "b_score",
                        "score_delta",
                        "score_z",
                        "a_norm",
                        "b_norm",
                        "norm_delta",
                        "norm_z",
                        "sum_z",
                        "min_z",
                        "product_z",
                    ]
                ]
                .sort_values(
                    [rank_by, "score_delta", "norm_delta", "layer", "head"],
                    ascending=[False, False, False, True, True],
                )
                .reset_index(drop=True)
            )
            ranked = ranked[ranked[rank_by] > 0.0]

            if top_k is not None:
                ranked = ranked.head(top_k).copy()

            if top_k is None:
                top_k = len(ranked)

            out[lang][model] = {}
            for row in ranked.head(top_k).to_dict(orient="records"):
                if row["layer"] not in out[lang][model]:
                    out[lang][model][row["layer"]] = []
                out[lang][model][row["layer"]].append(row["head"])
    return out
