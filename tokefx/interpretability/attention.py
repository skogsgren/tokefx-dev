from datetime import datetime
import json
from math import ceil
import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

from tokefx.utils import log


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


def _load_and_filter(parquet_fp: Path, max_layer: int = 0) -> pd.DataFrame:
    df = pd.read_parquet(parquet_fp).copy()
    df["layer"] = pd.to_numeric(df["layer"], errors="coerce")
    df["head"] = pd.to_numeric(df["head"], errors="coerce")

    df = df.dropna(subset=["layer", "head"]).copy()
    df["layer"] = df["layer"].astype(int)
    df["head"] = df["head"].astype(int)

    if max_layer:
        df = df[df["layer"] <= max_layer]

    return df


def _merge_modes(df: pd.DataFrame, mode_a: str, mode_b: str) -> pd.DataFrame:
    grouped = df.groupby(
        ["lang", "model", "mode", "layer", "head"], as_index=False
    ).agg(
        mean_score=("score", "mean"),
        mean_norm=("head_norm", "mean"),
    )
    wide = grouped[grouped["mode"].isin([mode_a, mode_b])].pivot(
        index=["lang", "model", "layer", "head"],
        columns="mode",
        values=["mean_score", "mean_norm"],
    )
    wide.columns = [f"{mode}_{metric.split('_')[1]}" for metric, mode in wide.columns]
    wide = wide.reset_index()
    wide = wide.rename(
        columns={
            f"{mode_a}_score": "a_score",
            f"{mode_a}_norm": "a_norm",
            f"{mode_b}_score": "b_score",
            f"{mode_b}_norm": "b_norm",
        }
    )

    wide["score_delta"] = wide["a_score"] - wide["b_score"]
    wide["norm_delta"] = wide["a_norm"] - wide["b_norm"]

    return wide


def _rank_heads(
    df_model: pd.DataFrame, rank_by: str, lang: str, model: str
) -> pd.DataFrame:
    # d = df_model[(df_model["score_delta"] > 0) & (df_model["norm_delta"] > 0)].copy()
    d = df_model.copy()
    if d.empty:
        return d

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

    ranked = ranked[ranked[rank_by] > 0].copy()
    ranked["lang"] = lang
    ranked["model"] = model
    return ranked


def _select_global_with_layer_caps(
    ranked: pd.DataFrame, topk: int | None
) -> pd.DataFrame:
    limit = len(ranked) if topk is None else min(topk, len(ranked))
    per_layer_cap = (
        ranked.groupby("layer")["head"].max().floordiv(2).astype(int).to_dict()
    )

    selected = []
    used_per_layer: dict[int, int] = {}

    for row in ranked.itertuples(index=False):
        layer = int(row.layer)
        used = used_per_layer.get(layer, 0)
        cap = per_layer_cap.get(layer, 0)

        if used >= cap:
            continue

        selected.append(row._asdict())
        used_per_layer[layer] = used + 1

        if len(selected) >= limit:
            break

    return pd.DataFrame(selected)


def _to_ablation_map(selected: pd.DataFrame) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for row in selected.itertuples(index=False):
        out.setdefault(int(row.layer), []).append(int(row.head))
    return out


def get_ablation_map(
    parquet_fp: Path,
    topk: float | None = None,
    mode_a: str = "in_boundary",
    mode_b: str = "out_boundary",
    rank_by: str = "sum_z",
    max_layer: int = 0,
) -> tuple[dict[str, dict[str, dict[int, list[int]]]], pd.DataFrame]:
    df = _load_and_filter(parquet_fp, max_layer=max_layer)
    merged = _merge_modes(df, mode_a=mode_a, mode_b=mode_b)

    model_map = {}
    for model in df["model"].unique().tolist():
        filtered_df = df[df["model"] == model]
        n_layers = int(filtered_df["layer"].max()) + 1
        n_heads = int(filtered_df["head"].max()) + 1
        model_map[model] = round(topk * n_layers * n_heads)
    del filtered_df

    out: dict[str, dict[str, dict[int, list[int]]]] = {}
    ranked_parts = []

    for (lang, model), df_model in merged.groupby(["lang", "model"], sort=True):
        ranked = _rank_heads(df_model, rank_by=rank_by, lang=lang, model=model)
        ranked_parts.append(ranked)

        selected = _select_global_with_layer_caps(ranked, topk=model_map[model])

        out.setdefault(lang, {})
        out[lang][model] = _to_ablation_map(selected)

    ranked_df = (
        pd.concat(ranked_parts, ignore_index=True) if ranked_parts else pd.DataFrame()
    )
    return out, ranked_df
