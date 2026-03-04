#!/usr/bin/env python3
"""
Layer plots + significance heatmaps (Welch one-sided) + head delta heatmaps.

CLI:
  --plot_format pdf|png            (default: pdf)
  --control MODE                  (default: out_boundary)
  --modes MODE1,MODE2,...         (default: in_boundary,compound)
    -> compares each MODE in --modes against --control
       for BOTH one-sided directions per layer (Welch t-test).

Important fix:
- The dataframe is FILTERED to only (control + modes) immediately after loading.
  So if you do --modes in_boundary, compound will not appear anywhere.

Optional switches:
  --disable_significance
  --disable_layer_plots
  --disable_head_deltas

Requires: numpy, pandas, matplotlib, scipy
"""

import os
import re
import argparse
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind


# -----------------------------
# Helpers
# -----------------------------
def extract_layer_number(layer_name: str):
    m = re.search(r"(\d+)", str(layer_name))
    return int(m.group(1)) if m else None


def safe_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


def compute_layer_order_from_cols(layer_cols: list[str]) -> list[str]:
    parsed = [(c, extract_layer_number(c)) for c in layer_cols]
    if parsed and all(n is not None for _, n in parsed):
        return [c for c, _ in sorted(parsed, key=lambda t: t[1])]
    return sorted(layer_cols)


def compute_layer_order_from_values(layer_values: list[str]) -> list[str]:
    parsed = [(c, extract_layer_number(c)) for c in layer_values]
    if parsed and all(n is not None for _, n in parsed):
        return [c for c, _ in sorted(parsed, key=lambda t: (t[1], str(t[0])))]
    return sorted(layer_values)


def build_relative_depth_map_from_layer_nums(
    layer_order: list[str],
) -> dict[str, float]:
    nums = [extract_layer_number(x) for x in layer_order]
    if all(n is not None for n in nums) and len(nums) >= 2:
        mn = min(nums)
        mx = max(nums)
        denom = float(mx - mn) if mx != mn else 1.0
        return {
            layer: (extract_layer_number(layer) - mn) / denom for layer in layer_order
        }

    if len(layer_order) <= 1:
        return {layer_order[0]: 0.0} if layer_order else {}
    denom = float(len(layer_order) - 1)
    return {layer: i / denom for i, layer in enumerate(layer_order)}


def apply_plot_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titlesize": 20,
            "axes.labelsize": 14,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "axes.linewidth": 1.2,
            "grid.linewidth": 0.8,
            "grid.alpha": 0.25,
            "lines.linewidth": 2.3,
        }
    )


def save_fig(fig: plt.Figure, out_path: str, fmt: str):
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format=fmt, bbox_inches="tight")
    plt.close(fig)


def save_tsv(df: pd.DataFrame, out_path: str):
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)


# -----------------------------
# Wide parquet -> long table
# -----------------------------
def load_and_melt_raw_parquet(input_parquet: str) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_parquet(input_parquet)

    required = {"lang", "model", "mode"}
    if not required.issubset(df.columns):
        raise ValueError(f"Input must contain columns: {sorted(required)}")

    layer_cols = [c for c in df.columns if c.startswith("layer_")]
    if not layer_cols:
        raise ValueError("No layer_* columns found in input.")

    layer_order = compute_layer_order_from_cols(layer_cols)

    long_df = df.melt(
        id_vars=["lang", "model", "mode"],
        value_vars=layer_cols,
        var_name="layer",
        value_name="value",
    )

    long_df["lang"] = long_df["lang"].astype(str)
    long_df["model"] = long_df["model"].astype(str)
    long_df["mode"] = long_df["mode"].astype(str)
    long_df["layer"] = long_df["layer"].astype(str)
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df.dropna(subset=["value"])

    return long_df, layer_order


# -----------------------------
# Plot stats helpers
# -----------------------------
def layer_stats(df_slice: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    g = df_slice.groupby("layer")[value_col]
    out = g.agg(n="count", mean="mean", std="std").reset_index()
    out["median"] = g.median().values
    out["q25"] = g.quantile(0.25).values
    out["q75"] = g.quantile(0.75).values
    return out


def reindex_by_layer(df_stats: pd.DataFrame, layer_order: list[str]) -> pd.DataFrame:
    if df_stats["layer"].duplicated().any():
        raise ValueError("reindex_by_layer called with duplicate layer labels.")
    return df_stats.set_index("layer").reindex(layer_order).reset_index()


def layer_order_for_model(df_m: pd.DataFrame, fallback: list[str]) -> list[str]:
    model_layers = df_m["layer"].unique().tolist()
    return compute_layer_order_from_values(model_layers) if model_layers else fallback


# -----------------------------
# Multiple testing correction (BH-FDR)
# -----------------------------
def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)

    ok = np.isfinite(p)
    p_ok = p[ok]
    if p_ok.size == 0:
        return q

    n = p_ok.size
    order = np.argsort(p_ok)
    ranked = p_ok[order]
    q_ranked = ranked * n / (np.arange(1, n + 1))
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)

    q_ok = np.empty_like(q_ranked)
    q_ok[order] = q_ranked
    q[ok] = q_ok
    return q


# -----------------------------
# Welch t-test (one-sided)
# -----------------------------
_HAS_TTEST_ALTERNATIVE = "alternative" in inspect.signature(ttest_ind).parameters


def welch_ttest_onesided(
    a: np.ndarray, b: np.ndarray, alternative: str
) -> tuple[float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return np.nan, np.nan

    if _HAS_TTEST_ALTERNATIVE:
        res = ttest_ind(a, b, equal_var=False, alternative=alternative)
        return float(res.statistic), float(res.pvalue)

    # fallback: two-sided -> one-sided
    res2 = ttest_ind(a, b, equal_var=False)
    t = float(res2.statistic)
    p2 = float(res2.pvalue)
    if not np.isfinite(t) or not np.isfinite(p2):
        return t, np.nan

    if alternative == "greater":
        p1 = p2 / 2.0 if t > 0 else 1.0 - (p2 / 2.0)
    elif alternative == "less":
        p1 = p2 / 2.0 if t < 0 else 1.0 - (p2 / 2.0)
    else:
        raise ValueError(f"Invalid alternative: {alternative}")
    return t, float(np.clip(p1, 0.0, 1.0))


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    n1, n2 = a.size, b.size
    if n1 < 2 or n2 < 2:
        return np.nan
    s1 = np.var(a, ddof=1)
    s2 = np.var(b, ddof=1)
    denom_df = n1 + n2 - 2
    if denom_df <= 0:
        return np.nan
    sp = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / denom_df)
    if not np.isfinite(sp) or sp == 0:
        return np.nan
    d = (np.mean(a) - np.mean(b)) / sp
    J = 1.0 - (3.0 / (4.0 * (n1 + n2) - 9.0)) if (n1 + n2) > 2 else 1.0
    return float(J * d)


# -----------------------------
# Significance heatmaps
# -----------------------------
def validate_modes(df_long: pd.DataFrame, control: str, modes: list[str]):
    available = sorted(df_long["mode"].unique().tolist())
    needed = [control] + modes
    missing = [m for m in needed if m not in available]
    if missing:
        raise ValueError(
            f"Mode name mismatch.\nMissing: {missing}\nAvailable modes: {available}\n"
        )


def q_to_intensity(q: float, alpha: float = 0.05, bright_q: float = 0.005) -> float:
    if not np.isfinite(q) or q >= alpha:
        return 0.0
    q = max(float(q), 1e-300)
    hi = -np.log10(alpha)
    lo = -np.log10(bright_q)
    v = -np.log10(q)
    t = (v - hi) / (lo - hi) if lo > hi else 0.0
    return float(np.clip(t, 0.0, 1.0))


def _layer_tick_labels(layer_order: list[str], max_ticks: int = 18):
    L = len(layer_order)
    if L <= max_ticks:
        ticks = np.arange(L)
        labels = [
            str(extract_layer_number(x) if extract_layer_number(x) is not None else x)
            for x in layer_order
        ]
        return ticks, labels
    step = max(1, L // max_ticks)
    ticks = np.arange(0, L, step)
    labels = []
    for i in ticks:
        layer = layer_order[int(i)]
        n = extract_layer_number(layer)
        labels.append(str(n) if n is not None else layer)
    return ticks, labels


def run_welch_layer_tests(
    df_long: pd.DataFrame,
    layer_order: list[str],
    comparisons: list[tuple[str, str, str]],
    min_n: int = 20,
    alpha: float = 0.05,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    group_cols = group_cols or []
    layer_depth_map = build_relative_depth_map_from_layer_nums(layer_order)

    key_cols = ["model"] + group_cols
    keys = df_long[key_cols].drop_duplicates().sort_values(key_cols).to_dict("records")

    rows = []
    for key in keys:
        sel = np.ones(len(df_long), dtype=bool)
        for k, v in key.items():
            sel &= df_long[k] == v
        df_k = df_long[sel]

        for comp_name, mode_a, mode_b in comparisons:
            for layer in layer_order:
                a = df_k[(df_k["mode"] == mode_a) & (df_k["layer"] == layer)][
                    "value"
                ].to_numpy(dtype=float)
                b = df_k[(df_k["mode"] == mode_b) & (df_k["layer"] == layer)][
                    "value"
                ].to_numpy(dtype=float)
                a = a[np.isfinite(a)]
                b = b[np.isfinite(b)]
                n_a, n_b = int(a.size), int(b.size)

                mean_a = float(np.mean(a)) if n_a else np.nan
                mean_b = float(np.mean(b)) if n_b else np.nan
                med_a = float(np.median(a)) if n_a else np.nan
                med_b = float(np.median(b)) if n_b else np.nan

                base = dict(
                    **key,
                    comparison=comp_name,
                    mode_a=mode_a,
                    mode_b=mode_b,
                    layer=layer,
                    relative_depth=layer_depth_map.get(layer, np.nan),
                    n_a=n_a,
                    n_b=n_b,
                    mean_a=mean_a,
                    mean_b=mean_b,
                    median_a=med_a,
                    median_b=med_b,
                    mean_diff=(mean_a - mean_b)
                    if np.isfinite(mean_a) and np.isfinite(mean_b)
                    else np.nan,
                    hedges_g=np.nan,
                    t_stat=np.nan,
                    p_value=np.nan,
                    q_value=np.nan,
                    significant=False,
                )

                if n_a < min_n or n_b < min_n:
                    rows.append({**base, "direction": "a_gt_b"})
                    rows.append({**base, "direction": "b_gt_a"})
                    continue

                g = hedges_g(a, b)

                t_ab, p_ab = welch_ttest_onesided(a, b, alternative="greater")
                t_ba, p_ba = welch_ttest_onesided(b, a, alternative="greater")

                rows.append(
                    {
                        **base,
                        "direction": "a_gt_b",
                        "hedges_g": g,
                        "t_stat": t_ab,
                        "p_value": p_ab,
                    }
                )
                rows.append(
                    {
                        **base,
                        "direction": "b_gt_a",
                        "hedges_g": (-g if np.isfinite(g) else np.nan),
                        "t_stat": t_ba,
                        "p_value": p_ba,
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    grp = ["model"] + group_cols + ["comparison", "direction"]
    out["q_value"] = np.nan
    for _, g in out.groupby(grp, sort=False):
        idx = g.index
        out.loc[idx, "q_value"] = benjamini_hochberg(
            out.loc[idx, "p_value"].to_numpy(dtype=float)
        )

    out["significant"] = out["q_value"] < float(alpha)
    out = out.sort_values(grp + ["relative_depth", "layer"]).reset_index(drop=True)
    return out


def plot_directional_significance_heatmap(
    tests_df: pd.DataFrame,
    layer_order: list[str],
    comparison: str,
    out_path: str,
    plot_format: str,
    alpha: float = 0.05,
    bright_q: float = 0.005,
    title_prefix: str = "",
):
    apply_plot_style()
    dfc = tests_df[tests_df["comparison"] == comparison].copy()
    if dfc.empty:
        return

    models = sorted(dfc["model"].unique().tolist())
    layer_to_col = {layer: i for i, layer in enumerate(layer_order)}
    mat = np.zeros((len(models), len(layer_order)), dtype=float)

    for r, model in enumerate(models):
        d = dfc[dfc["model"] == model]
        for layer in layer_order:
            col = layer_to_col[layer]
            da = d[(d["layer"] == layer) & (d["direction"] == "a_gt_b")]
            db = d[(d["layer"] == layer) & (d["direction"] == "b_gt_a")]
            qa = float(da["q_value"].iloc[0]) if len(da) else np.nan
            qb = float(db["q_value"].iloc[0]) if len(db) else np.nan
            ia = q_to_intensity(qa, alpha=alpha, bright_q=bright_q)
            ib = q_to_intensity(qb, alpha=alpha, bright_q=bright_q)
            if ia > 0 and ib > 0:
                mat[r, col] = ia if ia >= ib else -ib
            elif ia > 0:
                mat[r, col] = ia
            elif ib > 0:
                mat[r, col] = -ib
            else:
                mat[r, col] = 0.0

    fig_w = max(10.0, 0.34 * len(layer_order))
    fig_h = max(4.8, 0.34 * len(models))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(
        mat,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )

    ax.set_title(
        f"{title_prefix}{comparison}\n"
        f"Welch t-tests (two one-sided), BH-FDR over layers; alpha={alpha}\n"
        f"red=a>ctrl, blue=ctrl>a; brightness: q≈{bright_q}→{alpha}"
    )
    ax.set_ylabel("model")
    ax.set_xlabel("layer")

    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(models)

    xticks, xlabels = _layer_tick_labels(layer_order, max_ticks=18)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    tick_qs = [alpha, 0.02, 0.01, bright_q]
    tick_qs = sorted(
        set([q for q in tick_qs if bright_q <= q <= alpha] + [bright_q]), reverse=True
    )
    pos_ticks = [q_to_intensity(q, alpha=alpha, bright_q=bright_q) for q in tick_qs]
    neg_ticks = [-t for t in pos_ticks]
    ticks = neg_ticks[::-1] + [0.0] + pos_ticks
    labels = (
        [f"ctrl>a (q={q:g})" for q in tick_qs][::-1]
        + ["ns"]
        + [f"a>ctrl (q={q:g})" for q in tick_qs]
    )
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels)
    cbar.ax.tick_params(labelsize=8)

    save_fig(fig, out_path, fmt=plot_format)


def significance_pipeline(
    df_long: pd.DataFrame,
    output_dir: str,
    plot_format: str,
    control: str,
    modes: list[str],
    alpha: float,
    bright_q: float,
    min_n: int,
):
    layer_order = compute_layer_order_from_values(df_long["layer"].unique().tolist())
    comparisons = [(f"{m}_vs_{control}", m, control) for m in modes]

    # pooled
    pooled_dir = os.path.join(output_dir, "significance_heatmaps", "pooled")
    pooled_stats = os.path.join(pooled_dir, "stats")
    os.makedirs(pooled_dir, exist_ok=True)
    os.makedirs(pooled_stats, exist_ok=True)

    pooled_tests = run_welch_layer_tests(
        df_long=df_long,
        layer_order=layer_order,
        comparisons=comparisons,
        min_n=min_n,
        alpha=alpha,
        group_cols=None,
    )
    save_tsv(pooled_tests, os.path.join(pooled_stats, "tests_all_models.tsv"))

    for comp_name, _, _ in comparisons:
        save_tsv(
            pooled_tests[pooled_tests["comparison"] == comp_name].copy(),
            os.path.join(pooled_stats, f"tests_{safe_filename(comp_name)}.tsv"),
        )
        plot_directional_significance_heatmap(
            tests_df=pooled_tests,
            layer_order=layer_order,
            comparison=comp_name,
            out_path=os.path.join(
                pooled_dir, f"heatmap_{safe_filename(comp_name)}.{plot_format}"
            ),
            plot_format=plot_format,
            alpha=alpha,
            bright_q=bright_q,
            title_prefix="Pooled | ",
        )

    # per-language
    bylang_root = os.path.join(output_dir, "significance_heatmaps", "by_language")
    os.makedirs(bylang_root, exist_ok=True)

    perlang_tests = run_welch_layer_tests(
        df_long=df_long,
        layer_order=layer_order,
        comparisons=comparisons,
        min_n=min_n,
        alpha=alpha,
        group_cols=["lang"],
    )

    for lang, g_lang in perlang_tests.groupby("lang", sort=True):
        lang_dir = os.path.join(bylang_root, safe_filename(lang))
        lang_stats = os.path.join(lang_dir, "stats")
        os.makedirs(lang_dir, exist_ok=True)
        os.makedirs(lang_stats, exist_ok=True)

        save_tsv(g_lang, os.path.join(lang_stats, f"tests_{safe_filename(lang)}.tsv"))

        for comp_name, _, _ in comparisons:
            plot_directional_significance_heatmap(
                tests_df=g_lang,
                layer_order=layer_order,
                comparison=comp_name,
                out_path=os.path.join(
                    lang_dir, f"heatmap_{safe_filename(comp_name)}.{plot_format}"
                ),
                plot_format=plot_format,
                alpha=alpha,
                bright_q=bright_q,
                title_prefix=f"{lang} | ",
            )


# -----------------------------
# Head delta plots
# -----------------------------
def load_head_parquet(input_parquet: str) -> pd.DataFrame:
    df = pd.read_parquet(input_parquet)
    required = {"lang", "model", "mode", "layer", "head", "score"}
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise ValueError(f"Head parquet missing columns: {missing}")

    df = df.copy()
    df["lang"] = df["lang"].astype(str)
    df["model"] = df["model"].astype(str)
    df["mode"] = df["mode"].astype(str)
    df["layer"] = pd.to_numeric(df["layer"], errors="coerce")
    df["head"] = pd.to_numeric(df["head"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["layer", "head", "score"])
    df["layer"] = df["layer"].astype(int)
    df["head"] = df["head"].astype(int)
    return df


def _pivot_layer_head_mean(df_slice: pd.DataFrame) -> pd.DataFrame:
    return (
        df_slice.groupby(["layer", "head"], as_index=False)["score"]
        .mean()
        .pivot(index="layer", columns="head", values="score")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )


def _robust_symmetric_lim(values: np.ndarray, pct: float = 99.0) -> float:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return 1.0
    lim = float(np.percentile(np.abs(v), pct))
    return max(lim, 1e-6)


def _plot_heatmap(
    mat, x_labels, y_labels, title, out_path, plot_format, cmap, vmin, vmax, cbar_label
):
    apply_plot_style()
    fig, ax = plt.subplots(figsize=(14.0, 7.5))
    im = ax.imshow(
        mat,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("head")
    ax.set_ylabel("layer")

    if len(x_labels) <= 32:
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=0)
    else:
        step = max(1, len(x_labels) // 16)
        ticks = np.arange(0, len(x_labels), step)
        ax.set_xticks(ticks)
        ax.set_xticklabels([x_labels[i] for i in ticks], rotation=0)

    if len(y_labels) <= 32:
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels)
    else:
        step = max(1, len(y_labels) // 16)
        ticks = np.arange(0, len(y_labels), step)
        ax.set_yticks(ticks)
        ax.set_yticklabels([y_labels[i] for i in ticks])

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.set_ylabel(cbar_label, rotation=90)
    save_fig(fig, out_path, fmt=plot_format)


def head_delta_plots(
    head_parquet: str, output_dir: str, plot_format: str, control: str, modes: list[str]
):
    df = load_head_parquet(head_parquet)

    available = sorted(df["mode"].unique().tolist())
    needed = [control] + modes
    missing = [m for m in needed if m not in available]
    if missing:
        raise ValueError(
            f"Head parquet mode mismatch. Missing: {missing}. Available: {available}"
        )

    # FILTER heads to only what we care about (same fix as wide)
    df = df[df["mode"].isin(needed)].copy()

    delta_pairs = [(f"{m}_minus_{control}", m, control) for m in modes]

    for model, df_m in df.groupby("model", sort=True):
        out_dir = os.path.join(output_dir, safe_filename(model), "heads_deltas")
        stats_dir = os.path.join(out_dir, "stats")
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(stats_dir, exist_ok=True)

        piv = {
            mode: _pivot_layer_head_mean(df_m[df_m["mode"] == mode]) for mode in needed
        }

        for name, mode_a, mode_b in delta_pairs:
            pa = piv.get(mode_a, pd.DataFrame())
            pb = piv.get(mode_b, pd.DataFrame())
            if pa.empty or pb.empty:
                continue

            layers = sorted(set(pa.index) | set(pb.index))
            heads = sorted(set(pa.columns) | set(pb.columns))

            da = pa.reindex(index=layers, columns=heads)
            db = pb.reindex(index=layers, columns=heads)
            diff = da - db

            vals = diff.to_numpy().ravel()
            lim = _robust_symmetric_lim(vals, pct=99.0)

            out_tsv = os.path.join(
                stats_dir, f"delta_{safe_filename(name)}_alllangs.tsv"
            )
            diff.reset_index().to_csv(out_tsv, sep="\t", index=False)

            out_path = os.path.join(
                out_dir, f"heatmap_delta_{safe_filename(name)}_alllangs.{plot_format}"
            )
            _plot_heatmap(
                mat=diff.to_numpy(),
                x_labels=[int(h) for h in heads],
                y_labels=[int(l) for l in layers],
                title=f"{model} | Δ({mode_a} − {mode_b}) | all languages",
                out_path=out_path,
                plot_format=plot_format,
                cmap="RdBu_r",
                vmin=-lim,
                vmax=lim,
                cbar_label="mean(score) delta",
            )


# -----------------------------
# Original layer plots (but now you control modes via filtering df_long)
# -----------------------------
def layer_plots(
    df_long: pd.DataFrame,
    global_layer_order: list[str],
    output_dir: str,
    plot_format: str,
):
    palette = list(plt.get_cmap("tab10").colors)
    line_styles = ["-", "--", ":", "-."]

    for model, df_m in df_long.groupby("model", sort=True):
        model_dir = os.path.join(output_dir, safe_filename(model))
        stats_dir = os.path.join(model_dir, "stats")
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(stats_dir, exist_ok=True)

        layer_order = layer_order_for_model(df_m, global_layer_order)
        rel_depth_map = build_relative_depth_map_from_layer_nums(layer_order)

        model_raw_stats_rows = []
        for mode, g_mode in df_m.groupby("mode", sort=True):
            s = layer_stats(g_mode, value_col="value")
            s = reindex_by_layer(s, layer_order)
            s.insert(0, "mode", mode)
            model_raw_stats_rows.append(s)

        if model_raw_stats_rows:
            model_raw_stats = pd.concat(model_raw_stats_rows, ignore_index=True)
            model_raw_stats["relative_depth"] = model_raw_stats["layer"].map(
                rel_depth_map
            )
            save_tsv(
                model_raw_stats,
                os.path.join(stats_dir, "raw_mode_layer_stats_across_languages.tsv"),
            )

        for lang, g_lang in df_m.groupby("lang", sort=True):
            modes_here = sorted(g_lang["mode"].unique().tolist())
            color_map = {m: palette[i % len(palette)] for i, m in enumerate(modes_here)}
            style_map = {
                m: line_styles[(i // len(palette)) % len(line_styles)]
                for i, m in enumerate(modes_here)
            }

            fig, ax = plt.subplots(figsize=(11.5, 6.5))
            lang_stats_rows = []

            for mode, g_lm in g_lang.groupby("mode", sort=True):
                s = layer_stats(g_lm, value_col="value")
                s = reindex_by_layer(s, layer_order)
                s["relative_depth"] = s["layer"].map(rel_depth_map)
                s.insert(0, "lang", lang)
                s.insert(1, "mode", mode)
                lang_stats_rows.append(s)

                x = s["relative_depth"].to_numpy(dtype=float)
                med = s["median"].to_numpy(dtype=float)
                q25 = s["q25"].to_numpy(dtype=float)
                q75 = s["q75"].to_numpy(dtype=float)

                ax.fill_between(
                    x, q25, q75, color=color_map[mode], alpha=0.18, linewidth=0
                )
                ax.plot(
                    x, med, color=color_map[mode], linestyle=style_map[mode], label=mode
                )

            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(bottom=0.0)
            ax.set_xlabel("relative layer depth")
            ax.set_ylabel("metric value")
            ax.set_title(f"{model} | {lang}")
            ax.legend(frameon=True, loc="upper left", ncols=1)

            save_fig(
                fig,
                os.path.join(
                    model_dir, f"by_language_{safe_filename(lang)}.{plot_format}"
                ),
                fmt=plot_format,
            )

            if lang_stats_rows:
                lang_stats = pd.concat(lang_stats_rows, ignore_index=True)
                save_tsv(
                    lang_stats,
                    os.path.join(
                        stats_dir,
                        f"raw_mode_layer_stats_lang_{safe_filename(lang)}.tsv",
                    ),
                )

        per_lang_mean = (
            df_m.groupby(["lang", "mode", "layer"], as_index=False)["value"]
            .mean()
            .rename(columns={"value": "lang_mean"})
        )

        agg_rows = []
        for mode, g_mode in per_lang_mean.groupby("mode", sort=True):
            gg = g_mode.groupby("layer")["lang_mean"]
            s = gg.agg(n_lang="count", mean="mean", std="std").reset_index()
            s["median"] = gg.median().values
            s["q25"] = gg.quantile(0.25).values
            s["q75"] = gg.quantile(0.75).values
            s.insert(0, "mode", mode)

            s = reindex_by_layer(s, layer_order)
            s["mode"] = mode
            s["relative_depth"] = s["layer"].map(rel_depth_map)
            agg_rows.append(s)

        agg = pd.concat(agg_rows, ignore_index=True) if agg_rows else pd.DataFrame()
        if not agg.empty:
            save_tsv(agg, os.path.join(stats_dir, "lang_agg_mode_layer_stats.tsv"))

        modes_here = sorted(per_lang_mean["mode"].unique().tolist())
        color_map = {m: palette[i % len(palette)] for i, m in enumerate(modes_here)}
        style_map = {
            m: line_styles[(i // len(palette)) % len(line_styles)]
            for i, m in enumerate(modes_here)
        }

        fig, ax = plt.subplots(figsize=(11.5, 6.5))
        for mode in modes_here:
            if agg.empty:
                continue
            s = agg[agg["mode"] == mode].copy()
            if s["layer"].duplicated().any():
                s = s.drop_duplicates(subset=["layer"], keep="first")
            s = reindex_by_layer(s, layer_order)
            s["relative_depth"] = s["layer"].map(rel_depth_map)

            x = s["relative_depth"].to_numpy(dtype=float)
            med = s["median"].to_numpy(dtype=float)
            q25 = s["q25"].to_numpy(dtype=float)
            q75 = s["q75"].to_numpy(dtype=float)

            ax.fill_between(x, q25, q75, color=color_map[mode], alpha=0.18, linewidth=0)
            ax.plot(
                x, med, color=color_map[mode], linestyle=style_map[mode], label=mode
            )

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("relative layer depth")
        ax.set_ylabel("language-mean metric (median ± IQR across langs)")
        ax.set_title(f"{model} | across languages")
        ax.legend(frameon=True, loc="upper left")
        save_fig(
            fig,
            os.path.join(model_dir, f"mode_comparison_across_languages.{plot_format}"),
            fmt=plot_format,
        )


# -----------------------------
# Main
# -----------------------------
def main():
    p = argparse.ArgumentParser(
        description="Plot layer metrics + Welch one-sided significance heatmaps (pooled + per-language) + head delta heatmaps."
    )
    p.add_argument(
        "--input_tsv",
        required=True,
        help="Wide parquet with lang, model, mode, layer_* columns.",
    )
    p.add_argument(
        "--heads_tsv",
        required=False,
        default=None,
        help="Long head parquet with lang, model, mode, layer, head, score.",
    )
    p.add_argument("--output_dir", required=True, help="Directory to write plots into")

    p.add_argument(
        "--plot_format",
        choices=["pdf", "png"],
        default="pdf",
        help="Export plots as PDF or PNG.",
    )
    p.add_argument(
        "--control", default="out_boundary", help="Control mode to compare against."
    )
    p.add_argument(
        "--modes",
        default="in_boundary,compound",
        help="Comma-separated modes to compare vs control.",
    )

    p.add_argument(
        "--alpha", type=float, default=0.05, help="FDR threshold (q < alpha)."
    )
    p.add_argument(
        "--bright_q",
        type=float,
        default=0.005,
        help="q that maps to max brightness in significance heatmaps.",
    )
    p.add_argument(
        "--min_n",
        type=int,
        default=20,
        help="Minimum samples per group per layer for testing.",
    )

    p.add_argument(
        "--disable_significance",
        action="store_true",
        help="Skip significance tests and heatmaps.",
    )
    p.add_argument(
        "--disable_layer_plots",
        action="store_true",
        help="Skip the original layer curve plots.",
    )
    p.add_argument(
        "--disable_head_deltas",
        action="store_true",
        help="Skip head delta heatmaps (even if heads_tsv is provided).",
    )

    args = p.parse_args()

    apply_plot_style()

    modes = [m.strip() for m in str(args.modes).split(",") if m.strip()]
    if not modes:
        raise ValueError("You gave --modes but it parsed to an empty list.")

    df_long, global_layer_order = load_and_melt_raw_parquet(args.input_tsv)

    # Validate against the ORIGINAL data first, so you get good error messages.
    validate_modes(df_long, control=args.control, modes=modes)

    # FIX: Filter the dataframe so only requested modes appear anywhere.
    selected_modes = [args.control] + modes
    df_long = df_long[df_long["mode"].isin(selected_modes)].copy()

    if not args.disable_significance:
        significance_pipeline(
            df_long=df_long,
            output_dir=args.output_dir,
            plot_format=args.plot_format,
            control=args.control,
            modes=modes,
            alpha=args.alpha,
            bright_q=args.bright_q,
            min_n=args.min_n,
        )

    if not args.disable_layer_plots:
        layer_plots(
            df_long=df_long,
            global_layer_order=global_layer_order,
            output_dir=args.output_dir,
            plot_format=args.plot_format,
        )

    if (args.heads_tsv is not None) and (not args.disable_head_deltas):
        head_delta_plots(
            head_parquet=args.heads_tsv,
            output_dir=args.output_dir,
            plot_format=args.plot_format,
            control=args.control,
            modes=modes,
        )


if __name__ == "__main__":
    main()
