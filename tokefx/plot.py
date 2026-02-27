#!/usr/bin/env python3
import os
import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


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


def build_relative_depth_map(layer_order: list[str]) -> dict[str, float]:
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
            "axes.titlesize": 24,
            "axes.labelsize": 18,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "axes.linewidth": 1.2,
            "grid.linewidth": 0.9,
            "grid.alpha": 0.5,
            "lines.linewidth": 2.6,
        }
    )


def save_pdf(fig: plt.Figure, out_path: str):
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def save_tsv(df: pd.DataFrame, out_path: str):
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)


def load_and_melt_raw_tsv(input_tsv: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Input: wide TSV with columns:
      - lang, model, mode
      - layer_01 ... layer_N
      - plus arbitrary other columns (ignored)
    Output: long df with columns: lang, model, mode, layer, value
    """
    df = pd.read_csv(input_tsv, sep="\t")

    required = {"lang", "model", "mode"}
    if not required.issubset(df.columns):
        raise ValueError(f"Input TSV must contain columns: {sorted(required)}")

    layer_cols = [c for c in df.columns if c.startswith("layer_")]
    if not layer_cols:
        raise ValueError("No layer_* columns found in input TSV.")

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


def layer_stats(df_slice: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    """
    Compute per-layer stats for a slice.
    Returns columns: layer, n, mean, median, std, q25, q75
    """
    g = df_slice.groupby("layer")[value_col]
    out = g.agg(n="count", mean="mean", std="std").reset_index()
    out["median"] = g.median().values
    out["q25"] = g.quantile(0.25).values
    out["q75"] = g.quantile(0.75).values
    return out


def reindex_by_layer(df_stats: pd.DataFrame, layer_order: list[str]) -> pd.DataFrame:
    """
    Reindex a stats frame that has *unique* 'layer' values.
    If not unique, caller must split first (e.g., by mode).
    """
    if df_stats["layer"].duplicated().any():
        raise ValueError(
            "reindex_by_layer called with duplicate layer labels; split by mode first."
        )
    return df_stats.set_index("layer").reindex(layer_order).reset_index()


# -----------------------------
# Plotting logic
# -----------------------------
def attention_plots(input_tsv: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    apply_plot_style()

    df_long, global_layer_order = load_and_melt_raw_tsv(input_tsv)

    palette = list(plt.get_cmap("tab10").colors)
    line_styles = ["-", "--", ":", "-."]

    # ------------------------------------------------------------------
    # Per model
    # ------------------------------------------------------------------
    for model, df_m in df_long.groupby("model", sort=True):
        model_dir = os.path.join(output_dir, safe_filename(model))
        stats_dir = os.path.join(model_dir, "stats")
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(stats_dir, exist_ok=True)

        # Build layer order per model (models might differ)
        model_layers = df_m["layer"].unique().tolist()
        if model_layers:
            layer_order = sorted(
                model_layers,
                key=lambda x: (
                    extract_layer_number(x) is None,
                    extract_layer_number(x)
                    if extract_layer_number(x) is not None
                    else x,
                ),
            )
        else:
            layer_order = global_layer_order

        rel_depth_map = build_relative_depth_map(layer_order)

        # ------------------------------------------------------------
        # Export: model-level raw stats (across ALL languages/tokens)
        # per (mode, layer)
        # ------------------------------------------------------------
        model_raw_stats_rows = []
        for mode, g_mode in df_m.groupby("mode", sort=True):
            s = layer_stats(g_mode, value_col="value")
            # unique layers inside this mode slice, safe to reindex
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

        # ------------------------------------------------------------
        # 1) Language plots: median + IQR band per mode
        #    Stats TSV exported per language
        # ------------------------------------------------------------
        for lang, g_lang in df_m.groupby("lang", sort=True):
            modes = sorted(g_lang["mode"].unique().tolist())
            color_map = {m: palette[i % len(palette)] for i, m in enumerate(modes)}
            style_map = {
                m: line_styles[(i // len(palette)) % len(line_styles)]
                for i, m in enumerate(modes)
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

                c = color_map[mode]
                ls = style_map[mode]

                ax.fill_between(x, q25, q75, color=c, alpha=0.18, linewidth=0)
                ax.plot(x, med, color=c, linestyle=ls, label=mode)

            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(bottom=0.0)
            ax.set_xlabel("relative layer depth")
            ax.set_ylabel("metric value")
            ax.set_title(f"{model} | {lang}")
            ax.legend(frameon=True, loc="upper left", ncols=1)

            out_pdf = os.path.join(model_dir, f"by_language_{safe_filename(lang)}.pdf")
            save_pdf(fig, out_pdf)

            if lang_stats_rows:
                lang_stats = pd.concat(lang_stats_rows, ignore_index=True)
                save_tsv(
                    lang_stats,
                    os.path.join(
                        stats_dir,
                        f"raw_mode_layer_stats_lang_{safe_filename(lang)}.tsv",
                    ),
                )

        # ------------------------------------------------------------
        # 2) Cross-language mode comparison:
        #    - per-language mean per (lang, mode, layer)
        #    - across languages: median + IQR (and mean/std if you want)
        #    - export TSV + plot median +/- IQR
        # ------------------------------------------------------------
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

            # Reindex *inside* mode slice (unique layers here)
            s = reindex_by_layer(s, layer_order)
            s["mode"] = mode  # reindex resets some columns order, keep mode correct
            agg_rows.append(s)

        agg = pd.concat(agg_rows, ignore_index=True) if agg_rows else pd.DataFrame()
        if not agg.empty:
            agg["relative_depth"] = agg["layer"].map(rel_depth_map)
            save_tsv(agg, os.path.join(stats_dir, "lang_agg_mode_layer_stats.tsv"))

        modes = sorted(per_lang_mean["mode"].unique().tolist())
        color_map = {m: palette[i % len(palette)] for i, m in enumerate(modes)}
        style_map = {
            m: line_styles[(i // len(palette)) % len(line_styles)]
            for i, m in enumerate(modes)
        }

        fig, ax = plt.subplots(figsize=(11.5, 6.5))

        for mode in modes:
            if agg.empty:
                continue
            s = agg[agg["mode"] == mode].copy()
            # s already reindexed when built, but keep order consistent anyway:
            if s["layer"].duplicated().any():
                # should never happen now
                s = s.drop_duplicates(subset=["layer"], keep="first")
            s = reindex_by_layer(s, layer_order)
            s["relative_depth"] = s["layer"].map(rel_depth_map)

            x = s["relative_depth"].to_numpy(dtype=float)
            med = s["median"].to_numpy(dtype=float)
            q25 = s["q25"].to_numpy(dtype=float)
            q75 = s["q75"].to_numpy(dtype=float)

            c = color_map[mode]
            ls = style_map[mode]

            ax.fill_between(x, q25, q75, color=c, alpha=0.18, linewidth=0)
            ax.plot(x, med, color=c, linestyle=ls, label=mode)

        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel("relative layer depth")
        ax.set_ylabel("language-mean metric (median ± IQR across langs)")
        ax.set_title(f"{model} | across languages")
        ax.legend(frameon=True, loc="upper left")

        out_pdf = os.path.join(model_dir, "mode_comparison_across_languages.pdf")
        save_pdf(fig, out_pdf)


# -----------------------------
# Head TSV loading + plotting
# -----------------------------
def load_head_tsv(input_tsv: str) -> pd.DataFrame:
    """
    Input: long TSV with columns including:
      lang, model, mode, layer, head, score
    Output: cleaned df with those columns typed.
    """
    df = pd.read_csv(input_tsv, sep="\t")

    required = {"lang", "model", "mode", "layer", "head", "score"}
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise ValueError(f"Head TSV missing columns: {missing}")

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
    """
    Returns a pivot table with index=layer (sorted asc), columns=head (sorted asc),
    values=mean(score). Missing cells become NaN.
    """
    pt = (
        df_slice.groupby(["layer", "head"], as_index=False)["score"]
        .mean()
        .pivot(index="layer", columns="head", values="score")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    return pt


def _robust_vmin_vmax(values: np.ndarray, lo=1.0, hi=99.0) -> tuple[float, float]:
    """
    Robust color scaling ignoring NaNs.
    """
    v = values[np.isfinite(values)]
    if v.size == 0:
        return 0.0, 1.0
    vmin = float(np.percentile(v, lo))
    vmax = float(np.percentile(v, hi))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    return vmin, vmax


def _plot_heatmap(
    mat: np.ndarray,
    x_labels: list[int],
    y_labels: list[int],
    title: str,
    out_pdf: str,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    xlabel: str = "head",
    ylabel: str = "layer",
):
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
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Tick strategy: don’t label every single head/layer unless tiny
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
    cbar.ax.set_ylabel("mean attention score", rotation=90)

    save_pdf(fig, out_pdf)


def attention_head_plots(head_tsv: str, output_dir: str):
    """
    For each model:
      - Heatmap per mode: mean(score) for (layer, head) across all languages
      - If exactly 2 modes: difference heatmap (modeB - modeA)
      - Repeat per language
    Also exports TSV stats that back the heatmaps.
    """
    os.makedirs(output_dir, exist_ok=True)
    apply_plot_style()

    df = load_head_tsv(head_tsv)

    palette = list(plt.get_cmap("tab10").colors)

    # ------------------------------------------------------------
    # Per model (never mix models in a plot)
    # ------------------------------------------------------------
    for model, df_m in df.groupby("model", sort=True):
        model_dir = os.path.join(output_dir, safe_filename(model), "heads")
        stats_dir = os.path.join(model_dir, "stats")
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(stats_dir, exist_ok=True)

        modes = sorted(df_m["mode"].unique().tolist())

        # --------------------------------------------------------
        # A) Across all languages: per-mode heatmaps
        # --------------------------------------------------------
        pivots = {}
        for mode in modes:
            pt = _pivot_layer_head_mean(df_m[df_m["mode"] == mode])
            pivots[mode] = pt

            # Export the backing table as TSV (layer rows, head columns)
            out_tsv = os.path.join(
                stats_dir, f"mean_layer_head_alllangs_mode_{safe_filename(mode)}.tsv"
            )
            pt.reset_index().to_csv(out_tsv, sep="\t", index=False)

        # Shared color scale across modes for this model (so you can compare)
        all_vals = (
            np.concatenate(
                [p.to_numpy().ravel() for p in pivots.values() if not p.empty]
            )
            if pivots
            else np.array([])
        )
        vmin, vmax = _robust_vmin_vmax(all_vals, lo=1.0, hi=99.0)

        for mode, pt in pivots.items():
            if pt.empty:
                continue
            mat = pt.to_numpy()
            x_labels = pt.columns.astype(int).tolist()
            y_labels = pt.index.astype(int).tolist()
            out_pdf = os.path.join(
                model_dir, f"heatmap_alllangs_mode_{safe_filename(mode)}.pdf"
            )
            _plot_heatmap(
                mat=mat,
                x_labels=x_labels,
                y_labels=y_labels,
                title=f"{model} | all languages | mode={mode}",
                out_pdf=out_pdf,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )

        # Difference heatmap if exactly two modes
        if len(modes) == 2 and all(m in pivots for m in modes):
            a, b = modes[0], modes[1]
            # Align indices/columns
            pa = pivots[a]
            pb = pivots[b]
            common_layers = sorted(set(pa.index) | set(pb.index))
            common_heads = sorted(set(pa.columns) | set(pb.columns))

            pa2 = pa.reindex(index=common_layers, columns=common_heads)
            pb2 = pb.reindex(index=common_layers, columns=common_heads)
            diff = pb2 - pa2

            # robust scale symmetric around 0
            diff_vals = diff.to_numpy().ravel()
            dv = diff_vals[np.isfinite(diff_vals)]
            if dv.size:
                lim = float(np.percentile(np.abs(dv), 99.0))
                lim = max(lim, 1e-6)
            else:
                lim = 1.0

            out_tsv = os.path.join(
                stats_dir,
                f"diff_layer_head_alllangs_{safe_filename(b)}_minus_{safe_filename(a)}.tsv",
            )
            diff.reset_index().to_csv(out_tsv, sep="\t", index=False)

            out_pdf = os.path.join(
                model_dir,
                f"heatmap_alllangs_diff_{safe_filename(b)}_minus_{safe_filename(a)}.pdf",
            )
            _plot_heatmap(
                mat=diff.to_numpy(),
                x_labels=diff.columns.astype(int).tolist(),
                y_labels=diff.index.astype(int).tolist(),
                title=f"{model} | all languages | Δ({b} − {a})",
                out_pdf=out_pdf,
                cmap="RdBu_r",
                vmin=-lim,
                vmax=lim,
            )

        # --------------------------------------------------------
        # B) Per language heatmaps (per mode, and diff if 2 modes)
        # --------------------------------------------------------
        for lang, df_ml in df_m.groupby("lang", sort=True):
            lang_dir = os.path.join(model_dir, "by_language", safe_filename(lang))
            os.makedirs(lang_dir, exist_ok=True)

            piv_lang = {}
            for mode in modes:
                pt = _pivot_layer_head_mean(df_ml[df_ml["mode"] == mode])
                piv_lang[mode] = pt
                out_tsv = os.path.join(
                    stats_dir,
                    f"mean_layer_head_lang_{safe_filename(lang)}_mode_{safe_filename(mode)}.tsv",
                )
                pt.reset_index().to_csv(out_tsv, sep="\t", index=False)

            # Per-language shared scale across modes (more readable than forcing global)
            all_vals_lang = (
                np.concatenate(
                    [p.to_numpy().ravel() for p in piv_lang.values() if not p.empty]
                )
                if piv_lang
                else np.array([])
            )
            lvmin, lvmax = _robust_vmin_vmax(all_vals_lang, lo=1.0, hi=99.0)

            for mode, pt in piv_lang.items():
                if pt.empty:
                    continue
                out_pdf = os.path.join(
                    lang_dir, f"heatmap_mode_{safe_filename(mode)}.pdf"
                )
                _plot_heatmap(
                    mat=pt.to_numpy(),
                    x_labels=pt.columns.astype(int).tolist(),
                    y_labels=pt.index.astype(int).tolist(),
                    title=f"{model} | {lang} | mode={mode}",
                    out_pdf=out_pdf,
                    cmap="viridis",
                    vmin=lvmin,
                    vmax=lvmax,
                )

            if len(modes) == 2 and all(m in piv_lang for m in modes):
                a, b = modes[0], modes[1]
                pa = piv_lang[a]
                pb = piv_lang[b]
                common_layers = sorted(set(pa.index) | set(pb.index))
                common_heads = sorted(set(pa.columns) | set(pb.columns))

                pa2 = pa.reindex(index=common_layers, columns=common_heads)
                pb2 = pb.reindex(index=common_layers, columns=common_heads)
                diff = pb2 - pa2

                diff_vals = diff.to_numpy().ravel()
                dv = diff_vals[np.isfinite(diff_vals)]
                if dv.size:
                    lim = float(np.percentile(np.abs(dv), 99.0))
                    lim = max(lim, 1e-6)
                else:
                    lim = 1.0

                out_pdf = os.path.join(
                    lang_dir,
                    f"heatmap_diff_{safe_filename(b)}_minus_{safe_filename(a)}.pdf",
                )
                _plot_heatmap(
                    mat=diff.to_numpy(),
                    x_labels=diff.columns.astype(int).tolist(),
                    y_labels=diff.index.astype(int).tolist(),
                    title=f"{model} | {lang} | Δ({b} − {a})",
                    out_pdf=out_pdf,
                    cmap="RdBu_r",
                    vmin=-lim,
                    vmax=lim,
                )


def main():
    p = argparse.ArgumentParser(
        description="Plot aggregated layer metrics (wide TSV) + head heatmaps (long head TSV)."
    )
    p.add_argument(
        "--input_tsv",
        required=True,
        help="Wide TSV with columns including lang, model, mode, and layer_* columns.",
    )
    p.add_argument(
        "--heads_tsv",
        required=False,
        default=None,
        help="Long head TSV with columns: lang, model, mode, layer, head, score (plus optional metadata).",
    )
    p.add_argument("--output_dir", required=True, help="Directory to write plots into")
    args = p.parse_args()

    attention_plots(args.input_tsv, args.output_dir)

    if args.heads_tsv is not None:
        attention_head_plots(args.heads_tsv, args.output_dir)


if __name__ == "__main__":
    main()
