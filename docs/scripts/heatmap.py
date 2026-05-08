#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter

MODE_A = "in_boundary"
MODE_B = "out_boundary"

SIGNED_CMAP = LinearSegmentedColormap.from_list(
    "high_contrast_blue_white_orange",
    [
        "#005AB5",  # stronger blue
        "#F2F2F2",  # light zero
        "#DC7A00",  # stronger orange
    ],
)


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))


def robust_symmetric_limit(values: np.ndarray, percentile: float = 99.0) -> float:
    finite = values[np.isfinite(values)]

    if finite.size == 0:
        return 1.0

    limit = float(np.percentile(np.abs(finite), percentile))
    return max(limit, 1e-6)


def apply_plot_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 1.0,
        }
    )


def load_head_parquet(input_path: str) -> pd.DataFrame:
    df = pd.read_parquet(input_path)

    required = {"lang", "model", "mode", "layer", "head", "score"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Input parquet missing columns: {sorted(missing)}")

    df = df.copy()

    df["lang"] = df["lang"].astype(str)
    df["model"] = df["model"].astype(str)
    df["mode"] = df["mode"].astype(str)

    df["layer"] = pd.to_numeric(df["layer"], errors="coerce")
    df["head"] = pd.to_numeric(df["head"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    df = df.dropna(subset=["lang", "model", "mode", "layer", "head", "score"])

    df["layer"] = df["layer"].astype(int)
    df["head"] = df["head"].astype(int)

    return df


def validate_inputs(df: pd.DataFrame, model: str, max_layer: int) -> None:
    available_models = sorted(df["model"].unique().tolist())

    if model not in available_models:
        raise ValueError(
            "Requested model not found.\n"
            f"Requested: {model}\n"
            f"Available models: {available_models}"
        )

    df_model = df[df["model"] == model]
    available_modes = sorted(df_model["mode"].unique().tolist())

    missing_modes = [mode for mode in [MODE_A, MODE_B] if mode not in available_modes]

    if missing_modes:
        raise ValueError(
            "Required boundary modes missing for requested model.\n"
            f"Missing: {missing_modes}\n"
            f"Available modes for {model}: {available_modes}"
        )

    if max_layer < 0:
        raise ValueError("--max_layer must be >= 0")


def filter_base_df(df: pd.DataFrame, model: str, max_layer: int) -> pd.DataFrame:
    out = df[
        (df["model"] == model)
        & (df["mode"].isin([MODE_A, MODE_B]))
        & (df["layer"] <= max_layer)
    ].copy()

    if out.empty:
        raise ValueError(
            f"No rows found for model={model}, modes={MODE_A}/{MODE_B}, "
            f"layers <= {max_layer}."
        )

    return out


def compute_delta_table(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["mode", "layer", "head"], as_index=False)["score"]
        .mean()
        .rename(columns={"score": "mean_score"})
    )

    in_boundary = (
        grouped[grouped["mode"] == MODE_A]
        .drop(columns=["mode"])
        .rename(columns={"mean_score": "in_boundary_score"})
    )

    out_boundary = (
        grouped[grouped["mode"] == MODE_B]
        .drop(columns=["mode"])
        .rename(columns={"mean_score": "out_boundary_score"})
    )

    delta = in_boundary.merge(
        out_boundary,
        on=["layer", "head"],
        how="inner",
    )

    if delta.empty:
        raise ValueError(
            f"No overlapping layer/head cells found between {MODE_A} and {MODE_B}."
        )

    delta["delta"] = delta["in_boundary_score"] - delta["out_boundary_score"]

    return delta.sort_values(["layer", "head"]).reset_index(drop=True)


def delta_table_to_matrix(
    delta: pd.DataFrame,
) -> tuple[np.ndarray, list[int], list[int]]:
    layers = sorted(delta["layer"].unique().tolist())
    heads = sorted(delta["head"].unique().tolist())

    matrix = (
        delta.pivot(index="layer", columns="head", values="delta")
        .reindex(index=layers, columns=heads)
        .to_numpy()
    )

    return matrix, layers, heads


def set_sparse_ticks(ax, labels: list[int], axis: str, max_ticks: int = 32) -> None:
    if len(labels) <= max_ticks:
        ticks = np.arange(len(labels))
    else:
        step = max(1, len(labels) // max_ticks)
        ticks = np.arange(0, len(labels), step)

    tick_labels = [labels[int(i)] for i in ticks]

    if axis == "x":
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels)
    elif axis == "y":
        ax.set_yticks(ticks)
        ax.set_yticklabels(tick_labels)
    else:
        raise ValueError("axis must be 'x' or 'y'")


def plot_heatmap(
    matrix: np.ndarray,
    layers: list[int],
    heads: list[int],
    title: str,
    output_path: Path,
    fmt: str,
) -> None:
    apply_plot_style()

    limit = robust_symmetric_limit(matrix.ravel(), percentile=85.0)

    fig_width = max(8.0, 0.35 * len(heads))
    fig_height = max(5.0, 0.35 * len(layers))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    norm = TwoSlopeNorm(
        vmin=-limit,
        vcenter=0.0,
        vmax=limit,
    )

    image = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap=SIGNED_CMAP,
        norm=norm,
    )

    ax.set_xticks(np.arange(matrix.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(matrix.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_title(title)
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")

    set_sparse_ticks(ax, heads, axis="x")
    set_sparse_ticks(ax, layers, axis="y")

    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)

    cbar.ax.set_ylabel("Δ score", rotation=90)

    cbar.ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{x:+.3g}" if x != 0 else "0")
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format=fmt, bbox_inches="tight")
    plt.close(fig)


def export_one_heatmap(
    df: pd.DataFrame,
    title: str,
    output_path: Path,
    fmt: str,
    write_stats: bool,
) -> None:
    delta = compute_delta_table(df)
    matrix, layers, heads = delta_table_to_matrix(delta)

    plot_heatmap(
        matrix=matrix,
        layers=layers,
        heads=heads,
        title=title,
        output_path=output_path,
        fmt=fmt,
    )

    print(f"Wrote heatmap: {output_path}")

    if write_stats:
        stats_path = output_path.with_suffix(".tsv")
        delta.to_csv(stats_path, sep="\t", index=False)
        print(f"Wrote stats:   {stats_path}")


def export_heatmaps(
    df: pd.DataFrame,
    model: str,
    max_layer: int,
    output_dir: Path,
    fmt: str,
    write_stats: bool,
) -> None:
    model_slug = safe_filename(model)

    # pooled
    pooled_path = output_dir / f"heatmap_{model_slug}.{fmt}"

    export_one_heatmap(
        df=df,
        title=(
            f"{model}\n"
            f"Pooled languages | mean score delta: "
            f"{MODE_A} − {MODE_B} | layers ≤ {max_layer}"
        ),
        output_path=pooled_path,
        fmt=fmt,
        write_stats=write_stats,
    )

    # per-language
    for lang, df_lang in df.groupby("lang", sort=True):
        lang_slug = safe_filename(lang)

        lang_path = output_dir / f"heatmap_{model_slug}_{lang_slug}.{fmt}"

        export_one_heatmap(
            df=df_lang,
            title=(
                f"{model}\n"
                f"{lang} | mean score delta: "
                f"{MODE_A} − {MODE_B} | layers ≤ {max_layer}"
            ),
            output_path=lang_path,
            fmt=fmt,
            write_stats=write_stats,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export pooled and per-language colorblind-friendly heatmaps for "
            "in_boundary minus out_boundary head score deltas."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input parquet with lang, model, mode, layer, head, score columns.",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Model name to filter.",
    )

    parser.add_argument(
        "--max_layer",
        type=int,
        required=True,
        help="Maximum layer to include, inclusive.",
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where heatmaps and TSVs will be written.",
    )

    parser.add_argument(
        "--format",
        choices=["pdf", "png"],
        default="pdf",
        help="Output image format. Default: pdf.",
    )

    parser.add_argument(
        "--no_stats",
        action="store_true",
        help="Do not export underlying delta tables as TSV.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_head_parquet(args.input)

    validate_inputs(
        df=df,
        model=args.model,
        max_layer=args.max_layer,
    )

    df = filter_base_df(
        df=df,
        model=args.model,
        max_layer=args.max_layer,
    )

    export_heatmaps(
        df=df,
        model=args.model,
        max_layer=args.max_layer,
        output_dir=output_dir,
        fmt=args.format,
        write_stats=not args.no_stats,
    )


if __name__ == "__main__":
    main()
