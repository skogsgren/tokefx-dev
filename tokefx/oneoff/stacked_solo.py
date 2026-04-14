import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET_COMPONENT = "layer_out"
REQUIRED_COLUMNS = {"model", "layer", "component", "retrieved", "ablation"}

CLEAN_ABLATION = "clean"
RANDOM_ABLATION = "random"
TARGETED_ABLATION = "targeted"


def validate_required_columns(df: pd.DataFrame, required_columns: set[str]) -> None:
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def short_model_name(name: str) -> str:
    name = str(name)
    if "/" in name:
        name = name.split("/")[-1]
    return name


def slugify(value: object) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


def infer_item_id_columns(
    df: pd.DataFrame,
    excluded_cols: list[str] | None = None,
) -> list[str]:
    excluded = set(excluded_cols or [])

    preferred = ["model", "lang", "mode", "target", "text"]
    available = [col for col in preferred if col in df.columns and col not in excluded]

    if not available:
        raise ValueError(
            "Could not infer item identity columns. Expected at least one of: "
            "model, lang, mode, target, text"
        )

    return available


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    validate_required_columns(df, REQUIRED_COLUMNS)

    out = df.copy()
    out["retrieved"] = pd.to_numeric(out["retrieved"], errors="coerce")
    out["layer"] = pd.to_numeric(out["layer"], errors="coerce")

    out = out.dropna(subset=["retrieved", "layer"]).copy()
    out["retrieved"] = out["retrieved"].astype(int)
    out["layer"] = out["layer"].astype(int)

    out = out[out["component"] == TARGET_COMPONENT].copy()

    if out.empty:
        raise ValueError(
            f"No rows found for component={TARGET_COMPONENT!r}. "
            "Your parquet appears to contain only debug/non-target components."
        )

    return out


def compute_cumulative_first_retrieval(
    df: pd.DataFrame,
    item_id_cols: list[str],
    group_cols: list[str],
) -> pd.DataFrame:
    required = set(group_cols) | {"layer", "retrieved"}
    validate_required_columns(df, required)

    retrieved_df = df[df["retrieved"] == 1].copy()

    if retrieved_df.empty:
        return pd.DataFrame(
            columns=group_cols
            + ["layer", "first_retrieval_rate", "cumulative_first_retrieval_rate"]
        )

    item_cols = item_id_cols + [col for col in group_cols if col not in item_id_cols]

    first_hits = (
        retrieved_df.groupby(item_cols, dropna=False)["layer"]
        .min()
        .reset_index(name="first_layer")
    )

    totals = (
        df[item_cols]
        .drop_duplicates()
        .groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="total_items")
    )

    first_counts = (
        first_hits.groupby(group_cols + ["first_layer"], dropna=False)
        .size()
        .reset_index(name="count_first_retrieved")
        .rename(columns={"first_layer": "layer"})
    )

    out = first_counts.merge(totals, on=group_cols, how="left")
    out["first_retrieval_rate"] = out["count_first_retrieved"] / out["total_items"]

    all_layers = sorted(df["layer"].dropna().unique())
    groups_df = totals[group_cols].drop_duplicates().copy()
    groups_df["__key"] = 1
    layer_df = pd.DataFrame({"layer": all_layers, "__key": 1})
    full_index = groups_df.merge(layer_df, on="__key").drop(columns="__key")

    out = full_index.merge(
        out[group_cols + ["layer", "first_retrieval_rate"]],
        on=group_cols + ["layer"],
        how="left",
    )

    out["first_retrieval_rate"] = out["first_retrieval_rate"].fillna(0.0)
    out = out.sort_values(group_cols + ["layer"]).reset_index(drop=True)
    out["cumulative_first_retrieval_rate"] = out.groupby(group_cols, dropna=False)[
        "first_retrieval_rate"
    ].cumsum()

    return out


def compute_model_level_cumulative(
    df: pd.DataFrame,
    item_id_cols: list[str],
) -> pd.DataFrame:
    cumulative_df = compute_cumulative_first_retrieval(
        df=df,
        item_id_cols=item_id_cols,
        group_cols=["model", "ablation"],
    )

    if cumulative_df.empty:
        raise ValueError("No cumulative first retrieval data computed.")

    return cumulative_df


def add_relative_layer(df: pd.DataFrame) -> pd.DataFrame:
    required = {"model", "layer"}
    validate_required_columns(df, required)

    max_layer_df = (
        df.groupby("model", dropna=False)["layer"].max().reset_index(name="max_layer")
    )

    out = df.merge(max_layer_df, on="model", how="left")

    if out["max_layer"].isna().any():
        raise ValueError("Missing max_layer for some rows.")

    if (out["max_layer"] <= 0).any():
        bad_models = sorted(
            out.loc[out["max_layer"] <= 0, "model"].dropna().unique().tolist()
        )
        raise ValueError(
            f"Found model(s) with max_layer <= 0. Cannot normalize: {bad_models}"
        )

    out["rel_layer"] = out["layer"] / out["max_layer"]
    return out


def interpolate_to_common_grid(
    df: pd.DataFrame,
    value_col: str,
    num_points: int = 101,
) -> pd.DataFrame:
    required = {"model", "ablation", "rel_layer", value_col}
    validate_required_columns(df, required)

    grid = np.linspace(0.0, 1.0, num_points)
    rows = []

    for (model, ablation), group in df.groupby(["model", "ablation"], dropna=False):
        g = group.sort_values("rel_layer")
        if g.empty:
            continue

        x = g["rel_layer"].to_numpy(dtype=float)
        y = g[value_col].to_numpy(dtype=float)

        unique_x, unique_idx = np.unique(x, return_index=True)
        x = unique_x
        y = y[unique_idx]

        if len(x) == 1:
            y_interp = np.repeat(y[0], len(grid))
        else:
            y_interp = np.interp(grid, x, y)

        for xi, yi in zip(grid, y_interp):
            rows.append(
                {
                    "model": model,
                    "ablation": ablation,
                    "rel_layer": float(xi),
                    value_col: float(yi),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("Interpolation produced empty dataframe.")

    return out


def compute_delta_to_clean(cumulative_df: pd.DataFrame) -> pd.DataFrame:
    required = {"model", "ablation", "rel_layer", "cumulative_first_retrieval_rate"}
    validate_required_columns(cumulative_df, required)

    clean_df = cumulative_df[cumulative_df["ablation"] == CLEAN_ABLATION][
        ["model", "rel_layer", "cumulative_first_retrieval_rate"]
    ].rename(
        columns={
            "cumulative_first_retrieval_rate": "clean_cumulative_first_retrieval_rate"
        }
    )

    compare_df = cumulative_df[
        cumulative_df["ablation"].isin([RANDOM_ABLATION, TARGETED_ABLATION])
    ].copy()

    if compare_df.empty:
        raise ValueError("No random/targeted rows found for delta computation.")

    out = compare_df.merge(clean_df, on=["model", "rel_layer"], how="left")

    if out["clean_cumulative_first_retrieval_rate"].isna().any():
        raise ValueError(
            "Missing clean baseline for some model/rel_layer rows. "
            "Need clean data for all compared rows."
        )

    out["delta_to_clean"] = (
        out["cumulative_first_retrieval_rate"]
        - out["clean_cumulative_first_retrieval_rate"]
    )

    return out


def summarize_random_delta(delta_df: pd.DataFrame) -> pd.DataFrame:
    random_df = delta_df[delta_df["ablation"] == RANDOM_ABLATION].copy()
    if random_df.empty:
        return pd.DataFrame(columns=["rel_layer", "median", "q25", "q75"])

    return (
        random_df.groupby("rel_layer", dropna=False)["delta_to_clean"]
        .agg(
            median="median",
            q25=lambda s: s.quantile(0.25),
            q75=lambda s: s.quantile(0.75),
        )
        .reset_index()
        .sort_values("rel_layer")
    )


def prepare_plot_data(
    df: pd.DataFrame,
    num_points: int = 101,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    item_id_cols = infer_item_id_columns(
        df,
        excluded_cols=["ablation", "component", "layer", "retrieved"],
    )

    cumulative_df = compute_model_level_cumulative(
        df=df,
        item_id_cols=item_id_cols,
    )
    cumulative_df = add_relative_layer(cumulative_df)

    interp_cumulative_df = interpolate_to_common_grid(
        cumulative_df,
        value_col="cumulative_first_retrieval_rate",
        num_points=num_points,
    )

    delta_df = compute_delta_to_clean(interp_cumulative_df)

    clean_df = interp_cumulative_df[interp_cumulative_df["ablation"] == CLEAN_ABLATION]
    final_clean = (
        clean_df.sort_values("rel_layer")
        .groupby("model", dropna=False)
        .tail(1)[["model", "cumulative_first_retrieval_rate"]]
        .rename(columns={"cumulative_first_retrieval_rate": "final_clean"})
        .sort_values("final_clean", ascending=False)
    )
    model_order = final_clean["model"].tolist()

    return interp_cumulative_df, delta_df, model_order


def pool_plot_data_across_languages(
    cumulative_by_lang: list[pd.DataFrame],
    delta_by_lang: list[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if not cumulative_by_lang:
        raise ValueError("No per-language cumulative data to pool.")
    if not delta_by_lang:
        raise ValueError("No per-language delta data to pool.")

    cumulative_df = pd.concat(cumulative_by_lang, ignore_index=True)
    delta_df = pd.concat(delta_by_lang, ignore_index=True)

    pooled_cumulative_df = (
        cumulative_df.groupby(["model", "ablation", "rel_layer"], dropna=False)[
            "cumulative_first_retrieval_rate"
        ]
        .mean()
        .reset_index()
        .sort_values(["model", "ablation", "rel_layer"])
        .reset_index(drop=True)
    )

    pooled_delta_df = (
        delta_df.groupby(["model", "ablation", "rel_layer"], dropna=False)[
            "delta_to_clean"
        ]
        .mean()
        .reset_index()
        .sort_values(["model", "ablation", "rel_layer"])
        .reset_index(drop=True)
    )

    clean_df = pooled_cumulative_df[
        pooled_cumulative_df["ablation"] == CLEAN_ABLATION
    ].copy()

    final_clean = (
        clean_df.sort_values("rel_layer")
        .groupby("model", dropna=False)
        .tail(1)[["model", "cumulative_first_retrieval_rate"]]
        .rename(columns={"cumulative_first_retrieval_rate": "final_clean"})
        .sort_values("final_clean", ascending=False)
    )
    model_order = final_clean["model"].tolist()

    return pooled_cumulative_df, pooled_delta_df, model_order


def make_color_map(model_order: list[str]) -> dict[str, object]:
    cmap = plt.get_cmap("tab10")
    return {model: cmap(i % 10) for i, model in enumerate(model_order)}


def plot_single_language_vertical(
    cumulative_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    model_order: list[str],
    color_map: dict[str, object],
    output_path: Path,
    language_label: str,
    output_format: str,
) -> None:
    clean_df = cumulative_df[cumulative_df["ablation"] == CLEAN_ABLATION].copy()
    targeted_delta_df = delta_df[delta_df["ablation"] == TARGETED_ABLATION].copy()
    random_summary = summarize_random_delta(delta_df)

    if clean_df.empty:
        raise ValueError("No clean rows found.")
    if targeted_delta_df.empty:
        raise ValueError("No targeted rows found for delta plot.")

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(3.4, 5.0),
        sharex=True,
    )
    ax_top, ax_bottom = axes

    legend_handles = []
    legend_labels = []

    for model in model_order:
        subset = clean_df[clean_df["model"] == model].sort_values("rel_layer")
        if subset.empty:
            continue

        (line,) = ax_top.plot(
            subset["rel_layer"],
            subset["cumulative_first_retrieval_rate"],
            linewidth=1.8,
            color=color_map[model],
        )
        legend_handles.append(line)
        legend_labels.append(short_model_name(model))

    for model in model_order:
        subset = targeted_delta_df[targeted_delta_df["model"] == model].sort_values(
            "rel_layer"
        )
        if subset.empty:
            continue

        ax_bottom.plot(
            subset["rel_layer"],
            subset["delta_to_clean"],
            linewidth=1.8,
            color=color_map[model],
        )

    if not random_summary.empty:
        ax_bottom.fill_between(
            random_summary["rel_layer"],
            random_summary["q25"],
            random_summary["q75"],
            alpha=0.12,
            color="0.6",
        )
        ax_bottom.plot(
            random_summary["rel_layer"],
            random_summary["median"],
            linewidth=1.2,
            linestyle="--",
            color="0.4",
        )

    ax_top.set_title(f"{language_label} — Clean", fontsize=10, pad=4)
    ax_top.set_ylabel("Cumulative\nretrieval", fontsize=9)
    ax_top.set_ylim(0.0, 1.0)
    ax_top.set_xlim(0.0, 1.0)
    ax_top.grid(True, alpha=0.25)
    ax_top.tick_params(labelsize=8)

    ax_bottom.set_title(f"{language_label} — Targeted − Clean", fontsize=10, pad=4)
    ax_bottom.set_ylabel("Δ vs clean", fontsize=9)
    ax_bottom.set_xlabel("Relative layer depth", fontsize=9)
    ax_bottom.set_xlim(0.0, 1.0)
    ax_bottom.axhline(0.0, linewidth=0.8, color="black", alpha=0.8)
    ax_bottom.grid(True, alpha=0.25)
    ax_bottom.tick_params(labelsize=8)

    tick_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    tick_labels = ["0.0", "0.25", "0.5", "0.75", "1.0"]
    ax_bottom.set_xticks(tick_values)
    ax_bottom.set_xticklabels(tick_labels)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(2, max(1, len(legend_labels))),
        frameon=False,
        fontsize=8,
        handlelength=2.2,
        columnspacing=1.0,
    )

    fig.text(
        0.5,
        0.10,
        "Dashed gray: random - clean median; shaded: IQR",
        ha="center",
        va="bottom",
        fontsize=7,
        color="0.3",
    )

    fig.subplots_adjust(
        left=0.18,
        right=0.98,
        top=0.90,
        bottom=0.22,
        hspace=0.28,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        format=output_format,
        dpi=300 if output_format == "png" else None,
        bbox_inches="tight",
    )
    plt.close(fig)


def build_output_path(output_dir: Path, suffix: str, output_format: str) -> Path:
    return output_dir / f"layerwise_retrieval_{slugify(suffix)}.{output_format}"


def main(input_parquet: Path, output_dir: Path, output_format: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_parquet)
    df = prepare_dataframe(df)

    if "lang" not in df.columns:
        raise ValueError(
            "No 'lang' column found. This script expects language-specific plots."
        )

    langs = sorted(df["lang"].dropna().unique().tolist())
    if not langs:
        raise ValueError("Column 'lang' exists but has no non-null values.")

    pooled_cumulative_parts = []
    pooled_delta_parts = []

    for lang in langs:
        lang_df = df[df["lang"] == lang].copy()
        if lang_df.empty:
            continue

        cumulative_df, delta_df, model_order = prepare_plot_data(lang_df)
        color_map = make_color_map(model_order)

        output_path = build_output_path(output_dir, f"lang_{lang}", output_format)
        plot_single_language_vertical(
            cumulative_df=cumulative_df,
            delta_df=delta_df,
            model_order=model_order,
            color_map=color_map,
            output_path=output_path,
            language_label=str(lang).title(),
            output_format=output_format,
        )
        print(f"Saved plot to: {output_path}")

        pooled_cumulative_parts.append(cumulative_df.assign(lang=lang))
        pooled_delta_parts.append(delta_df.assign(lang=lang))

    pooled_cumulative_df, pooled_delta_df, pooled_model_order = (
        pool_plot_data_across_languages(
            pooled_cumulative_parts,
            pooled_delta_parts,
        )
    )
    pooled_color_map = make_color_map(pooled_model_order)

    pooled_output_path = build_output_path(output_dir, "pooled_mean", output_format)
    plot_single_language_vertical(
        cumulative_df=pooled_cumulative_df,
        delta_df=pooled_delta_df,
        model_order=pooled_model_order,
        color_map=pooled_color_map,
        output_path=pooled_output_path,
        language_label="Pooled mean",
        output_format=output_format,
    )
    print(f"Saved plot to: {pooled_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_parquet", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--format",
        dest="output_format",
        choices=["png", "pdf"],
        default="pdf",
        help="Output file format (default: pdf)",
    )
    args = parser.parse_args()

    main(args.input_parquet, args.output_dir, args.output_format)
