#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET_COMPONENT = "layer_out"

CLEAN_ABLATION = "clean"
RANDOM_ABLATION = "random"
TARGETED_ABLATION = "targeted"

REQUIRED_COLUMNS = {"model", "layer", "component", "retrieved", "ablation"}
COLORS = ["#000000", "#E69F00", "#56B4E9", "#009E73"]

LINESTYLES = [
    "solid",
    "dashed",
    "dashdot",
    "dotted",
]


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


def format_level(level: float) -> str:
    return f"{level * 100:g}%"


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
        raise ValueError(f"No rows found for component={TARGET_COMPONENT!r}.")

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


def add_relative_layer(df: pd.DataFrame) -> pd.DataFrame:
    validate_required_columns(df, {"model", "layer"})

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
        raise ValueError(f"Found model(s) with max_layer <= 0: {bad_models}")

    out["rel_layer"] = out["layer"] / out["max_layer"]
    return out


def interpolate_to_common_grid(
    df: pd.DataFrame,
    value_col: str,
    group_cols: list[str],
    num_points: int,
) -> pd.DataFrame:
    validate_required_columns(df, set(group_cols) | {"rel_layer", value_col})

    grid = np.linspace(0.0, 1.0, num_points)
    rows = []

    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        key_data = dict(zip(group_cols, keys))
        g = group.sort_values("rel_layer")

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
                    **key_data,
                    "rel_layer": float(xi),
                    value_col: float(yi),
                }
            )

    out = pd.DataFrame(rows)

    if out.empty:
        raise ValueError("Interpolation produced empty dataframe.")

    return out


def compute_delta_to_clean(cumulative_df: pd.DataFrame) -> pd.DataFrame:
    validate_required_columns(
        cumulative_df,
        {"model", "ablation", "rel_layer", "cumulative_first_retrieval_rate"},
    )

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

    if clean_df.empty:
        raise ValueError("No clean rows found.")
    if compare_df.empty:
        raise ValueError("No random/targeted rows found.")

    out = compare_df.merge(clean_df, on=["model", "rel_layer"], how="left")

    if out["clean_cumulative_first_retrieval_rate"].isna().any():
        raise ValueError("Missing clean baseline for some model/rel_layer rows.")

    out["delta_to_clean"] = (
        out["cumulative_first_retrieval_rate"]
        - out["clean_cumulative_first_retrieval_rate"]
    )

    return out


def prepare_plot_data_for_subset(
    df: pd.DataFrame,
    num_points: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    item_id_cols = infer_item_id_columns(
        df,
        excluded_cols=["ablation", "component", "layer", "retrieved"],
    )

    cumulative_df = compute_cumulative_first_retrieval(
        df=df,
        item_id_cols=item_id_cols,
        group_cols=["model", "ablation"],
    )

    cumulative_df = add_relative_layer(cumulative_df)

    cumulative_df = interpolate_to_common_grid(
        cumulative_df,
        value_col="cumulative_first_retrieval_rate",
        group_cols=["model", "ablation"],
        num_points=num_points,
    )

    delta_df = compute_delta_to_clean(cumulative_df)

    clean_df = cumulative_df[cumulative_df["ablation"] == CLEAN_ABLATION].copy()

    model_order = (
        clean_df.sort_values("rel_layer")
        .groupby("model", dropna=False)
        .tail(1)
        .sort_values("cumulative_first_retrieval_rate", ascending=False)["model"]
        .tolist()
    )

    return cumulative_df, delta_df, model_order


def prepare_all_plot_data(
    all_df: pd.DataFrame,
    levels: list[float],
    num_points: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    cumulative_parts = []
    delta_parts = []
    model_order_parts = []

    for level in levels:
        level_df = all_df[np.isclose(all_df["ablation_level"], level)].copy()

        if level_df.empty:
            continue

        cumulative_df, delta_df, model_order = prepare_plot_data_for_subset(
            level_df,
            num_points=num_points,
        )

        cumulative_parts.append(cumulative_df.assign(ablation_level=level))
        delta_parts.append(delta_df.assign(ablation_level=level))
        model_order_parts.extend(model_order)

    if not cumulative_parts or not delta_parts:
        raise ValueError("No plot data computed.")

    all_cumulative = pd.concat(cumulative_parts, ignore_index=True)
    all_delta = pd.concat(delta_parts, ignore_index=True)

    model_order = []
    seen = set()

    for model in model_order_parts:
        if model not in seen:
            seen.add(model)
            model_order.append(model)

    return all_cumulative, all_delta, model_order


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


def make_style_map(model_order: list[str]) -> dict[str, dict[str, object]]:
    style_map = {}

    for i, model in enumerate(model_order):
        color = COLORS[i % len(COLORS)]
        linestyle = LINESTYLES[(i // len(COLORS)) % len(LINESTYLES)]
        style_map[model] = {
            "color": color,
            "linestyle": linestyle,
        }

    return style_map


def compute_global_ylim(values: pd.Series) -> tuple[float, float]:
    vals = values.dropna().to_numpy(dtype=float)

    if len(vals) == 0:
        return -0.1, 0.02

    lo = min(float(vals.min()), 0.0)
    hi = max(float(vals.max()), 0.0)

    span = hi - lo
    pad = 0.06 * span if span > 0 else 0.02

    return lo - pad, hi + pad


def representative_clean_cumulative(cumulative_df: pd.DataFrame) -> pd.DataFrame:
    clean_df = cumulative_df[cumulative_df["ablation"] == CLEAN_ABLATION].copy()

    if clean_df.empty:
        raise ValueError("No clean rows found.")

    level_counts = (
        clean_df.groupby("ablation_level", dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["n", "ablation_level"], ascending=[False, True])
    )

    representative_level = float(level_counts.iloc[0]["ablation_level"])

    return clean_df[np.isclose(clean_df["ablation_level"], representative_level)].copy()


def plot_dose_response_grid(
    cumulative_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    levels: list[float],
    model_order: list[str],
    style_map: dict[str, dict[str, object]],
    output_path: Path,
    output_format: str,
    title: str,
) -> None:
    clean_df = representative_clean_cumulative(cumulative_df)
    targeted_df = delta_df[delta_df["ablation"] == TARGETED_ABLATION].copy()

    if targeted_df.empty:
        raise ValueError("No targeted rows found for delta plot.")

    fig = plt.figure(figsize=(2.85 * len(levels), 4.75))

    gs = fig.add_gridspec(
        nrows=2,
        ncols=len(levels),
        height_ratios=[0.9, 1.25],
        hspace=0.42,
        wspace=0.28,
    )

    ax_clean = fig.add_subplot(gs[0, :])
    bottom_axes = [fig.add_subplot(gs[1, col]) for col in range(len(levels))]

    delta_y_min, delta_y_max = compute_global_ylim(delta_df["delta_to_clean"])

    legend_handles = []
    legend_labels = []

    common_xticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    common_xtick_labels = ["0.0", "0.25", "0.5", "0.75", "1.0"]

    for model in model_order:
        subset = clean_df[clean_df["model"] == model].sort_values("rel_layer")

        if subset.empty:
            continue

        style = style_map[model]

        (line,) = ax_clean.plot(
            subset["rel_layer"],
            subset["cumulative_first_retrieval_rate"],
            linewidth=3.25,
            color=style["color"],
            linestyle=style["linestyle"],
        )

        legend_handles.append(line)
        legend_labels.append(short_model_name(model))

    ax_clean.set_title("Clean cumulative retrieval", fontsize=10, pad=4)
    ax_clean.set_xlabel("Relative layer depth", fontsize=9, labelpad=6)
    ax_clean.set_ylabel("Cumulative\nretrieval", fontsize=9)
    ax_clean.set_xlim(0.0, 1.0)
    ax_clean.set_ylim(0.0, 1.0)
    ax_clean.set_xticks(common_xticks)
    ax_clean.set_xticklabels(common_xtick_labels)
    ax_clean.grid(True, alpha=0.25)
    ax_clean.tick_params(labelsize=8)

    for col_idx, (ax, level) in enumerate(zip(bottom_axes, levels)):
        level_delta = delta_df[np.isclose(delta_df["ablation_level"], level)].copy()

        targeted_level = level_delta[
            level_delta["ablation"] == TARGETED_ABLATION
        ].copy()

        random_summary = summarize_random_delta(level_delta)

        for model in model_order:
            subset = targeted_level[targeted_level["model"] == model].sort_values(
                "rel_layer"
            )

            if subset.empty:
                continue

            style = style_map[model]

            ax.plot(
                subset["rel_layer"],
                subset["delta_to_clean"],
                linewidth=3.25,
                color=style["color"],
                linestyle=style["linestyle"],
            )

        if not random_summary.empty:
            x = random_summary["rel_layer"].to_numpy(dtype=float)
            q25 = random_summary["q25"].to_numpy(dtype=float)
            q75 = random_summary["q75"].to_numpy(dtype=float)
            median = random_summary["median"].to_numpy(dtype=float)

            ax.fill_between(
                x,
                q25,
                q75,
                color="0.65",
                alpha=0.18,
                linewidth=0,
            )

            ax.plot(
                x,
                median,
                color="0.25",
                linestyle=(0, (2, 2)),
                linewidth=1.15,
            )

        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.85)
        ax.set_title(f"{format_level(level)} targeted − clean", fontsize=10, pad=4)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(delta_y_min, delta_y_max)
        ax.set_xticks(common_xticks)
        ax.set_xticklabels(common_xtick_labels)
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("Relative layer depth", fontsize=9)

        if col_idx == 0:
            ax.set_ylabel("Δ cumulative retrieval\nvs clean", fontsize=9)
        else:
            ax.tick_params(labelleft=False)

    fig.suptitle(title, fontsize=11, y=0.985)

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.008),
            ncol=min(4, max(1, len(legend_labels))),
            frameon=False,
            fontsize=9,
            handlelength=2.4,
            columnspacing=1.0,
        )

    fig.text(
        0.5,
        0.090,
        "Bottom row: targeted − clean. Dashed gray: random − clean median; shaded: IQR.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="0.3",
    )

    fig.subplots_adjust(
        left=0.09,
        right=0.995,
        top=0.912,
        bottom=0.20,
    )

    # Must happen after subplots_adjust, otherwise matplotlib stretches it again.
    clean_pos = ax_clean.get_position()
    clean_width = clean_pos.width * 0.66
    clean_left = clean_pos.x0 + (clean_pos.width - clean_width) / 2.0
    ax_clean.set_position(
        [
            clean_left,
            clean_pos.y0,
            clean_width,
            clean_pos.height,
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        output_path,
        format=output_format,
        dpi=300 if output_format == "png" else None,
        bbox_inches="tight",
    )

    plt.close(fig)


def parse_ablation_specs(raw_specs: list[list[str]]) -> list[tuple[float, Path]]:
    specs = []

    for raw_level, raw_path in raw_specs:
        try:
            level = float(raw_level)
        except ValueError as exc:
            raise ValueError(f"Invalid ablation level: {raw_level!r}") from exc

        if level <= 0:
            raise ValueError(f"Ablation level must be positive: {raw_level!r}")

        path = Path(raw_path)

        if not path.exists():
            raise ValueError(f"Parquet file does not exist: {path}")

        specs.append((level, path))

    specs = sorted(specs, key=lambda pair: pair[0])
    levels = [level for level, _ in specs]

    if len(levels) != len(set(levels)):
        raise ValueError(f"Duplicate ablation levels provided: {levels}")

    return specs


def read_level_parquets(ablation_specs: list[tuple[float, Path]]) -> pd.DataFrame:
    parts = []

    for level, path in ablation_specs:
        df = pd.read_parquet(path)
        df = prepare_dataframe(df)
        df["ablation_level"] = float(level)
        parts.append(df)

    if not parts:
        raise ValueError("No ablation parquet files provided.")

    return pd.concat(parts, ignore_index=True)


def build_output_path(output_dir: Path, suffix: str, output_format: str) -> Path:
    return output_dir / f"dose_response_grid_{slugify(suffix)}.{output_format}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot dose-response ablation grids. Each ablation level is supplied "
            "as a separate parquet. The output has one clean cumulative plot on "
            "top and one targeted-minus-clean delta panel per ablation level below."
        )
    )

    parser.add_argument(
        "--abl",
        nargs=2,
        action="append",
        required=True,
        metavar=("LEVEL", "PARQUET"),
        help=(
            "Ablation level and parquet path. Repeat this option, e.g. "
            "--abl 0.01 abl_001.parquet --abl 0.03 abl_003.parquet "
            "--abl 0.05 abl_005.parquet."
        ),
    )

    parser.add_argument("output_dir", type=Path)

    parser.add_argument(
        "--format",
        dest="output_format",
        choices=["png", "pdf"],
        default="pdf",
    )

    parser.add_argument(
        "--num-points",
        type=int,
        default=101,
        help="Number of interpolation points over relative depth.",
    )

    parser.add_argument(
        "--pooled-only",
        action="store_true",
        help="Only write the pooled plot, not per-language plots.",
    )

    args = parser.parse_args()

    ablation_specs = parse_ablation_specs(args.abl)
    levels = [level for level, _ in ablation_specs]

    all_df = read_level_parquets(ablation_specs)

    pooled_cumulative_df, pooled_delta_df, model_order = prepare_all_plot_data(
        all_df=all_df,
        levels=levels,
        num_points=args.num_points,
    )

    style_map = make_style_map(model_order)

    pooled_output = build_output_path(
        args.output_dir,
        "pooled",
        args.output_format,
    )

    plot_dose_response_grid(
        cumulative_df=pooled_cumulative_df,
        delta_df=pooled_delta_df,
        levels=levels,
        model_order=model_order,
        style_map=style_map,
        output_path=pooled_output,
        output_format=args.output_format,
        title="Effect of targeted ablation",
    )

    print(f"Saved plot to: {pooled_output}")

    if not args.pooled_only and "lang" in all_df.columns:
        for lang in sorted(all_df["lang"].dropna().unique().tolist()):
            lang_df = all_df[all_df["lang"] == lang].copy()

            if lang_df.empty:
                continue

            lang_cumulative_df, lang_delta_df, lang_model_order = prepare_all_plot_data(
                all_df=lang_df,
                levels=levels,
                num_points=args.num_points,
            )

            lang_style_map = {
                model: style_map[model]
                for model in lang_model_order
                if model in style_map
            }

            lang_output = build_output_path(
                args.output_dir,
                f"lang_{lang}",
                args.output_format,
            )

            plot_dose_response_grid(
                cumulative_df=lang_cumulative_df,
                delta_df=lang_delta_df,
                levels=levels,
                model_order=lang_model_order,
                style_map=lang_style_map,
                output_path=lang_output,
                output_format=args.output_format,
                title=f"Dose-response of targeted ablation — {str(lang).title()}",
            )

            print(f"Saved plot to: {lang_output}")


if __name__ == "__main__":
    main()
