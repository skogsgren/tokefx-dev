import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TARGET_COMPONENT = "layer_out"
REQUIRED_COLUMNS = {"model", "layer", "component", "retrieved", "ablation"}
TARGET_ABLATIONS = ["clean", "targeted"]

LINESTYLES = {
    "clean": "-",
    "targeted": "--",
}


def sanitize_filename(text: object) -> str:
    return (
        str(text)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace("=", "_")
        .replace(":", "_")
        .replace(",", "_")
    )


def shorten_model_name(model_name: object) -> str:
    return str(model_name).split("/")[-1]


def validate_required_columns(df: pd.DataFrame, required_columns: set[str]) -> None:
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def non_null_unique_sorted(series: pd.Series) -> list:
    return sorted(series.dropna().unique())


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
            "Your parquet may only contain debug/non-target components."
        )

    out = out[out["ablation"].isin(TARGET_ABLATIONS)].copy()
    if out.empty:
        raise ValueError(
            f"No rows found for ablations in {TARGET_ABLATIONS!r} "
            f"after filtering to component={TARGET_COMPONENT!r}."
        )

    return out


def add_relative_depth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds relative_depth in [0, 1] within each model using min-max normalization.
    """
    validate_required_columns(df, {"model", "layer"})

    out = df.copy()

    def normalize(series: pd.Series) -> pd.Series:
        min_layer = series.min()
        max_layer = series.max()
        if max_layer == min_layer:
            return pd.Series(0.0, index=series.index)
        return (series - min_layer) / (max_layer - min_layer)

    out["relative_depth"] = out.groupby("model", dropna=False)["layer"].transform(
        normalize
    )
    return out


def compute_retrieved_rate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    required = set(group_cols) | {"retrieved"}
    validate_required_columns(df, required)

    return (
        df.groupby(group_cols, dropna=False)["retrieved"]
        .mean()
        .reset_index(name="retrieved_rate")
    )


def compute_pooled_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one row per (model, ablation, relative_depth) with retrieved_rate.
    Assumes any balancing has already been handled upstream.
    """
    return compute_retrieved_rate(
        df=df,
        group_cols=["model", "ablation", "relative_depth"],
    )


def compute_language_specific_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns one row per (lang, model, ablation, relative_depth) with retrieved_rate.
    """
    if "lang" not in df.columns:
        return pd.DataFrame(
            columns=["lang", "model", "ablation", "relative_depth", "retrieved_rate"]
        )

    return compute_retrieved_rate(
        df=df,
        group_cols=["lang", "model", "ablation", "relative_depth"],
    )


def get_model_color_map(models: list[str]) -> dict[str, str]:
    """
    Assign one consistent matplotlib default-cycle color per model.
    """
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {model: color_cycle[i % len(color_cycle)] for i, model in enumerate(models)}


def plot_model_pairs(
    plot_df: pd.DataFrame,
    output_path: Path,
) -> None:
    if plot_df.empty:
        return

    models = non_null_unique_sorted(plot_df["model"])
    model_colors = get_model_color_map(models)
    short_model_names = {model: shorten_model_name(model) for model in models}

    fig, ax = plt.subplots(figsize=(4.8, 4.8))

    for model in models:
        model_df = plot_df[plot_df["model"] == model].copy()
        color = model_colors[model]

        for ablation in TARGET_ABLATIONS:
            line_df = model_df[model_df["ablation"] == ablation].sort_values(
                "relative_depth"
            )
            if line_df.empty:
                continue

            # Only label clean lines so the legend describes colors/models only.
            label = short_model_names[model] if ablation == "clean" else None

            ax.plot(
                line_df["relative_depth"],
                line_df["retrieved_rate"],
                linewidth=1.6,
                linestyle=LINESTYLES[ablation],
                color=color,
                label=label,
            )

    ax.set_xlabel("Relative Depth")
    ax.set_ylabel("Retrieved Rate")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, alpha=0.3)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=1,
        frameon=False,
        fontsize=9,
        title=None,
        handlelength=2.4,
        handletextpad=0.6,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(
        left=0.16,
        right=0.97,
        top=0.97,
        bottom=0.32,
    )

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_pooled(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    pooled_rates = compute_pooled_rates(df)

    if pooled_rates.empty:
        print("No pooled retrieved-rate data to plot.")
        return

    plot_model_pairs(
        plot_df=pooled_rates,
        output_path=output_dir / "retrieved_rate_model_comparison_pooled.pdf",
    )


def plot_by_language(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    if "lang" not in df.columns:
        print("No 'lang' column found. Skipping language-specific plots.")
        return

    lang_rates = compute_language_specific_rates(df)

    if lang_rates.empty:
        print("No language-specific retrieved-rate data to plot.")
        return

    for lang in non_null_unique_sorted(lang_rates["lang"]):
        lang_df = lang_rates[lang_rates["lang"] == lang].copy()
        if lang_df.empty:
            continue

        plot_model_pairs(
            plot_df=lang_df,
            output_path=(
                output_dir
                / f"retrieved_rate_model_comparison_{sanitize_filename(lang)}.pdf"
            ),
        )


def make_output_dirs(base_dir: Path) -> dict[str, Path]:
    dirs = {
        "pooled": base_dir / "retrieved_rate_model_comparison" / "pooled",
        "by_language": base_dir / "retrieved_rate_model_comparison" / "by_language",
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def main(input_parquet: Path, output_dir: Path) -> None:
    df = pd.read_parquet(input_parquet)
    df = prepare_dataframe(df)
    df = add_relative_depth(df)

    dirs = make_output_dirs(output_dir)

    plot_pooled(
        df=df,
        output_dir=dirs["pooled"],
    )

    plot_by_language(
        df=df,
        output_dir=dirs["by_language"],
    )

    print(f"Plots saved under {output_dir}")
    print(f"Filtered to component={TARGET_COMPONENT!r}")
    print(f"Filtered to ablations={TARGET_ABLATIONS!r}")
    print("X-axis uses relative depth in [0, 1].")
    print("Legend labels correspond to model colors only.")
    print("Use the caption to explain linestyle: clean=solid, targeted=dashed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Plot retrieved rate across relative depth, comparing clean vs targeted "
            "across models. Produces one pooled plot and one plot per language."
        )
    )
    parser.add_argument("input_parquet", type=Path)
    parser.add_argument("output_dir", type=Path)

    args = parser.parse_args()

    main(
        input_parquet=args.input_parquet,
        output_dir=args.output_dir,
    )
