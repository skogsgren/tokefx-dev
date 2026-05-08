import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TARGET_COMPONENT = "layer_out"
TARGET_ABLATION = "clean"

REQUIRED_COLUMNS = {
    "model",
    "layer",
    "component",
    "retrieved",
    "ablation",
}


COLORS = ["#000000", "#E69F00", "#56B4E9", "#009E73"]


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
        raise ValueError(f"No rows found for component={TARGET_COMPONENT!r}.")

    out = out[out["ablation"] == TARGET_ABLATION].copy()
    if out.empty:
        raise ValueError(f"No rows found for ablation={TARGET_ABLATION!r}.")

    return out


def add_relative_depth(df: pd.DataFrame) -> pd.DataFrame:
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
    return compute_retrieved_rate(
        df=df,
        group_cols=["model", "relative_depth"],
    )


def compute_language_specific_rates(df: pd.DataFrame) -> pd.DataFrame:
    if "lang" not in df.columns:
        return pd.DataFrame(
            columns=["lang", "model", "relative_depth", "retrieved_rate"]
        )

    return compute_retrieved_rate(
        df=df,
        group_cols=["lang", "model", "relative_depth"],
    )


def get_model_color_map(models: list[str]) -> dict[str, str]:
    return {model: COLORS[i % len(COLORS)] for i, model in enumerate(models)}


def plot_clean_models(plot_df: pd.DataFrame, output_path: Path) -> None:
    if plot_df.empty:
        return

    models = non_null_unique_sorted(plot_df["model"])
    model_colors = get_model_color_map(models)

    short_model_names = {model: shorten_model_name(model) for model in models}

    fig, ax = plt.subplots(figsize=(3.2, 2.6))

    for model in models:
        line_df = plot_df[plot_df["model"] == model].sort_values("relative_depth")

        if line_df.empty:
            continue

        ax.plot(
            line_df["relative_depth"],
            line_df["retrieved_rate"],
            color=model_colors[model],
            linewidth=3,
            label=short_model_names[model],
        )

    ax.set_xlabel("Relative Depth", fontsize=9)
    ax.set_ylabel("Retrieved Rate", fontsize=9)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)

    ax.tick_params(axis="both", labelsize=7)

    ax.grid(
        True,
        alpha=0.18,
        linewidth=0.5,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
        frameon=False,
        fontsize=7,
        handlelength=2.0,
        columnspacing=1.0,
        handletextpad=0.5,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(
        left=0.18,
        right=0.98,
        top=0.98,
        bottom=0.32,
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_pooled(df: pd.DataFrame, output_dir: Path) -> None:
    pooled_rates = compute_pooled_rates(df)

    if pooled_rates.empty:
        print("No pooled retrieved-rate data to plot.")
        return

    plot_clean_models(
        plot_df=pooled_rates,
        output_path=output_dir / "retrieved_rate_clean_pooled.pdf",
    )


def plot_by_language(df: pd.DataFrame, output_dir: Path) -> None:
    lang_rates = compute_language_specific_rates(df)
    for lang in non_null_unique_sorted(lang_rates["lang"]):
        lang_df = lang_rates[lang_rates["lang"] == lang].copy()

        if lang_df.empty:
            continue

        plot_clean_models(
            plot_df=lang_df,
            output_path=output_dir / f"retrieved_rate_clean_{lang}.pdf",
        )


def main(input_parquet: Path, output_dir: Path) -> None:
    df = pd.read_parquet(input_parquet)

    df = prepare_dataframe(df)
    df = add_relative_depth(df)

    plot_pooled(df=df, output_dir=output_dir)
    plot_by_language(df=df, output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Plot clean-condition retrieved rate across relative depth. "
            "Produces one pooled plot and one plot per language if a 'lang' "
            "column exists."
        )
    )

    parser.add_argument("input_parquet", type=Path)
    parser.add_argument("output_dir", type=Path)

    args = parser.parse_args()

    main(
        input_parquet=args.input_parquet,
        output_dir=args.output_dir,
    )
