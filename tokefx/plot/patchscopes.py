import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TARGET_COMPONENT = "layer_out"
REQUIRED_COLUMNS = {"model", "layer", "component", "retrieved", "ablation"}


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


def validate_required_columns(df: pd.DataFrame, required_columns: set[str]) -> None:
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def non_null_unique_sorted(series: pd.Series) -> list:
    return sorted(series.dropna().unique())


def infer_item_id_columns(
    df: pd.DataFrame,
    excluded_cols: list[str] | None = None,
) -> list[str]:
    excluded = set(excluded_cols or [])

    # These should identify one "item" whose first retrieval layer we care about.
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

    # Hard filter to the only component that matters now.
    out = out[out["component"] == TARGET_COMPONENT].copy()

    if out.empty:
        raise ValueError(
            f"No rows found for component={TARGET_COMPONENT!r}. "
            "Your parquet appears to contain only debug/non-target components."
        )

    return out


def make_output_dirs(base_dir: Path) -> dict[str, Path]:
    dirs = {
        "cumulative_all_languages": base_dir
        / "cumulative_first_retrieval"
        / "all_languages",
        "cumulative_by_language": base_dir
        / "cumulative_first_retrieval"
        / "by_language",
        "retrieved_rate_all_languages": base_dir / "retrieved_rate" / "all_languages",
        "retrieved_rate_by_language": base_dir / "retrieved_rate" / "by_language",
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def compute_retrieved_rate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    required = set(group_cols) | {"retrieved"}
    validate_required_columns(df, required)

    return (
        df.groupby(group_cols, dropna=False)["retrieved"]
        .mean()
        .reset_index(name="retrieved_rate")
    )


def compute_cumulative_first_retrieval(
    df: pd.DataFrame,
    item_id_cols: list[str],
    group_cols: list[str],
) -> pd.DataFrame:
    """
    Returns one row per (group_cols + layer) with:
      - first_retrieval_rate
      - cumulative_first_retrieval_rate
    """
    required = set(group_cols) | {"layer", "retrieved"}
    validate_required_columns(df, required)

    retrieved_df = df[df["retrieved"] == 1].copy()

    if retrieved_df.empty:
        return pd.DataFrame(
            columns=group_cols
            + ["layer", "first_retrieval_rate", "cumulative_first_retrieval_rate"]
        )

    # item_cols identifies a unique retrieval candidate within a group.
    item_cols = item_id_cols + [col for col in group_cols if col not in item_id_cols]

    # First layer at which each item is retrieved.
    first_hits = (
        retrieved_df.groupby(item_cols, dropna=False)["layer"]
        .min()
        .reset_index(name="first_layer")
    )

    # Denominator: number of unique items in each group.
    totals = (
        df[item_cols]
        .drop_duplicates()
        .groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="total_items")
    )

    # Numerator: how many items first appear at each layer.
    first_counts = (
        first_hits.groupby(group_cols + ["first_layer"], dropna=False)
        .size()
        .reset_index(name="count_first_retrieved")
        .rename(columns={"first_layer": "layer"})
    )

    out = first_counts.merge(totals, on=group_cols, how="left")
    out["first_retrieval_rate"] = out["count_first_retrieved"] / out["total_items"]

    # Fill in missing layers with zero so cumulative curves are continuous.
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


def plot_cumulative_first_retrieval_all_languages(
    df: pd.DataFrame,
    item_id_cols: list[str],
    output_dir: Path,
) -> None:
    """
    One plot per model.
    X-axis: layer
    Y-axis: cumulative first retrieval rate
    Lines: ablation types
    Languages: pooled together
    """
    cumulative_df = compute_cumulative_first_retrieval(
        df=df,
        item_id_cols=item_id_cols,
        group_cols=["model", "ablation"],
    )

    if cumulative_df.empty:
        print("No cumulative first retrieval data to plot (all languages).")
        return

    for model in non_null_unique_sorted(cumulative_df["model"]):
        model_df = cumulative_df[cumulative_df["model"] == model]
        if model_df.empty:
            continue

        plt.figure(figsize=(10, 6))

        for ablation in non_null_unique_sorted(model_df["ablation"]):
            ablation_df = model_df[model_df["ablation"] == ablation].sort_values(
                "layer"
            )
            if ablation_df.empty:
                continue

            plt.plot(
                ablation_df["layer"],
                ablation_df["cumulative_first_retrieval_rate"],
                marker="o",
                label=str(ablation),
            )

        plt.xlabel("Layer")
        plt.ylabel("Cumulative First Retrieval Rate")
        plt.title(
            f"Cumulative First Retrieval by Ablation ({model}, pooled across languages, component={TARGET_COMPONENT})"
        )
        plt.legend(title="Ablation")
        plt.tight_layout()

        output_path = (
            output_dir
            / f"cumulative_first_retrieval_all_languages_{sanitize_filename(model)}.png"
        )
        plt.savefig(output_path, dpi=200)
        plt.close()


def plot_cumulative_first_retrieval_by_language(
    df: pd.DataFrame,
    item_id_cols: list[str],
    output_dir: Path,
) -> None:
    """
    One plot per (model, language).
    X-axis: layer
    Y-axis: cumulative first retrieval rate
    Lines: ablation types
    """
    if "lang" not in df.columns:
        print("No 'lang' column found. Skipping per-language cumulative plots.")
        return

    cumulative_df = compute_cumulative_first_retrieval(
        df=df,
        item_id_cols=item_id_cols,
        group_cols=["model", "lang", "ablation"],
    )

    if cumulative_df.empty:
        print("No cumulative first retrieval data to plot (by language).")
        return

    for model in non_null_unique_sorted(cumulative_df["model"]):
        model_df = cumulative_df[cumulative_df["model"] == model]

        for lang in non_null_unique_sorted(model_df["lang"]):
            subset_df = model_df[model_df["lang"] == lang]
            if subset_df.empty:
                continue

            plt.figure(figsize=(10, 6))

            for ablation in non_null_unique_sorted(subset_df["ablation"]):
                ablation_df = subset_df[subset_df["ablation"] == ablation].sort_values(
                    "layer"
                )
                if ablation_df.empty:
                    continue

                plt.plot(
                    ablation_df["layer"],
                    ablation_df["cumulative_first_retrieval_rate"],
                    marker="o",
                    label=str(ablation),
                )

            plt.xlabel("Layer")
            plt.ylabel("Cumulative First Retrieval Rate")
            plt.title(
                f"Cumulative First Retrieval by Ablation ({model}, lang={lang}, component={TARGET_COMPONENT})"
            )
            plt.legend(title="Ablation")
            plt.tight_layout()

            output_path = (
                output_dir
                / f"cumulative_first_retrieval_{sanitize_filename(model)}_{sanitize_filename(lang)}.png"
            )
            plt.savefig(output_path, dpi=200)
            plt.close()


def plot_retrieved_rate_all_languages(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    One plot per model.
    X-axis: layer
    Y-axis: retrieved rate
    Lines: ablation types
    Languages: pooled together
    """
    retrieved_df = compute_retrieved_rate(
        df=df,
        group_cols=["model", "ablation", "layer"],
    )

    if retrieved_df.empty:
        print("No retrieved-rate data to plot (all languages).")
        return

    for model in non_null_unique_sorted(retrieved_df["model"]):
        model_df = retrieved_df[retrieved_df["model"] == model]
        if model_df.empty:
            continue

        plt.figure(figsize=(10, 6))

        for ablation in non_null_unique_sorted(model_df["ablation"]):
            ablation_df = model_df[model_df["ablation"] == ablation].sort_values(
                "layer"
            )
            if ablation_df.empty:
                continue

            plt.plot(
                ablation_df["layer"],
                ablation_df["retrieved_rate"],
                marker="o",
                label=str(ablation),
            )

        plt.xlabel("Layer")
        plt.ylabel("Retrieved Rate")
        plt.title(
            f"Retrieved Rate Across Layers by Ablation ({model}, pooled across languages, component={TARGET_COMPONENT})"
        )
        plt.legend(title="Ablation")
        plt.tight_layout()

        output_path = (
            output_dir / f"retrieved_rate_all_languages_{sanitize_filename(model)}.png"
        )
        plt.savefig(output_path, dpi=200)
        plt.close()


def plot_retrieved_rate_by_language(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    One plot per (model, language).
    X-axis: layer
    Y-axis: retrieved rate
    Lines: ablation types
    """
    if "lang" not in df.columns:
        print("No 'lang' column found. Skipping per-language retrieved-rate plots.")
        return

    retrieved_df = compute_retrieved_rate(
        df=df,
        group_cols=["model", "lang", "ablation", "layer"],
    )

    if retrieved_df.empty:
        print("No retrieved-rate data to plot (by language).")
        return

    for model in non_null_unique_sorted(retrieved_df["model"]):
        model_df = retrieved_df[retrieved_df["model"] == model]

        for lang in non_null_unique_sorted(model_df["lang"]):
            subset_df = model_df[model_df["lang"] == lang]
            if subset_df.empty:
                continue

            plt.figure(figsize=(10, 6))

            for ablation in non_null_unique_sorted(subset_df["ablation"]):
                ablation_df = subset_df[subset_df["ablation"] == ablation].sort_values(
                    "layer"
                )
                if ablation_df.empty:
                    continue

                plt.plot(
                    ablation_df["layer"],
                    ablation_df["retrieved_rate"],
                    marker="o",
                    label=str(ablation),
                )

            plt.xlabel("Layer")
            plt.ylabel("Retrieved Rate")
            plt.title(
                f"Retrieved Rate Across Layers by Ablation ({model}, lang={lang}, component={TARGET_COMPONENT})"
            )
            plt.legend(title="Ablation")
            plt.tight_layout()

            output_path = (
                output_dir
                / f"retrieved_rate_{sanitize_filename(model)}_{sanitize_filename(lang)}.png"
            )
            plt.savefig(output_path, dpi=200)
            plt.close()


def main(input_parquet: Path, output_dir: Path) -> None:
    df = pd.read_parquet(input_parquet)
    df = prepare_dataframe(df)

    item_id_cols = infer_item_id_columns(
        df,
        excluded_cols=["ablation", "component", "layer", "retrieved"],
    )

    dirs = make_output_dirs(output_dir)

    plot_cumulative_first_retrieval_all_languages(
        df=df,
        item_id_cols=item_id_cols,
        output_dir=dirs["cumulative_all_languages"],
    )

    plot_cumulative_first_retrieval_by_language(
        df=df,
        item_id_cols=item_id_cols,
        output_dir=dirs["cumulative_by_language"],
    )

    plot_retrieved_rate_all_languages(
        df=df,
        output_dir=dirs["retrieved_rate_all_languages"],
    )

    plot_retrieved_rate_by_language(
        df=df,
        output_dir=dirs["retrieved_rate_by_language"],
    )

    print(f"Plots saved under {output_dir}")
    print(f"Filtered to component={TARGET_COMPONENT!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_parquet", type=Path)
    parser.add_argument("output_dir", type=Path)

    args = parser.parse_args()
    main(args.input_parquet, args.output_dir)
