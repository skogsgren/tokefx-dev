#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

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


def latex_escape(value: object) -> str:
    s = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in s)


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

        rows.extend(
            {
                "model": model,
                "ablation": ablation,
                "rel_layer": float(xi),
                value_col: float(yi),
            }
            for xi, yi in zip(grid, y_interp)
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


def prepare_plot_data(
    df: pd.DataFrame,
    num_points: int = 101,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    item_id_cols = infer_item_id_columns(
        df,
        excluded_cols=["ablation", "component", "layer", "retrieved"],
    )

    cumulative_df = compute_model_level_cumulative(df=df, item_id_cols=item_id_cols)
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


def value_at_final(df: pd.DataFrame, value_col: str) -> float:
    matches = df[np.isclose(df["rel_layer"], 1.0)]
    if matches.empty:
        raise ValueError("No row found at rel_layer=1.0")
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one row at rel_layer=1.0, found {len(matches)}"
        )
    return float(matches.iloc[0][value_col])


def summarize_subset(df: pd.DataFrame, language_label: str) -> pd.DataFrame:
    cumulative_df, delta_df, model_order = prepare_plot_data(df)

    clean_df = cumulative_df[cumulative_df["ablation"] == CLEAN_ABLATION].copy()
    targeted_df = delta_df[delta_df["ablation"] == TARGETED_ABLATION].copy()
    random_df = delta_df[delta_df["ablation"] == RANDOM_ABLATION].copy()

    rows = []

    for model in model_order:
        clean_model = clean_df[clean_df["model"] == model].copy()
        targeted_model = targeted_df[targeted_df["model"] == model].copy()
        random_model = random_df[random_df["model"] == model].copy()

        if clean_model.empty or targeted_model.empty:
            continue

        rows.append(
            {
                "Language": language_label,
                "Model": short_model_name(model),
                "Clean final": value_at_final(
                    clean_model, "cumulative_first_retrieval_rate"
                ),
                "Targeted delta final": value_at_final(
                    targeted_model, "delta_to_clean"
                ),
                "Random delta final": (
                    value_at_final(random_model, "delta_to_clean")
                    if not random_model.empty
                    else np.nan
                ),
            }
        )

    if not rows:
        raise ValueError(f"No summary rows computed for language={language_label!r}")

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["Language", "Clean final"],
        ascending=[True, False],
    ).reset_index(drop=True)
    return out


def build_summary_table(df: pd.DataFrame, pooled: bool) -> pd.DataFrame:
    summaries = []

    if pooled:
        summaries.append(summarize_subset(df, language_label="Pooled"))

    if "lang" in df.columns:
        langs = sorted(df["lang"].dropna().unique().tolist())
        for lang in langs:
            lang_df = df[df["lang"] == lang].copy()
            if lang_df.empty:
                continue
            summaries.append(
                summarize_subset(lang_df, language_label=str(lang).title())
            )

    if not summaries:
        raise ValueError("No summary tables could be built.")

    return pd.concat(summaries, ignore_index=True)


def format_number(x: object, decimals: int) -> str:
    if pd.isna(x):
        return "--"
    return f"{float(x):.{decimals}f}"


def dataframe_to_latex_booktabs(
    df: pd.DataFrame,
    caption: str,
    label: str,
    decimals: int,
    table_env: str,
) -> str:
    columns = [
        "Language",
        "Model",
        "Clean final",
        "Targeted delta final",
        "Random delta final",
    ]

    colspec = "llrrr"
    header = (
        "\\textbf{Language} & \\textbf{Model} & \\textbf{Clean final $\\uparrow$} "
        "& \\textbf{Targeted $\\Delta$ final} & \\textbf{Random $\\Delta$ final} \\\\"
    )

    lines = []
    lines.append(f"\\begin{{{table_env}}}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")
    lines.append(header)
    lines.append("\\midrule")

    grouped = list(df[columns].groupby("Language", sort=False, dropna=False))

    for group_idx, (language, group_df) in enumerate(grouped):
        nrows = len(group_df)
        for row_idx, (_, row) in enumerate(group_df.iterrows()):
            language_cell = (
                f"\\multirow{{{nrows}}}{{*}}{{{latex_escape(language)}}}"
                if row_idx == 0
                else ""
            )
            cells = [
                language_cell,
                f"\\texttt{{{latex_escape(row['Model'])}}}",
                format_number(row["Clean final"], decimals),
                format_number(row["Targeted delta final"], decimals),
                format_number(row["Random delta final"], decimals),
            ]
            lines.append(" & ".join(cells) + r" \\")
        if group_idx < len(grouped) - 1:
            lines.append("\\addlinespace")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{latex_escape(caption)}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\end{{{table_env}}}")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print a LaTeX summary table for layerwise retrieval results."
    )
    parser.add_argument("input_parquet", type=Path, help="Input parquet file.")
    parser.add_argument(
        "--pooled",
        action="store_true",
        help="Also include a pooled-languages summary block.",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=2,
        help="Number of decimal places to print (default: 2).",
    )
    parser.add_argument(
        "--table-env",
        choices=["table", "table*"],
        default="table*",
        help="LaTeX table environment to use (default: table*).",
    )
    parser.add_argument(
        "--caption",
        default=(
            "Summary of final-layer retrieval by language and model. "
            "Clean final is the cumulative first-retrieval rate at the final relative layer. "
            "Targeted and random deltas are measured relative to the clean condition; "
            "more negative values indicate stronger disruption."
        ),
        help="LaTeX caption text.",
    )
    parser.add_argument(
        "--label",
        default="tab:layerwise_summary",
        help="LaTeX label.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_parquet(args.input_parquet)
    df = prepare_dataframe(df)

    summary_df = build_summary_table(df, pooled=args.pooled)

    latex = dataframe_to_latex_booktabs(
        df=summary_df,
        caption=args.caption,
        label=args.label,
        decimals=args.decimals,
        table_env=args.table_env,
    )
    print(latex)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
