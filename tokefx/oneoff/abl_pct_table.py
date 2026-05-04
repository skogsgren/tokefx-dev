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


def format_level(level: float) -> str:
    return f"{level * 100:g}\\%"


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


def compute_final_cumulative_first_retrieval(
    df: pd.DataFrame,
    item_id_cols: list[str],
    group_cols: list[str],
) -> pd.DataFrame:
    validate_required_columns(df, set(group_cols) | {"layer", "retrieved"})

    item_cols = item_id_cols + [col for col in group_cols if col not in item_id_cols]

    totals = (
        df[item_cols]
        .drop_duplicates()
        .groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="total_items")
    )

    retrieved_df = df[df["retrieved"] == 1].copy()

    if retrieved_df.empty:
        out = totals.copy()
        out["final_cumulative_first_retrieval_rate"] = 0.0
        return out[group_cols + ["final_cumulative_first_retrieval_rate"]]

    first_hits = (
        retrieved_df.groupby(item_cols, dropna=False)["layer"]
        .min()
        .reset_index(name="first_layer")
    )

    final_counts = (
        first_hits.groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="count_retrieved_by_final")
    )

    out = totals.merge(final_counts, on=group_cols, how="left")
    out["count_retrieved_by_final"] = out["count_retrieved_by_final"].fillna(0)
    out["final_cumulative_first_retrieval_rate"] = (
        out["count_retrieved_by_final"] / out["total_items"]
    )

    return out[group_cols + ["final_cumulative_first_retrieval_rate"]]


def compute_final_summary_for_level(
    df: pd.DataFrame,
    level: float,
    language_label: str,
) -> pd.DataFrame:
    item_id_cols = infer_item_id_columns(
        df,
        excluded_cols=["ablation", "component", "layer", "retrieved"],
    )

    final_df = compute_final_cumulative_first_retrieval(
        df=df,
        item_id_cols=item_id_cols,
        group_cols=["model", "ablation"],
    )

    clean_df = final_df[final_df["ablation"] == CLEAN_ABLATION][
        ["model", "final_cumulative_first_retrieval_rate"]
    ].rename(columns={"final_cumulative_first_retrieval_rate": "Clean final"})

    targeted_df = final_df[final_df["ablation"] == TARGETED_ABLATION][
        ["model", "final_cumulative_first_retrieval_rate"]
    ].rename(columns={"final_cumulative_first_retrieval_rate": "Targeted final"})

    random_df = final_df[final_df["ablation"] == RANDOM_ABLATION][
        ["model", "final_cumulative_first_retrieval_rate"]
    ].rename(columns={"final_cumulative_first_retrieval_rate": "Random final"})

    if clean_df.empty:
        raise ValueError(f"No clean rows found for {language_label}, level={level}.")
    if targeted_df.empty:
        raise ValueError(f"No targeted rows found for {language_label}, level={level}.")

    out = clean_df.merge(targeted_df, on="model", how="left")
    out = out.merge(random_df, on="model", how="left")

    out["Targeted delta final"] = out["Targeted final"] - out["Clean final"]
    out["Random delta final"] = out["Random final"] - out["Clean final"]

    out["Language"] = language_label
    out["ModelRaw"] = out["model"]
    out["Model"] = out["model"].map(short_model_name)
    out["Ablation level"] = level

    out = out.sort_values("Clean final", ascending=False).reset_index(drop=True)

    return out[
        [
            "Language",
            "ModelRaw",
            "Model",
            "Ablation level",
            "Clean final",
            "Targeted delta final",
            "Random delta final",
        ]
    ]


def build_long_summary_table(
    all_df: pd.DataFrame,
    levels: list[float],
    pooled: bool,
) -> pd.DataFrame:
    summaries = []

    if pooled:
        for level in levels:
            level_df = all_df[np.isclose(all_df["ablation_level"], level)].copy()
            if not level_df.empty:
                summaries.append(
                    compute_final_summary_for_level(
                        df=level_df,
                        level=level,
                        language_label="Pooled",
                    )
                )

    if "lang" in all_df.columns:
        langs = sorted(all_df["lang"].dropna().unique().tolist())

        for lang in langs:
            for level in levels:
                subset = all_df[
                    (all_df["lang"] == lang)
                    & np.isclose(all_df["ablation_level"], level)
                ].copy()

                if subset.empty:
                    continue

                summaries.append(
                    compute_final_summary_for_level(
                        df=subset,
                        level=level,
                        language_label=str(lang).title(),
                    )
                )

    if not summaries:
        raise ValueError("No summary tables could be built.")

    return pd.concat(summaries, ignore_index=True)


def format_number(x: object, decimals: int) -> str:
    if pd.isna(x):
        return "--"
    return f"{float(x):.{decimals}f}"


def build_wide_summary_table(
    long_df: pd.DataFrame,
    levels: list[float],
    decimals: int,
    include_gap: bool,
) -> pd.DataFrame:
    rows = []

    grouped = long_df.groupby(
        ["Language", "ModelRaw", "Model"], sort=False, dropna=False
    )

    for (language, model_raw, model), group_df in grouped:
        clean_values = group_df["Clean final"].dropna().unique()
        clean_final = group_df.sort_values("Ablation level").iloc[0]["Clean final"]

        if len(clean_values) > 1:
            max_diff = np.max(clean_values) - np.min(clean_values)
            if abs(max_diff) > 1e-8:
                print(
                    f"Warning: clean final differs across levels for "
                    f"{language}/{model}: range={max_diff:.6g}",
                    file=sys.stderr,
                )

        row = {
            "Language": language,
            "Model": model,
            "Clean final": format_number(clean_final, decimals),
        }

        for level in levels:
            level_rows = group_df[np.isclose(group_df["Ablation level"], level)]

            if level_rows.empty:
                targeted = np.nan
                random = np.nan
            else:
                targeted = level_rows.iloc[0]["Targeted delta final"]
                random = level_rows.iloc[0]["Random delta final"]

            row[("Targeted", level)] = format_number(targeted, decimals)
            row[("Random", level)] = format_number(random, decimals)

        if include_gap:
            max_level = max(levels)
            max_rows = group_df[np.isclose(group_df["Ablation level"], max_level)]

            if max_rows.empty:
                gap = np.nan
            else:
                targeted = max_rows.iloc[0]["Targeted delta final"]
                random = max_rows.iloc[0]["Random delta final"]
                gap = (
                    np.nan
                    if pd.isna(targeted) or pd.isna(random)
                    else targeted - random
                )

            row["Gap"] = format_number(gap, decimals)

        rows.append(row)

    out = pd.DataFrame(rows)

    sort_clean = pd.to_numeric(
        out["Clean final"].replace("--", np.nan), errors="coerce"
    )
    out = (
        out.assign(__sort_clean=sort_clean)
        .sort_values(["Language", "__sort_clean"], ascending=[True, False])
        .drop(columns="__sort_clean")
        .reset_index(drop=True)
    )

    return out


def dataframe_to_latex_booktabs(
    df: pd.DataFrame,
    levels: list[float],
    caption: str,
    label: str,
    table_env: str,
    resize: bool,
    include_gap: bool,
    font_size: str,
    tabcolsep: float,
) -> str:
    n_levels = len(levels)

    colspec = "llr" + ("r" * n_levels) + ("r" * n_levels)
    if include_gap:
        colspec += "r"

    header_1 = [
        r"\textbf{Language}",
        r"\textbf{Model}",
        r"\textbf{Clean final $\uparrow$}",
        rf"\multicolumn{{{n_levels}}}{{c}}{{\textbf{{Targeted $\Delta$ final}}}}",
        rf"\multicolumn{{{n_levels}}}{{c}}{{\textbf{{Random $\Delta$ final}}}}",
    ]

    if include_gap:
        header_1.append(rf"\textbf{{Gap {format_level(max(levels))}}}")

    header_2 = ["", "", ""]
    header_2.extend(format_level(level) for level in levels)
    header_2.extend(format_level(level) for level in levels)
    if include_gap:
        header_2.append("")

    cmidrules = [
        rf"\cmidrule(lr){{4-{3 + n_levels}}}",
        rf"\cmidrule(lr){{{4 + n_levels}-{3 + 2 * n_levels}}}",
    ]

    lines = []
    lines.append(f"\\begin{{{table_env}}}[t]")
    lines.append("\\centering")
    lines.append(f"\\{font_size}")
    lines.append(f"\\setlength{{\\tabcolsep}}{{{tabcolsep}pt}}")

    if resize:
        lines.append("\\resizebox{\\linewidth}{!}{%")

    lines.append(f"\\begin{{tabular}}{{{colspec}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(header_1) + r" \\")
    lines.extend(cmidrules)
    lines.append(" & ".join(header_2) + r" \\")
    lines.append("\\midrule")

    grouped = list(df.groupby("Language", sort=False, dropna=False))

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
                str(row["Clean final"]),
            ]

            for level in levels:
                cells.append(str(row[("Targeted", level)]))

            for level in levels:
                cells.append(str(row[("Random", level)]))

            if include_gap:
                cells.append(str(row["Gap"]))

            lines.append(" & ".join(cells) + r" \\")

        if group_idx < len(grouped) - 1:
            lines.append("\\addlinespace")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    if resize:
        lines.append("}%")

    lines.append(f"\\caption{{{latex_escape(caption)}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\end{{{table_env}}}")

    return "\n".join(lines)


def default_caption(include_gap: bool, levels: list[float]) -> str:
    base = (
        "Summary of final-layer retrieval under increasing ablation strength. "
        "Clean final is the cumulative first-retrieval rate at the final layer. "
        "Targeted and random deltas are measured relative to the clean condition. "
        "More negative values indicate stronger disruption after all layers have had "
        "the opportunity to compensate."
    )

    if include_gap:
        base += (
            f" Gap {format_level(max(levels)).replace(chr(92), '')} is the targeted "
            "delta minus the random delta at the largest ablation level."
        )

    return base


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print a grouped-column LaTeX summary table for dose-response retrieval results."
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

    parser.add_argument("--pooled", action="store_true")
    parser.add_argument("--decimals", type=int, default=2)
    parser.add_argument("--table-env", choices=["table", "table*"], default="table*")
    parser.add_argument("--no-resize", action="store_true")
    parser.add_argument("--no-gap", action="store_true")
    parser.add_argument(
        "--font-size",
        choices=["small", "footnotesize", "scriptsize"],
        default="scriptsize",
    )
    parser.add_argument("--tabcolsep", type=float, default=3.0)
    parser.add_argument("--caption", default=None)
    parser.add_argument("--label", default="tab:layerwise_dose_response_summary")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ablation_specs = parse_ablation_specs(args.abl)
    levels = [level for level, _ in ablation_specs]
    include_gap = not args.no_gap

    all_df = read_level_parquets(ablation_specs)

    long_summary_df = build_long_summary_table(
        all_df=all_df,
        levels=levels,
        pooled=args.pooled,
    )

    wide_summary_df = build_wide_summary_table(
        long_df=long_summary_df,
        levels=levels,
        decimals=args.decimals,
        include_gap=include_gap,
    )

    caption = (
        args.caption
        if args.caption is not None
        else default_caption(include_gap=include_gap, levels=levels)
    )

    latex = dataframe_to_latex_booktabs(
        df=wide_summary_df,
        levels=levels,
        caption=caption,
        label=args.label,
        table_env=args.table_env,
        resize=not args.no_resize,
        include_gap=include_gap,
        font_size=args.font_size,
        tabcolsep=args.tabcolsep,
    )

    print(latex)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
