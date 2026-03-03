#!/usr/bin/env python3
import argparse
from datetime import datetime
import gc
import json
from pathlib import Path
import shutil

import pandas as pd
import torch
from transformers import AutoTokenizer

from tokefx.data import PUD_Data
from tokefx.interpretability.attention import AttentionAnalyzer
from tokefx.plot import attention_plots, attention_head_plots
from tokefx.utils import load_config

parser = argparse.ArgumentParser()
parser.add_argument("cfg", type=Path, help="path to config file")
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="overwrites existing attention files",
)
args = parser.parse_args()
CFG = load_config(args.cfg)
analyses = set(CFG["eval"]["analyses"])
print(f"{datetime.now()} starting attention analysis run")
print(analyses)
print(CFG)

OUT_DIR = CFG["dir"]["out"]
OUT_ATTN = OUT_DIR / "full_attn.parquet"
OUT_HEADS = OUT_DIR / "full_attn_heads.parquet"
if args.overwrite:
    OUT_ATTN.unlink(missing_ok=True)
    OUT_HEADS.unlink(missing_ok=True)
    if (OUT_DIR / "plots").exists():
        assert (OUT_DIR / "plots").is_dir()
        shutil.rmtree(OUT_DIR / "plots")
OUT_DIR.mkdir(exist_ok=True)
df = pd.read_parquet(OUT_ATTN) if OUT_ATTN.exists() else pd.DataFrame()
head_df = pd.read_parquet(OUT_HEADS) if OUT_HEADS.exists() else pd.DataFrame()


def attn_run_wrapper(**kwargs) -> list[dict]:
    """runs analysis through all configurations given kwargs"""
    global df
    global head_df
    all_rows = []
    all_heads = []
    for model, tokenizer in CFG["eval"]["models"]:
        for lang, spec in CFG["lang"].items():
            print(
                f"{datetime.now()} {kwargs['mode_label']} analyzing {lang} with {model}"
            )

            datafp = CFG["dir"]["ud_base"] / spec["pud-conllu"]
            data = PUD_Data(datafp=datafp)
            analyzer = AttentionAnalyzer(
                model,
                tokenizer,
                device=CFG["eval"]["device"],
                add_special_tokens=CFG["eval"]["add_special_tokens"],
                ignored_pos=CFG["eval"].get("ignored_pos", set()),
            )
            run_kwargs = dict(kwargs)
            run_kwargs["lang"] = lang
            run_kwargs["model"] = model
            if src_agg := CFG["eval"].get("source_aggregation"):
                run_kwargs["source_aggregation"] = src_agg
            rows, heads = analyzer.analyze(data, CFG["eval"]["n_rows"], **run_kwargs)

            del analyzer
            gc.collect()
            torch.cuda.empty_cache()

            for i in range(len(rows)):
                rows[i]["lang"] = lang
                rows[i]["model"] = model
                if not rows[i].get("mode"):
                    rows[i]["mode"] = kwargs["mode_label"]
                all_rows.append(rows[i])
            for i in range(len(heads)):
                heads[i]["lang"] = lang
                heads[i]["model"] = model
                if not heads[i].get("mode"):
                    heads[i]["mode"] = kwargs["mode_label"]
                all_heads.append(heads[i])

    df = pd.concat([df, pd.DataFrame(all_rows)], ignore_index=True)
    df.to_parquet(OUT_ATTN, index=False)
    head_df = pd.concat([head_df, pd.DataFrame(all_heads)], ignore_index=True)
    head_df.to_parquet(OUT_HEADS, index=False)


def no_previous_run(label: str) -> bool:
    if args.overwrite:
        print("overwrite set to true. overwriting previous run...")
        return True
    mode = df.get("mode")
    if mode is None:
        return True
    if (mode == label).any():
        print(f"previous run found for {label}. skipping...")
        return False
    return True


if "compound" in analyses:
    if no_previous_run("compound"):
        attn_run_wrapper(
            mode="compound",
            mode_label="compound",
            min_context=4,
        )


if "boundary" in analyses:
    if no_previous_run("in_boundary") and no_previous_run("out_boundary"):
        attn_run_wrapper(
            mode="boundary",
            mode_label="boundary",
            min_context=4,
        )

attention_plots(OUT_ATTN, OUT_DIR / "plots")
attention_head_plots(OUT_HEADS, OUT_DIR / "plots")
