#!/usr/bin/env python3
import argparse
from datetime import datetime
import gc
import json
from itertools import product
from pathlib import Path
import shutil
import subprocess

import pandas as pd
import torch
from transformers import AutoTokenizer

from tokefx.data import PUD_Data
from tokefx.interpretability.attention import AttentionAnalyzer
from tokefx.encoder import UDEncoder, get_embed_attn_in_candidates
from tokefx.utils import load_config, log
from tokefx.interpretability.patchscopes import (
    PatchScopesAnalyzer,
    create_embed_candidates,
)
from tokefx.data import (
    write_candidates_jsonl,
    iter_candidates_jsonl,
)

parser = argparse.ArgumentParser()
parser.add_argument("cfg", type=Path, help="path to config file")
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="removes existing files before run",
)
parser.add_argument(
    "--overwrite_embed",
    action="store_true",
    help="overwrites cache which catalogues words not retrieved from input embed layer",
)
parser.add_argument(
    "--in_boundary_mode",
    choices=["all", "embed"],
    default="embed",
    help="which mode to derive in_boundary from. defaults to %(default)s.",
)
args = parser.parse_args()
CFG = load_config(args.cfg)
BASE_KWARGS = CFG["eval"]
if BASE_KWARGS.get("ignored_pos"):
    BASE_KWARGS["ignored_pos"] = set(BASE_KWARGS["ignored_pos"])

log("starting attention analysis run")
log(CFG)

OUT_DIR = CFG["dir"]["out"]
OUT_ATTN = OUT_DIR / f"full_attn_{args.in_boundary_mode}.parquet"
OUT_HEADS = OUT_DIR / f"full_attn_heads_{args.in_boundary_mode}.parquet"
OUT_PLOT_DIR = OUT_DIR / "plots" / f"raw_attention_{args.in_boundary_mode}"

if args.overwrite:
    OUT_ATTN.unlink(missing_ok=True)
    OUT_HEADS.unlink(missing_ok=True)
    if OUT_PLOT_DIR.exists():
        assert OUT_PLOT_DIR.is_dir()
        shutil.rmtree(OUT_PLOT_DIR)

OUT_DIR.mkdir(exist_ok=True)
OUT_PLOT_DIR.mkdir(exist_ok=True, parents=True)

OUT_EMBED_JSONL = OUT_DIR / "embed_candidates.jsonl"
OUT_EMBED_FULL = OUT_DIR / "embed_full.parquet"
OUT_EMBED_SUMMARY = OUT_DIR / "embed_summary.tsv"

df = pd.read_parquet(OUT_ATTN) if OUT_ATTN.exists() else pd.DataFrame()
head_df = pd.read_parquet(OUT_HEADS) if OUT_HEADS.exists() else pd.DataFrame()

if args.overwrite_embed:
    OUT_EMBED_JSONL.unlink(missing_ok=True)
    OUT_EMBED_SUMMARY.unlink(missing_ok=True)


configurations = []
for model_handle, lang_handle in product(
    CFG["eval"]["models"],
    CFG["lang"].items(),
):
    model_name = model_handle[1]
    language = lang_handle[0]
    configurations.append((model_handle, lang_handle))

if not OUT_EMBED_JSONL.exists() and args.in_boundary_mode == "embed":
    log("creating embed candidates file")
    run_kwargs = BASE_KWARGS.copy()
    create_embed_candidates(
        configurations=configurations,
        ud_base=CFG["dir"]["ud_base"],
        out_embed_jsonl=OUT_EMBED_JSONL,
        out_embed_summary=OUT_EMBED_SUMMARY,
        out_full=OUT_EMBED_FULL,
        **run_kwargs,
    )
    log("finished creating embed candidates and summary")


for model_handle, lang_handle in configurations:
    model_spec, tokenizer_spec = model_handle
    lang, lang_spec = lang_handle

    run_kwargs = BASE_KWARGS.copy()
    run_kwargs["model_spec"] = model_spec
    run_kwargs["tokenizer_spec"] = tokenizer_spec
    run_kwargs["ignored_pos"] = set(run_kwargs.get("ignored_pos", []))

    device = torch.device(BASE_KWARGS["device"])
    analyzer = AttentionAnalyzer(**run_kwargs)

    layer_rows = []
    head_rows = []

    log("getting candidates for out_boundary")
    out_candidates = []
    datafp = CFG["dir"]["ud_base"] / lang_spec["pud-conllu"]
    data = PUD_Data(datafp=datafp)
    encoder = UDEncoder(**run_kwargs)
    for i, cand in enumerate(encoder.get_candidates(data, "out_boundary")):
        if i >= BASE_KWARGS["n_rows"]:
            break
        out_candidates.append(cand)
    assert (
        len(out_candidates) == BASE_KWARGS["n_rows"]
    ), f"{len(out_candidates)=}, {model_spec=}, {lang=}"
    del encoder

    log("analyzing attention for out_boundary")
    out_layer_rows, out_head_rows = analyzer(out_candidates, BASE_KWARGS["n_rows"])
    for i, row in enumerate(out_layer_rows):
        out_layer_rows[i]["mode"] = "out_boundary"
    for i, row in enumerate(out_head_rows):
        out_head_rows[i]["mode"] = "out_boundary"
    layer_rows += out_layer_rows
    head_rows += out_head_rows
    del out_candidates, out_layer_rows, out_head_rows

    log("getting candidates for in_boundary")
    if args.in_boundary_mode == "embed":
        in_candidates = get_embed_attn_in_candidates(
            jsonl_fp=OUT_EMBED_JSONL,
            tokenizer_spec=tokenizer_spec,
            model_spec=model_spec,
            lang=lang,
            device=device,
        )
    elif args.in_boundary_mode == "all":  # i.e. same as for out_boundary
        datafp = CFG["dir"]["ud_base"] / lang_spec["pud-conllu"]
        data = PUD_Data(datafp=datafp)
        encoder = UDEncoder(**run_kwargs)
        in_candidates = []
        for i, cand in enumerate(encoder.get_candidates(data, "in_boundary")):
            if i >= BASE_KWARGS["n_rows"]:
                break
            in_candidates.append(cand)
        del encoder
    else:
        raise ValueError("unsupported in_boundary_mode {args.in_boundary_mode=}")
    assert (
        len(in_candidates) == BASE_KWARGS["n_rows"]
    ), f"{len(in_candidates)=}, {model_spec=}, {lang=}"

    log("analyzing attention for in_boundary")
    in_layer_rows, in_head_rows = analyzer(in_candidates, BASE_KWARGS["n_rows"])
    for i, row in enumerate(in_layer_rows):
        in_layer_rows[i]["mode"] = "in_boundary"
    for i, row in enumerate(in_head_rows):
        in_head_rows[i]["mode"] = "in_boundary"
    layer_rows += in_layer_rows
    head_rows += in_head_rows
    del in_candidates, in_layer_rows, in_head_rows
    del analyzer

    working_layer_df = pd.DataFrame(layer_rows)
    working_layer_df["model"] = model_spec
    working_layer_df["lang"] = lang

    working_head_df = pd.DataFrame(head_rows)
    working_head_df["model"] = model_spec
    working_head_df["lang"] = lang

    df = pd.concat([df, working_layer_df], ignore_index=True)
    head_df = pd.concat([head_df, working_head_df], ignore_index=True)
    df.to_parquet(OUT_ATTN, index=False)
    head_df.to_parquet(OUT_HEADS, index=False)

log("creating plots")
subprocess.run(
    [
        "python3",
        "tokefx/plot/attention.py",
        "--input_tsv",
        str(OUT_ATTN),
        "--heads_tsv",
        str(OUT_HEADS),
        "--plot_format",
        "png",
        "--modes",
        "in_boundary",
        "--output_dir",
        str(OUT_PLOT_DIR),
    ]
)
log("finished attention analysis")
