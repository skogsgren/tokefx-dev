#!/usr/bin/env python3
import argparse
from collections import defaultdict
from datetime import datetime
import gc
import json
from itertools import product
from pathlib import Path
import shutil
import subprocess

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

from tokefx.data import (
    PUD_Data,
    PUDCandidate,
    write_candidates_jsonl,
    iter_candidates_jsonl,
)
from tokefx.encoder import UDEncoder, get_embed_attn_in_candidates
from tokefx.interpretability.lahis import LahisAnalyzer
from tokefx.interpretability.patchscopes import (
    PatchScopesAnalyzer,
    create_embed_candidates,
)
from tokefx.plot.lahis import export_head_heatmap
from tokefx.utils import load_config, log

from icecream import ic

parser = argparse.ArgumentParser()
parser.add_argument("cfg", type=Path, help="path to config file")
parser.add_argument(
    "--overwrite",
    action="store_true",
    help="overwrites existing Patchscope files",
)
parser.add_argument(
    "--overwrite_embed",
    action="store_true",
    help="overwrites cached dataframe which catalogues words not retrieved from input embed layer",
)
parser.add_argument(
    "--in_boundary_mode",
    choices=["all", "embed"],
    default="embed",
    help="which in_boundary mode to use. defaults to %(default)s.",
)
args = parser.parse_args()
CFG = load_config(args.cfg)
BASE_KWARGS = CFG["eval"]
if BASE_KWARGS.get("ignored_pos"):
    BASE_KWARGS["ignored_pos"] = set(BASE_KWARGS["ignored_pos"])

log("starting lahis evaluation run")
log(CFG)

OUT_DIR = CFG["dir"]["out"]
OUT_EMBED_JSONL = OUT_DIR / "embed_candidates.jsonl"
OUT_EMBED_SUMMARY = OUT_DIR / "embed_summary.tsv"

OUT_HEAD_DIR = OUT_DIR / f"lahis_head_candidates_{args.in_boundary_mode}"
OUT_PLOT_DIR = OUT_DIR / "plots" / f"lahis_head_candidates_{args.in_boundary_mode}"

if args.overwrite:
    shutil.rmtree(OUT_HEAD_DIR)
    shutil.rmtree(OUT_PLOT_DIR)

if args.overwrite_embed:
    OUT_EMBED_JSONL.unlink(missing_ok=True)
    OUT_EMBED_SUMMARY.unlink(missing_ok=True)

OUT_PLOT_DIR.mkdir(exist_ok=True, parents=True)
OUT_HEAD_DIR.mkdir(exist_ok=True, parents=True)

configurations = []
for model_handle, lang_handle in product(CFG["eval"]["models"], CFG["lang"].items()):
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
        **run_kwargs,
    )
    log("finished creating embed candidates and summary")


log("finding attention head candidates")
for model_handle, lang_handle in configurations:
    model_spec, tokenizer_spec = model_handle
    lang, lang_spec = lang_handle

    run_kwargs = BASE_KWARGS.copy()
    run_kwargs["model_spec"] = model_spec
    run_kwargs["tokenizer_spec"] = tokenizer_spec
    run_kwargs["ignored_pos"] = set(run_kwargs.get("ignored_pos", []))

    device = torch.device(BASE_KWARGS["device"])
    analyzer = LahisAnalyzer(**run_kwargs)

    # out_boundary condition
    log("getting candidates for out_boundary")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_spec)
    out_candidates = []
    encoder = UDEncoder(**run_kwargs)
    data = PUD_Data(datafp=CFG["dir"]["ud_base"] / lang_spec["pud-conllu"])
    for i, cand in enumerate(encoder.get_candidates(data, "out_boundary")):
        if i >= BASE_KWARGS["n_rows"]:
            break
        target_idx = int(cand["model_inp"]["input_ids"][0, -1].item())
        enc = {k: v[:, :-1] for k, v in cand["model_inp"].items()}
        out_candidates.append((enc, target_idx))
        target_token_form = tokenizer.decode([target_idx])
        source_token_form = tokenizer.decode([enc["input_ids"][0][-1]])
        log(
            f"{i+1}/{BASE_KWARGS['n_rows']} ({source_token_form}) > {target_token_form}"
        )
    del encoder, tokenizer
    out_candidates_matrix = analyzer(out_candidates)  # L, H
    log(f"{out_candidates_matrix.shape=}")
    del out_candidates

    # in_boundary condition
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

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_spec)
    final_in_candidates = []
    processed_tokens = set()
    for i, cand in enumerate(in_candidates):
        seq = cand["seq"]
        token = cand["token"]
        if token.form.lower() in processed_tokens:
            continue
        enc = tokenizer(seq.text_til_i(token.idx), return_tensors="pt")
        target_idx = int(enc["input_ids"][0, -1].item())
        enc = {k: v[:, :-1].to(device) for k, v in enc.items()}

        final_in_candidates.append((enc, target_idx))
        processed_tokens.add(token.form.lower())

        target_token_form = tokenizer.decode([target_idx])
        source_token_form = tokenizer.decode([enc["input_ids"][0][-1]])

        log(
            f"{i+1}/{BASE_KWARGS['n_rows']} ({source_token_form}) > {target_token_form}"
        )
    del in_candidates
    del tokenizer
    in_candidates_matrix = analyzer(final_in_candidates)
    del final_in_candidates

    out_heatmap = (
        OUT_PLOT_DIR / f"{model_spec.replace('/', '-')}_{lang}_head_heatmap.png"
    )
    out_head_json = OUT_HEAD_DIR / f"{model_spec.replace('/', '-')}_{lang}_heads.json"

    export_head_heatmap(out_candidates_matrix - in_candidates_matrix, out_heatmap)
    head_map = analyzer.select_heads(
        out_candidates_matrix, in_candidates_matrix, topk=20
    )
    with open(out_head_json, "w") as f:
        json.dump(head_map, f)

    del analyzer

log("finished identifying attention heads")
