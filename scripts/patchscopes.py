#!/usr/bin/env python3
import argparse
from dataclasses import asdict
import json
from itertools import product
from pathlib import Path
import shutil
import subprocess

import pandas as pd
import torch
from transformers import AutoTokenizer

from tokefx.data import PUD_Data, PUDCandidate, PUDToken, PUDSequence
from tokefx.data import write_candidates_jsonl, iter_candidates_jsonl
from tokefx.encoder import UDEncoder
from tokefx.interpretability.attention import get_ablation_map
from tokefx.interpretability.lahis import get_lahis_ablation_map
from tokefx.interpretability.patchscopes import PatchScopesAnalyzer
from tokefx.interpretability.patchscopes import create_embed_candidates
from tokefx.plot.patchscopes import main as plot_patchscopes
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
    "--ablation_map_type",
    choices=["lahis", "attention_mass"],
    default="attention_mass",
    help="which method to derive ablation map from. defaults to %(default)s.",
)
parser.add_argument(
    "--in_boundary_mode",
    choices=["all", "embed"],
    default="embed",
    help="which in_boundary mode to use. defaults to %(default)s.",
)
args = parser.parse_args()
ABLATION_MAP_TYPE = args.ablation_map_type
CFG = load_config(args.cfg)
BASE_KWARGS = CFG["eval"]
if BASE_KWARGS.get("ignored_pos"):
    BASE_KWARGS["ignored_pos"] = set(BASE_KWARGS["ignored_pos"])

log(f"starting Patchscope evaluation run ({ABLATION_MAP_TYPE=}")
log(CFG)

BASE_DIR_FN = f"patchscopes_{args.in_boundary_mode}_{args.ablation_map_type}"
OUT_DIR = CFG["dir"]["out"]
OUT_PATCHSCOPES = OUT_DIR / f"{BASE_DIR_FN}.parquet"
OUT_EMBED_JSONL = OUT_DIR / "embed_candidates.jsonl"
OUT_EMBED_SUMMARY = OUT_DIR / "embed_summary.tsv"
OUT_EMBED_FULL = OUT_DIR / "embed_full.parquet"
PLOT_DIR = OUT_DIR / "plots" / BASE_DIR_FN
PLOT_DIR.mkdir(parents=True, exist_ok=True)

if args.overwrite:
    OUT_PATCHSCOPES.unlink(missing_ok=True)
    if PLOT_DIR.exists():
        assert PLOT_DIR.is_dir()
        shutil.rmtree(PLOT_DIR)
df = pd.read_parquet(OUT_PATCHSCOPES) if OUT_PATCHSCOPES.exists() else pd.DataFrame()

if args.overwrite_embed:
    OUT_EMBED_JSONL.unlink(missing_ok=True)
    OUT_EMBED_SUMMARY.unlink(missing_ok=True)

log("getting attention head candidates for ablation")
if ABLATION_MAP_TYPE == "lahis":
    OUT_HEAD_JSON_DIR = OUT_DIR / f"lahis_head_candidates_{args.in_boundary_mode}"
    assert OUT_HEAD_JSON_DIR.is_dir()
    global_ablation_map = get_lahis_ablation_map(OUT_HEAD_JSON_DIR)
elif ABLATION_MAP_TYPE == "attention_mass":
    OUT_HEAD_PARQUET = OUT_DIR / f"full_attn_heads_{args.in_boundary_mode}.parquet"
    global_ablation_map = get_ablation_map(
        OUT_HEAD_PARQUET, max_layer=10, rank_by="min_z"
    )
else:
    raise ValueError("unknown ablation map type. how did you get here?")

if len(df) == 0:
    existing = set()
else:
    existing = set(zip(df["model"], df["lang"], df["ablation"]))
configurations = []
for model_handle, lang_handle, ablation in product(
    CFG["eval"]["models"],
    CFG["lang"].items(),
    CFG["eval"]["ablations"],
):
    model_name = model_handle[1]
    language = lang_handle[0]
    if (model_name, language, ablation) not in existing:
        configurations.append((model_handle, lang_handle, ablation))

if not OUT_EMBED_JSONL.exists() and args.in_boundary_mode == "embed":
    log("creating embed candidates file")
    create_embed_candidates(
        list(product(CFG["eval"]["models"], CFG["lang"].items())),
        CFG["dir"]["ud_base"],
        OUT_EMBED_JSONL,
        OUT_EMBED_SUMMARY,
        OUT_EMBED_FULL,
        **BASE_KWARGS.copy(),
    )

log("starting Patchscope evaluation run")
for model_handle, lang_handle, ablation in configurations:
    model_spec, tokenizer_spec = model_handle
    lang, lang_spec = lang_handle

    log(f"starting analysis for {model_spec=} {lang=}")

    if ablation in {"targeted", "random"}:
        abl_map = global_ablation_map[lang][model_spec]
    else:
        abl_map = {}
    log(f"{abl_map=}")

    datafp = CFG["dir"]["ud_base"] / lang_spec["pud-conllu"]
    data = PUD_Data(datafp=datafp)

    run_kwargs = BASE_KWARGS.copy()
    run_kwargs["model_spec"] = model_spec
    run_kwargs["tokenizer_spec"] = tokenizer_spec
    run_kwargs["patchscopes_prompt"] = lang_spec["patchscopes_prompt"]
    run_kwargs["ablation"] = ablation
    run_kwargs["ablation_map"] = abl_map
    analyzer = PatchScopesAnalyzer(**run_kwargs)

    processed_tokens = set()
    rows = []

    if args.in_boundary_mode == "embed":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_spec)
        in_candidates = [
            asdict(x) for x in iter_candidates_jsonl(OUT_EMBED_JSONL, model_spec, lang)
        ]
        for i, cand in enumerate(in_candidates):
            wdc = cand.copy()
            wdc["token"] = PUDToken.from_dict(cand["token"])
            wdc["seq"] = PUDSequence.from_dict(cand["seq"])
            device = torch.device(run_kwargs["device"])
            enc = tokenizer(
                wdc["seq"].text_til_i(wdc["token"].idx), return_tensors="pt"
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            wdc["model_inp"] = enc
            in_candidates[i] = wdc
        del tokenizer
    elif args.in_boundary_mode == "all":
        encoder = UDEncoder(**run_kwargs)
        in_candidates = []
        for i, cand in enumerate(encoder.get_candidates(data, "in_boundary")):
            if i >= BASE_KWARGS["n_rows"]:
                break
            in_candidates.append(cand)
        del encoder
    else:
        raise ValueError(f"unsupported {args.in_boundary_mode=}")

    for i, cand in enumerate(in_candidates):
        seq = cand["seq"]
        tform = cand["token"].form.lower()
        if tform in processed_tokens:
            continue
        res = analyzer(cand, **run_kwargs)
        if not res:
            continue
        for i, raw_row in enumerate(res):
            add_row = raw_row
            add_row["ablation"] = ablation
            add_row["ablation_map"] = str({str(k): v for k, v in abl_map.items()})
            add_row["lang"] = lang
            add_row["model"] = model_spec
            add_row["text"] = seq.full_text()
            res[i] = add_row
        rows += res
        processed_tokens.add(tform)
        log(f"{len(processed_tokens)}/{BASE_KWARGS['n_rows']} processed {tform}")
        if len(processed_tokens) == BASE_KWARGS["n_rows"]:
            break
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df.to_parquet(OUT_PATCHSCOPES)
    del analyzer

log("creating plots")
plot_patchscopes(OUT_PATCHSCOPES, PLOT_DIR)
log("finished Patchscope evaluation run")
