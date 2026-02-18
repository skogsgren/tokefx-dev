#!/usr/bin/env python3
import os
from pathlib import Path

from icecream import ic
import toml
import torch

from tokefx.benchmarks.belebele import belebele
from tokefx.utils import load_config, read_json, set_json

cfg = load_config("scripts/config.toml")
models = cfg["eval"]["models"]

metrics_file_out = cfg["dir"]["out"] / "benchmark_original.json"
if not metrics_file_out.parent.exists():
    metrics_file_out.parent.mkdir(parents=True)

metrics = read_json(metrics_file_out)
for _, t in models:
    if t not in metrics:
        metrics[t] = {}

for model, tokenizer in models:
    for lang, spec in cfg["lang"].items():
        if metrics[tokenizer].get(spec["iso639"]):
            print(f"already evaluated {spec['iso639']} ({model}, {tokenizer})")
            continue
        print(f"testing {spec['iso639']} for ({model}, {tokenizer})")
        belebele_fp = cfg["dir"]["belebele_base"] / (spec["iso639"] + ".jsonl")
        belebele_score = belebele(
            belebele_fp,
            model,
            tokenizer,
            device=torch.device("cuda"),
            max_seq_len=4096,
        )
        metrics[tokenizer][spec["iso639"]] = belebele_score
        set_json(metrics_file_out, metrics)
        ic(metrics)
