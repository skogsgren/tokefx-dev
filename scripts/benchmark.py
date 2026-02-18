#!/usr/bin/env python3
from pathlib import Path

from icecream import ic
import toml
import torch

from tokefx.benchmarks.belebele import belebele

cfg = toml.load("scripts/config.toml")
models = cfg["eval"]["models"]
DATADIR = Path(cfg["data"]["BELEBELE_BASE"])

metrics: dict = {t: {} for _, t in models}
for model, tokenizer in models:
    for lang, spec in cfg["data"]["lang"].items():
        print(f"testing {spec['iso639']} for ({model}, {tokenizer})")
        belebele_fp = DATADIR / (spec["iso639"] + ".jsonl")
        belebele_score = belebele(
            belebele_fp,
            model,
            tokenizer,
            device=torch.device("cuda"),
            max_seq_len=512,
        )
        metrics[tokenizer][spec["iso639"]] = belebele_score
        ic(metrics)
