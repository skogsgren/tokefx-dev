#!/usr/bin/env python3
from pathlib import Path

import iso639
import json
import toml

from tokefx.data import PUD_Data
from tokefx.interpretability.attention import AttentionAnalyzer
from tokefx.utils import load_config

cfg = load_config("scripts/config.toml")

models = cfg["eval"]["models"]

OUT_DIR = cfg["dir"]["out"]
OUT_DIR.mkdir(exist_ok=True)

# =============================================================================
# reproducing kaplan2025, look at previous subtoken in two-token words
# =============================================================================
# TODO: add all parameters, e.g. addspecialtokens
two_token_out = OUT_DIR / "attn_two_token"
two_token_out.mkdir(exist_ok=True)
for model, tokenizer in models:
    for lang, spec in cfg["lang"].items():
        out_json = two_token_out / f"{lang}_{model}.json"
        if out_json.exists():
            print(f"{str(out_json)} already exists. skipping...")
            continue
        print(f"analyzing two token attention for {lang}/{model}")
        data = PUD_Data(datafp=spec["pud-conllu"], tokenizer_spec=tokenizer)
        analyzer = AttentionAnalyzer(model, tokenizer, device=cfg["eval"]["device"])
        df = analyzer.analyze(
            data=data,
            mode="prev_subtokens",
            min_subtoken=2,
            max_subtoken=2,
        )
        layer_cols = [c for c in df.columns if c.startswith("layer_")]
        mean = df[layer_cols].mean().sort_index().to_dict()
        with open(out_json, "w") as f:
            json.dump({str(k): float(v) for k, v in mean.items()}, f)
        exit(1)
