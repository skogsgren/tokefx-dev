#!/usr/bin/env python3
import os
from pathlib import Path

import iso639
import toml
import torch

from tokefx.benchmarks.belebele import belebele
from tokefx.benchmarks.xcopa import xcopa
from tokefx.benchmarks.xnli import xnli
from tokefx.utils import load_config, read_json, set_json

cfg = load_config("scripts/config.toml")
models = cfg["eval"]["models"]


def initialize_metrics_file(out: Path):
    if not out.parent.exists():
        out.parent.mkdir(parents=True)
    metrics = read_json(out)
    for _, t in models:
        if t not in metrics:
            metrics[t] = {}
    return metrics


# == belebele ==
belebele_out_fp = cfg["dir"]["out"] / "benchmark_belebele_original.json"
belebele_metrics = initialize_metrics_file(belebele_out_fp)
for model, tokenizer in models:
    for lang, spec in cfg["lang"].items():
        if belebele_metrics[tokenizer].get(spec["iso639"]):
            print(f"already evaluated {spec['iso639']} ({model}, {tokenizer})")
            continue
        print(f"testing belebele {spec['iso639']} for ({model}, {tokenizer})")
        belebele_fp = cfg["dir"]["belebele_base"] / (spec["iso639"] + ".jsonl")
        belebele_score = belebele(
            belebele_fp,
            model,
            tokenizer,
            device=torch.device("cuda"),
            max_seq_len=4096,
        )
        belebele_metrics[tokenizer][spec["iso639"]] = belebele_score
        set_json(belebele_out_fp, belebele_metrics)
        print(belebele_metrics)

# == xcopa ==
xcopa_out_fp = cfg["dir"]["out"] / "benchmark_xcopa_original.json"
xcopa_metrics = initialize_metrics_file(xcopa_out_fp)
for model, tokenizer in models:
    for lang, spec in cfg["lang"].items():
        if xcopa_metrics[tokenizer].get(spec["iso639"]):
            print(f"already evaluated {spec['iso639']} ({model}, {tokenizer})")
            continue
        print(f"testing xcopa {spec['iso639']} for ({model}, {tokenizer})")

        xcopa_score = xcopa(
            iso639.Language.from_part3(spec["iso639"][:3]).part1,
            model,
            tokenizer,
            device=torch.device("cuda"),
            max_seq_len=4096,
        )
        xcopa_metrics[tokenizer][spec["iso639"]] = xcopa_score
        set_json(xcopa_out_fp, xcopa_metrics)
        print(xcopa_metrics)

# == xnli ==
xnli_out_fp = cfg["dir"]["out"] / "benchmark_xnli_original.json"
xnli_metrics = initialize_metrics_file(xnli_out_fp)
for model, tokenizer in models:
    for lang, spec in cfg["lang"].items():
        if xnli_metrics[tokenizer].get(spec["iso639"]):
            print(f"already evaluated {spec['iso639']} ({model}, {tokenizer})")
            continue
        print(f"testing xnli {spec['iso639']} for ({model}, {tokenizer})")

        xnli_score = xnli(
            iso639.Language.from_part3(spec["iso639"][:3]).part1,
            model,
            tokenizer,
            device=torch.device("cuda"),
            max_seq_len=4096,
        )
        xnli_metrics[tokenizer][spec["iso639"]] = xnli_score
        set_json(xnli_out_fp, xnli_metrics)
        print(xnli_metrics)
