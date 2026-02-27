#!/usr/bin/env python3
import os
from pathlib import Path

from deepmerge import always_merger
import json
import toml

import torch
from transformers import AutoTokenizer


def load_config(cfg_path: Path | str) -> dict:
    cfg = toml.load(cfg_path)
    # we want directories to be Path object for iteration
    for k, v in cfg["dir"].items():
        cfg["dir"][k] = Path(os.path.expandvars(v))
    return cfg


def read_json(json_path: Path | str) -> dict:
    json_path = Path(json_path)
    if json_path.exists():
        return json.loads(json_path.read_text())
    else:
        return {}


def set_json(json_path: Path | str, new_data: dict):
    data = read_json(json_path)
    json_path.write_text(
        json.dumps(
            always_merger.merge(data, new_data),
            indent=2,
        )
    )


def tokenize(
    tokenizer: AutoTokenizer,
    text: str,
    context_window: int,
    add_special_tokens: bool = False,
    device: torch.device = torch.device("cpu"),
):
    enc = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
        truncation=True,
        max_length=context_window,
    )
    return {k: v.to(device) for k, v in enc.items()}


def token_len(
    tokenizer: AutoTokenizer,
    text: str,
    context_window: int,
    add_special_tokens: bool = False,
) -> int:
    enc = tokenize(
        tokenizer=tokenizer,
        text=text,
        context_window=context_window,
        add_special_tokens=add_special_tokens,
    )
    return len(enc["input_ids"][0])
