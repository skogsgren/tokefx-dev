#!/usr/bin/env python3
from datetime import datetime
import os
from pathlib import Path

from deepmerge import always_merger
import json
import toml

import torch
from transformers import AutoTokenizer

LOG_FILE = Path("./ALLMIGHTY.log")


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


def log(msg: str):
    fmt_msg = f"{datetime.now()} {msg}"
    print(fmt_msg)
    with open(LOG_FILE, "a") as f:
        f.write(fmt_msg + "\n")
