#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from tokefx.data import get_rho
from tokefx.utils import load_config

parser = argparse.ArgumentParser()
parser.add_argument("config_file", type=Path)
parser.add_argument("out_json", type=Path)
args = parser.parse_args()

assert not args.out_json.is_dir()
args.out_json.parent.mkdir(exist_ok=True, parents=True)

cfg = load_config(args.config_file)
print("calculating rho")
rho = get_rho(cfg)

print(f"exporting results to {args.out_json}")
with open(args.out_json, "w") as rho_out:
    json.dump(rho, rho_out)
