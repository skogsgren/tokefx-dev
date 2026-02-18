#!/usr/bin/env python3
import toml
from transformers import AutoTokenizer

cfg = toml.load("/workspace/scripts/config.toml")
for _, tokenizer_handle in cfg["eval"]["models"]:
    print(tokenizer_handle)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_handle, use_fast=False)
    print(tokenizer_handle, ": ", tokenizer.special_tokens_map)
