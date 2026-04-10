#!/usr/bin/env python3
from pathlib import Path
import json

import codecs
from datasets import load_dataset
import iso639
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
from tqdm import tqdm

from tokefx.constants import SPECIAL_TOKEN_RULES


def xcopa(
    lang: str,
    model_spec: str,
    tokenizer_spec: str,
    device: torch.device = torch.device("cpu"),
    max_seq_len: int = 4096,
) -> float:
    if lang == "en":
        data = load_dataset("pkavumba/balanced-copa")["test"]
    else:
        data = load_dataset("cambridgeltl/xcopa", lang)["test"]
    model = AutoModelForCausalLM.from_pretrained(model_spec).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_spec)
    pid = -100 if tokenizer.pad_token_id is None else tokenizer.pad_token_id

    corr = []
    loss = nn.CrossEntropyLoss(ignore_index=pid, reduction="none")

    for r in tqdm(data):
        prefix = r["premise"].strip() + " "
        mc_texts = []
        for choice_i in range(2):
            mc_texts.append(prefix + r[f"choice{choice_i + 1}"].strip())
        corr_i = int(r["label"])
        option_losses = torch.zeros(len(mc_texts))
        for mc_i, mc_text in enumerate(mc_texts):
            inputs = tokenizer([mc_text], add_special_tokens=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            if len(input_ids[0]) > max_seq_len:
                input_ids[0] = input_ids[0][:max_seq_len]
                attention_mask[0] = attention_mask[0][:max_seq_len]
            input_ids = torch.tensor(input_ids).to(device)
            attention_mask = torch.tensor(attention_mask).to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            logits = outputs["logits"].detach()
            del outputs
            labels = input_ids[:, 1:]
            logits = logits[:, :-1, :]
            logits = torch.transpose(logits, 1, 2)
            losses = loss(logits, labels).cpu()
            option_loss = torch.sum(losses, dim=-1).item()
            option_losses[mc_i] = option_loss
        print(option_losses)
        pred_i = int(torch.argmin(option_losses).item())
        corr.append(pred_i == corr_i)
    return float(np.mean(np.array(corr)))
