#!/usr/bin/env python3
from pathlib import Path
import json

import codecs
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
from tqdm import tqdm

from tokefx.constants import SPECIAL_TOKEN_RULES


def belebele(
    data: Path,
    model_spec: str,
    tokenizer_spec: str,
    device: torch.device = torch.device("cpu"),
    max_seq_len: int = 4096,
) -> float:
    model = AutoModelForCausalLM.from_pretrained(model_spec).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_spec)

    dataset = []
    infile = codecs.open(data, "rb", encoding="utf-8")
    for line in infile:
        dataset.append(json.loads(line.strip()))
    infile.close()
    assert len(dataset) == 900

    # TODO: fix to include model specific bos token behavior
    # TODO: need to check what happens for every model
    # prepend_token_id = tokenizer.cls_token_id
    # NOTE: I think just removing the prepend token id is fine since it's a goldfish thing
    pid = -100 if tokenizer.pad_token_id is None else tokenizer.pad_token_id

    corr = []
    loss = nn.CrossEntropyLoss(ignore_index=pid, reduction="none")

    for r in tqdm(dataset):
        prefix = r["flores_passage"].strip() + " " + r["question"].strip() + " "
        mc_texts = []
        for ans_i in range(4):
            mc_texts.append(prefix + r[f"mc_answer{ans_i + 1}"].strip())
        corr_i = int(r["correct_answer_num"]) - 1

        option_losses = torch.zeros(len(mc_texts))
        for mc_i, mc_text in enumerate(mc_texts):
            inputs = tokenizer([mc_text], add_special_tokens=False)
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

        pred_i = int(torch.argmin(option_losses).item())
        corr.append(pred_i == corr_i)
    return float(np.mean(np.array(corr)))
