import argparse
import importlib
import json
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from tokefx.data import PUDCandidate


class LahisAnalyzer:
    def __init__(self, model_spec, device: str = "cpu", **kwargs):
        self.device = torch.device(device)

        self.model_spec = model_spec
        self.model = AutoModelForCausalLM.from_pretrained(self.model_spec).to(
            self.device
        )

        self.L = self.model.config.num_hidden_layers
        self.H = self.model.config.num_attention_heads

    def _register_hooks(self, head_mask: nn.Parameter):
        hooks = []
        for layer_idx, layer in enumerate(self.model.model.layers):
            o_proj = layer.self_attn.o_proj

            def make_pre_hook(layer_idx):
                def pre_hook(module, inputs):
                    x = inputs[0]  # [B, T, hidden]
                    b, t, c = x.shape

                    n_heads = head_mask.shape[1]
                    head_dim = c // n_heads

                    x = x.view(b, t, n_heads, head_dim)
                    x = x * head_mask[layer_idx].view(1, 1, n_heads, 1)
                    x = x.view(b, t, c)
                    return (x,)

                return pre_hook

            hooks.append(o_proj.register_forward_pre_hook(make_pre_hook(layer_idx)))

        return hooks

    def __call__(self, candidates) -> torch.tensor:
        head_mask = nn.Parameter(
            torch.ones(self.L, self.H, device=self.device, dtype=torch.float32),
            requires_grad=True,
        )

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        total_importance = torch.zeros_like(head_mask, dtype=torch.float32)
        neg_grad_counts = torch.zeros_like(head_mask, dtype=torch.int32)
        optimizer = torch.optim.AdamW([head_mask], lr=1e-3)

        hooks = self._register_hooks(head_mask)

        n = 0
        for enc, target_idx in candidates:
            outputs = self.model(**enc)
            logits = outputs.logits[0, -1]
            loss = -torch.log_softmax(logits, dim=-1)[target_idx]
            optimizer.zero_grad(set_to_none=True)

            loss.backward()
            grad = head_mask.grad

            # NOTE: 1000.0 for readability
            total_importance += (grad.abs() * head_mask * 1000.0).detach().float()
            neg_grad_counts += (grad * head_mask < 0).int().detach()
            optimizer.step()
            n += 1

        for hook in hooks:
            hook.remove()

        avg_importance = total_importance / n
        avg_neg_frac = neg_grad_counts.float() / n
        return avg_importance * avg_neg_frac

    def select_heads(self, condition_matrix, control_matrix, topk=None, quantile=None):
        flat = (condition_matrix - control_matrix).view(-1)
        if topk is not None:
            k = min(topk, flat.numel())
            vals, idx = torch.topk(flat, k=k)
        elif quantile is not None:
            thr = torch.quantile(flat, quantile)
            idx = torch.nonzero(flat > thr, as_tuple=False).squeeze(-1)
            vals = flat[idx]
            order = torch.argsort(vals, descending=True)
            idx, vals = idx[order], vals[order]
        else:
            raise ValueError("Set either --topk or --quantile.")

        rows = []
        for i in idx.tolist():
            rows.append(
                {"flat_idx": int(i), "layer": int(i // self.H), "head": int(i % self.H)}
            )
        return rows


def get_lahis_ablation_map(json_dir: Path) -> dict:
    global_ablation_map = {}
    for file in json_dir.glob("*.json"):
        pieces = file.name.split("_")
        model = pieces[0].split("-")[0] + "/" + "-".join(pieces[0].split("-")[1:])
        lang = pieces[1]
        if lang not in global_ablation_map:
            global_ablation_map[lang] = {}
        if model not in global_ablation_map[lang]:
            global_ablation_map[lang][model] = {}

        for row in json.loads(file.read_text()):  # i.e. each individual layer/head map
            if row["layer"] not in global_ablation_map[lang][model]:
                global_ablation_map[lang][model][row["layer"]] = []
            global_ablation_map[lang][model][row["layer"]].append(row["head"])
    return global_ablation_map
