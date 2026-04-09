from collections import defaultdict
from datetime import datetime
from pathlib import Path
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

from tokefx.data import PUDSequence, PUDCandidate, PUD_Data, write_candidates_jsonl
from tokefx.encoder import UDEncoder
from tokefx.utils import log

from icecream import ic


class PatchScopesAnalyzer:
    def __init__(
        self,
        tokenizer_spec: str,
        model_spec: str,
        ablation: str,
        ablation_map: dict,
        embed_mode: bool = False,
        num_tokens_to_generate: int = 10,
        patchscopes_prompt: str = "X, X, X, X, ",
        prompt_placeholder: str = "X",
        device="cpu",
        seed: int = 42,
        **kwargs,
    ):
        self.model_spec = model_spec
        self.num_tokens_to_generate = num_tokens_to_generate

        self.tokenizer_spec = tokenizer_spec
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_spec)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.ablation = ablation
        self.ablation_map = ablation_map

        self.embed_mode = embed_mode

        self.model = AutoModelForCausalLM.from_pretrained(
            model_spec,
            attn_implementation="eager",
        )
        self.device = torch.device(device)
        self.model.to(self.device).eval()

        self.prompt_placeholder = prompt_placeholder
        self.patchscopes_prompt = patchscopes_prompt
        self.probe_prompt_enc, self.probe_prompt_idx = self._build_template()
        self.probe_prompt_enc = torch.tensor(self.probe_prompt_enc).to(self.device)

        self.probe_prompt_embed = self.model.get_input_embeddings()(
            self.probe_prompt_enc
        ).unsqueeze(0)

        self.seed = seed

    def _build_template(self) -> tuple[dict, list]:
        parts = self.patchscopes_prompt.split(self.prompt_placeholder)
        input_ids = []
        placeholder_indices = []

        for i, part in enumerate(parts):
            part_ids = self.tokenizer.encode(part, add_special_tokens=False)
            input_ids.extend(part_ids)
            if i < len(parts) - 1:
                placeholder_indices.append(len(input_ids))
                input_ids.append(0)  # dummy token ID replaced at the probe stage
        return input_ids, placeholder_indices

    def _collector(self, model_input: dict) -> dict:
        """my precious..."""
        blocks = self.model.model.layers

        collection = defaultdict(dict)
        hooks = []

        def save(layer_idx, site, hidden_states):
            collection[layer_idx][site] = hidden_states[0, -1].detach().clone()

        if not self.embed_mode:
            for layer_idx, block in enumerate(blocks):

                def post_attn_hook(module, inputs, output, layer_idx=layer_idx):
                    attn_out = output[0] if isinstance(output, tuple) else output
                    save(layer_idx, "post_attn", attn_out)

                def ffn_out_hook(module, inputs, output, layer_idx=layer_idx):
                    # NOTE: not sure why [0] is not required here
                    save(layer_idx, "ffn_out", output)

                def layer_out_hook(module, inputs, output, layer_idx=layer_idx):
                    save(layer_idx, "layer_out", output[0])

                # after adding residuals
                # hooks.append(block.mlp.register_forward_pre_hook(post_attn_hook))
                # before adding residuals
                # hooks.append(block.self_attn.register_forward_hook(post_attn_hook))

                # hooks.append(block.mlp.register_forward_hook(ffn_out_hook))
                hooks.append(block.register_forward_hook(layer_out_hook))

        def input_embed_hook(module, inputs):
            save(0, "input_emb", inputs[0])

        hooks.append(blocks[0].register_forward_pre_hook(input_embed_hook))

        # ==============
        # ablation logic
        # ==============
        #
        # NOTE: together with the latter forward pass, this is for troubleshooting
        # with torch.no_grad():
        #     out_clean = self.model(**model_input).logits.detach().clone()

        if self.ablation in {"targeted", "random"}:
            cfg = self.model.config
            n_heads = getattr(cfg, "num_attention_heads")
            hidden = getattr(cfg, "hidden_size")
            head_dim = hidden // n_heads
            assert hidden % n_heads == 0
        if self.ablation == "targeted":
            ablation_map = self.ablation_map.copy()
        elif self.ablation == "random":
            random.seed(self.seed)
            ablation_map = {}
            for layer_idx, heads in self.ablation_map.items():
                ablation_map[layer_idx] = random.sample(
                    [x for x in range(n_heads) if x not in heads],
                    k=min(len(heads), n_heads - len(heads)),
                )
        else:
            ablation_map = {}

        for layer_idx, heads in ablation_map.items():
            o_proj = blocks[layer_idx].self_attn.o_proj

            def ablation_hook(
                module, inputs, heads=heads, head_dim=head_dim, layer_idx=layer_idx
            ):
                x = inputs[0].clone()
                for h in heads:
                    s = h * head_dim
                    e = (h + 1) * head_dim
                    x[..., s:e] = 0
                return (x,) + inputs[1:]

            def mlp_out_hook(module, inputs, output, layer_idx=layer_idx):
                y = output.clone()
                y.zero_()
                return y

            hooks.append(o_proj.register_forward_pre_hook(ablation_hook))
            # FFN out ablation hook
            # NOTE: for debugging making sure that ablation works
            # if layer_idx < 2:
            #     hooks.append(blocks[layer_idx].mlp.register_forward_hook(mlp_out_hook))

        # with torch.no_grad():
        #     out_ablate = self.model(**model_input).logits.detach().clone()
        # log(f"logit diff: {(out_clean - out_ablate).abs().max()}")

        # then we perform a forward call using our hooks
        self.model(**model_input)

        for hook in hooks:
            hook.remove()

        return collection

    def _run_probe(self, representation) -> str:
        # get input embeddings for the prompt
        inputs_embeds = self.probe_prompt_embed.detach().clone()

        # patch each placeholder idx with representation
        for i in self.probe_prompt_idx:
            inputs_embeds[:, i, :] = representation.unsqueeze(0).to(self.device)
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], dtype=torch.long, device=self.model.device
        )

        # generate n new tokens
        out = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=self.num_tokens_to_generate,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False,
            num_beams=1,
            top_p=1.0,
            temperature=None,
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)

    def _probe_loop(self, token, collection) -> dict:
        rows = []
        for layer_idx, components in collection.items():
            # NOTE: uncomment for quicker troubleshooting
            # if layer_idx > 8:
            #     break
            for component_label, representation in components.items():
                pred = self._run_probe(representation)
                retrieved = 1 if token.form.lower() in pred.lower() else 0
                row = {
                    "layer": layer_idx,
                    "component": component_label,
                    "retrieved": retrieved,
                    "token_idx": token.idx,
                    "target": token.form,
                    "prediction": pred.replace("\n", " "),
                }
                rows.append(row)
        return rows

    def __call__(self, cand: dict, **kwargs):
        collection = self._collector(cand["model_inp"])
        return self._probe_loop(cand["token"], collection)


def create_embed_candidates(
    configurations: list[tuple],
    ud_base: Path,
    out_embed_jsonl: Path,
    out_embed_summary: Path,
    out_full: Path,
    **kwargs,
):
    """returns candidates which can't be reconstructed from the input embed layer"""
    candidates = []
    summary_rows = []
    full_rows = []
    for model_handle, lang_handle in configurations:
        model_spec, tokenizer_spec = model_handle
        lang, lang_spec = lang_handle

        datafp = ud_base / lang_spec["pud-conllu"]
        data = PUD_Data(datafp=datafp)

        run_kwargs = kwargs.copy()
        run_kwargs["model_spec"] = model_spec
        run_kwargs["tokenizer_spec"] = tokenizer_spec
        run_kwargs["patchscopes_prompt"] = lang_spec["patchscopes_prompt"]
        run_kwargs["ablation"] = "clean"
        run_kwargs["ablation_map"] = {}
        run_kwargs["embed_mode"] = True
        analyzer = PatchScopesAnalyzer(**run_kwargs)
        encoder = UDEncoder(**run_kwargs)

        recoverable = 0
        non_recoverable = 0
        processed_tokens = set()
        for cand in encoder.get_candidates(data, "in_boundary"):
            tform = cand["token"].form.lower()
            if non_recoverable >= kwargs["n_rows"]:
                break
            if tform in processed_tokens:
                continue
            res = analyzer(cand)
            embed_row = None
            for row in res:
                if row["component"] == "input_emb":
                    embed_row = row
                    break
            if not embed_row:
                continue
            full_rows.append(
                {
                    "lang": lang,
                    "model": model_spec,
                    "token": tform,
                    "token_debug": embed_row["target"],
                    "recoverable": embed_row["retrieved"],
                    "prediction": embed_row["prediction"],
                }
            )
            if embed_row["retrieved"]:
                recoverable += 1
            else:
                non_recoverable += 1
            processed_tokens.add(tform)

            candidates.append(
                PUDCandidate(
                    lang=lang,
                    model=model_spec,
                    token=cand["token"],
                    seq=cand["seq"],
                )
            )
            if not embed_row["retrieved"]:
                log(f"{non_recoverable}/{kwargs['n_rows']} processed {tform}")
            else:
                log(f"retrieved {tform=} from embedding layer")
        del analyzer
        summary_rows.append(
            {
                "lang": lang,
                "model_spec": model_spec,
                "retrieve_rate": recoverable / (non_recoverable + recoverable),
            }
        )
    write_candidates_jsonl(candidates, out_embed_jsonl)
    pd.DataFrame(summary_rows).drop_duplicates().to_csv(
        out_embed_summary,
        sep="\t",
        index=False,
    )
    pd.DataFrame(full_rows).to_parquet(out_full, index=False)
