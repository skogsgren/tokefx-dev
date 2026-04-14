#!/usr/bin/env python3
from itertools import product
from pathlib import Path
import sys

import pandas as pd
from transformers import AutoTokenizer

from tokefx.data import PUD_Data
from tokefx.encoder import PUDCandidate, UDEncoder
from tokefx.utils import load_config


def get_fertility(tokenizer, candidates: list[PUDCandidate]) -> float:
    n_bytes = 0
    n_tokens = 0
    for candidate in candidates:
        n_tokens += len(candidate["model_inp"]["input_ids"][0])
        n_bytes += len(
            tokenizer.decode(candidate["model_inp"]["input_ids"][0]).encode("utf-8")
        )
    return n_bytes / n_tokens


if __name__ == "__main__":
    CFG = load_config(Path(sys.argv[1]))

    configurations = []
    for model_handle, lang_handle in product(
        CFG["eval"]["models"],
        CFG["lang"].items(),
    ):
        model_name = model_handle[1]
        language = lang_handle[0]
        configurations.append((model_handle, lang_handle))

    df_rows = []
    BASE_KWARGS = CFG["eval"]
    for model_handle, lang_handle in configurations:
        model_spec, tokenizer_spec = model_handle
        lang, lang_spec = lang_handle
        datafp = CFG["dir"]["ud_base"] / lang_spec["pud-conllu"]
        data = PUD_Data(datafp=datafp)
        run_kwargs = BASE_KWARGS.copy()
        run_kwargs["tokenizer_spec"] = tokenizer_spec
        run_kwargs["lang"] = lang
        run_kwargs["ignored_pos"] = set(run_kwargs.get("ignored_pos", []))
        encoder = UDEncoder(**run_kwargs)

        candidates = []
        for i, cand in enumerate(encoder.get_candidates(data, "in_boundary")):
            if i >= BASE_KWARGS["n_rows"]:
                break
            candidates.append(cand)

        fert = get_fertility(AutoTokenizer.from_pretrained(tokenizer_spec), candidates)
        df_rows.append({"model": model_spec, "lang": lang, "fertility": round(fert, 2)})
    print(pd.DataFrame(df_rows).to_csv(sep="\t", index=False))
