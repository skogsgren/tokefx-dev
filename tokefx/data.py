from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from statistics import fmean
from transformers import AutoTokenizer
import conllu

from tokefx.utils import token_len


@dataclass
class PUDToken:
    idx: int
    form: str
    upos: str
    deprel: str
    head: int
    space_after: bool


@dataclass
class PUDSequence:
    sentence_id: str
    pud_tokens: list[PUDToken]

    def __iter__(self) -> Iterator[PUDToken]:
        for token in self.pud_tokens:
            yield token

    def __getitem__(self, i: int) -> PUDToken:
        return self.pud_tokens[i]

    def __len__(self) -> int:
        return len(self.pud_tokens)

    def text_until_token(self, idx: int) -> str:
        if idx < 0:
            return ""
        if idx >= len(self.pud_tokens):
            raise IndexError(
                f"idx={idx} out of range for sequence length {len(self.pud_tokens)}"
            )

        parts: list[str] = []
        for i in range(idx + 1):
            tok = self.pud_tokens[i]
            parts.append(tok.form)
            if tok.space_after and i != idx:
                parts.append(" ")
        return "".join(parts)

    def full_text(self) -> str:
        if not self.pud_tokens:
            return ""
        return self.text_until_token(len(self.pud_tokens) - 1)


class PUD_Data:
    def __init__(self, datafp: Path):
        self.datafp = datafp
        self._seqs: Optional[list] = None

    def _load_seqs(self):
        if self._seqs is None:
            text = self.datafp.read_text(encoding="utf-8")
            self._seqs = conllu.parse(text)
        return self._seqs

    def _parse_seq(self, seq: conllu.models.TokenList) -> PUDSequence:
        sentence_id = seq.metadata["parallel_id"]
        pud_tokens: list[PUDToken] = []

        # keep only syntactic words and skip range IDs / empty nodes.
        filtered = [tok for tok in seq if isinstance(tok.get("id"), int)]
        for i, tok in enumerate(filtered):
            space_after = (tok.get("misc") or {}).get("SpaceAfter") != "No"
            pud_tokens.append(
                PUDToken(
                    idx=i,
                    form=str(tok["form"]),
                    upos=str(tok["upos"]),
                    space_after=space_after,
                    deprel=tok["deprel"],
                    head=int(tok["head"]) - 1,
                )
            )
        return PUDSequence(sentence_id=str(sentence_id), pud_tokens=pud_tokens)

    def __len__(self) -> int:
        return len(self._load_seqs())

    def __iter__(self) -> Iterator[PUDSequence]:
        for seq in self._load_seqs():
            parsed = self._parse_seq(seq)
            if len(parsed) == 0:
                continue
            yield parsed


def get_rho(cfg: dict) -> list[dict[str, object]]:
    """compute rho = mean(num_subtokens / token_bytes) for each (model, lang)"""
    rho_rows = []
    for model_name, tokenizer_spec in cfg["eval"]["models"]:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_spec)
        for lang, spec in cfg["lang"].items():
            time = datetime.now()
            print(f"{time} calculating rho for {lang=} {model_name=} {tokenizer_spec=}")
            data = PUD_Data(datafp=cfg["dir"]["ud_base"] / spec["pud-conllu"])
            total_subtokens = 0
            total_bytes = 0
            for seq in data:
                prev_end = 0  # == len(tokenize(prefix up to previous token))
                for i, tok in enumerate(seq):
                    if cfg["eval"]["skip_punct"] and tok.upos == "PUNCT":
                        continue
                    if tok.upos == "FW":
                        continue
                    end = token_len(
                        tokenizer=tokenizer,
                        text=seq.text_until_token(i),
                        context_window=cfg["eval"]["context_window"],
                        add_special_tokens=cfg["eval"]["add_special_tokens"],
                    )
                    n_raw = end - prev_end
                    prev_end = end
                    token_bytes = len(tok.form.encode("utf-8"))
                    if n_raw == 0 or token_bytes == 0:
                        continue
                    total_subtokens += n_raw
                    total_bytes += token_bytes
            print(f"\t{lang=} {tokenizer_spec=} {total_subtokens=} {total_bytes=}")
            rho = total_subtokens / total_bytes
            rho_rows.append({"model": model_name, "lang": lang, "rho": rho})
    return rho_rows


def get_ref(rho: list[dict], lang: str) -> dict[str, float]:
    """helper function to make looping over a reference language easier"""
    res = {}
    for row in rho:
        if row["lang"] != lang:
            continue
        res[row["model"]] = row["rho"]
    return res
