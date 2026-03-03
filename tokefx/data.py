from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional
import random

from transformers import AutoTokenizer
import conllu


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

    def prefix_before_token(self, idx: int) -> str:
        if idx < 0:
            return ""
        parts: list[str] = []
        for i in range(idx):
            tok = self.pud_tokens[i]
            parts.append(tok.form)
            if tok.space_after:
                parts.append(" ")
        return "".join(parts)

    def full_text(self) -> str:
        parts = []
        for tok in self.pud_tokens:
            parts.append(tok.form)
            if tok.space_after:
                parts.append(" ")
        return "".join(parts).rstrip()


class PUD_Data:
    def __init__(self, datafp: Path, seed: int = 42):
        self.datafp = datafp
        self._seqs = None
        self.seed: int = seed

    def _load_seqs(self):
        if self._seqs is None:
            text = self.datafp.read_text(encoding="utf-8")
            self._seqs = conllu.parse(text)
            random.seed(self.seed)
            random.shuffle(self._seqs)
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
