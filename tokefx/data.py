from datetime import datetime
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Iterator, Optional
import random

from transformers import AutoTokenizer
import conllu

FORM_EXCEPTIONS = {
    "'s",
    "'",
}


@dataclass
class PUDToken:
    idx: int
    form: str
    upos: str
    deprel: str
    head: int
    space_after: bool

    @classmethod
    def from_dict(cls, data: dict) -> "PUDToken":
        return cls(**data)


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

    def text_til_i(self, tgt: int) -> str:
        parts = []
        for i, tok in enumerate(self.pud_tokens):
            parts.append(tok.form)
            if i == tgt:
                break
            if tok.space_after and tok.form not in FORM_EXCEPTIONS:
                parts.append(" ")
        return "".join(parts)

    def full_text(self) -> str:
        """returns string of how the sequence would look like in text"""
        parts = []
        for tok in self.pud_tokens:
            parts.append(tok.form)
            if tok.space_after:
                parts.append(" ")
        return "".join(parts).rstrip()

    def to_dict(self) -> dict:
        return {
            "sentence_id": self.sentence_id,
            "pud_tokens": [asdict(tok) for tok in self.pud_tokens],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PUDSequence":
        return cls(
            sentence_id=data["sentence_id"],
            pud_tokens=[PUDToken.from_dict(tok) for tok in data["pud_tokens"]],
        )


@dataclass
class PUDCandidate:
    lang: str
    model: str
    token: PUDToken
    seq: PUDSequence

    @property
    def seq_id(self) -> str:
        return self.seq.sentence_id

    def to_dict(self) -> dict:
        return {
            "lang": self.lang,
            "model": self.model,
            "seq_id": self.seq.sentence_id,
            "token": asdict(self.token),
            "seq": self.seq.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PUDCandidate":
        return cls(
            lang=data["lang"],
            model=data["model"],
            token=PUDToken.from_dict(data["token"]),
            seq=PUDSequence.from_dict(data["seq"]),
        )


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

        # filter out multi-word tokens
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


def write_candidates_jsonl(candidates: list[PUDCandidate], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for cand in candidates:
            json.dump(cand.to_dict(), f, ensure_ascii=False, separators=(",", ":"))
            f.write("\n")


def iter_candidates_jsonl(path: Path, model: str, lang: str) -> Iterator[PUDCandidate]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            candidate = PUDCandidate.from_dict(json.loads(line.strip()))
            if candidate.model != model:
                continue
            if candidate.lang != lang:
                continue
            yield candidate
