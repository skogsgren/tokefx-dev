from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import conllu
from transformers import AutoTokenizer


@dataclass
class PUDToken:
    idx: int
    form: str
    upos: str
    input_ids: list[int]
    space_after: bool


@dataclass
class PUDSequence:
    sentence_id: int
    pud_tokens: list[PUDToken]

    def __iter__(self) -> Iterator[PUDToken]:
        for token in self.pud_tokens:
            yield token

    def __getitem__(self, i: int):
        return self.pud_tokens[i]

    def get_text_til_i(self, idx: int):
        text = ""
        for i in range(idx + 1):  # +1 since 0 indexing is a thing
            text += self.pud_tokens[i].form
            # since spaces are added after and not before tokens we have to
            # make sure we don't add a space after the index we're after
            if self.pud_tokens[i].space_after and i != idx:
                text += " "
        return text


class PUD_Data:
    def __init__(
        self,
        datafp: Path,
        tokenizer_spec: str,
        context_window: int = 4096,
        add_special_tokens: bool = False,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_spec)
        self.context_window = context_window
        self.datafp = datafp
        self.comparison_mode = "intra_token"
        self.add_special_tokens = add_special_tokens

        self._seqs: list | None = None

    def _parse_seq(self, seq: conllu.models.TokenList) -> PUDSequence:
        sequence_id = seq.metadata["parallel_id"]
        pud_tokens = []
        prev_space_after = False  # at start of sequence we don't have a space
        for i, tok in enumerate(seq):
            space_after = (tok.get("misc") or {}).get("SpaceAfter") != "No"
            form = " " + tok["form"] if prev_space_after else tok["form"]
            input_ids = self.tokenizer(
                form,
                add_special_tokens=self.add_special_tokens,
                truncation=True,
                max_length=self.context_window,
            )["input_ids"]
            pud_tokens.append(
                PUDToken(i, tok["form"], tok["upos"], input_ids, space_after)
            )
            prev_space_after = space_after
        return PUDSequence(sequence_id, pud_tokens)

    def _load_seqs(self):
        if self._seqs is None:
            text = self.datafp.read_text()
            self._seqs = conllu.parse(text)
        return self._seqs

    def __len__(self) -> int:
        return len(self._load_seqs())

    def __iter__(self) -> Iterator[PUDSequence]:
        for seq in self._load_seqs():
            yield self._parse_seq(seq)
