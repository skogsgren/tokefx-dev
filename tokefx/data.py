from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import conllu
from transformers import AutoTokenizer

from icecream import ic


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

    def get_text_til_i(self, idx: int):
        text = ""
        for i in range(idx):
            text += self.pud_tokens[i].form
            # since spaces are added after and not before tokens we have to
            # make sure we don't add a space after the index we're after
            if self.pud_tokens[i].space_after and i != idx - 1:
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
        self.add_special_tokens = add_special_tokens

    def _parse_seq(self, seq: conllu.models.TokenList) -> PUDSequence:
        sequence_id = seq.metadata["parallel_id"]
        pud_tokens = []
        for i, tok in enumerate(seq):
            space_after = (tok.get("misc") or {}).get("SpaceAfter") != "No"
            input_ids = self.tokenizer(
                tok["form"],
                add_special_tokens=self.add_special_tokens,
                truncation=True,
                max_length=self.context_window,
            )["input_ids"]
            pud_tokens.append(
                PUDToken(i, tok["form"], tok["upos"], input_ids, space_after)
            )
        return PUDSequence(sequence_id, pud_tokens)

    def __iter__(self) -> Iterator[PUDSequence]:
        text = self.datafp.read_text()
        for sent_idx, seq in enumerate(conllu.parse(text)):
            yield self._parse_seq(seq)


if __name__ == "__main__":
    tokenizer_spec = "google/byt5-small"
    tokenizer_spec = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_spec)

    for n in range(2, 5):
        for lang in ["en", "zh", "tr"]:
            length = 0
            data = PUD_Data(
                datafp=Path(f"./{lang}_pud-ud-test.conllu"),
                tokenizer_spec=tokenizer_spec,
            )
            for i, x in enumerate(data):
                for tok in x.pud_tokens:
                    if len(tok.input_ids) != n:
                        continue
                    length += 1
                    if length <= 2:
                        ic(n, lang, tok.form, tok.input_ids)
            print(f"({lang.upper()}, {n})\t{length}")
