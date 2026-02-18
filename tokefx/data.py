from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from conllu import parse_incr
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from icecream import ic


class TokenList:
    def __init__(
        self,
        data_file: Path,
        tokenizer: Tokenizer,
        min_tokens: int = 1,
        min_bytes: int = 1,
        max_tokens: int = 0,
    ):
        self.tokenizer = tokenizer

        # controls at least how many tokens/bytes each "Token" yielded should be
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.min_bytes = min_bytes

        self.token_list = self._parse_tokens(data_file)

    def __iter__(self):
        for x in self.token_list:
            yield x

    def _parse_tokens(self, data_file: Path) -> Iterable:
        with open(data_file) as f:
            for seq in parse_incr(f):
                for token in seq:
                    if not (tokens := self._parse_token(str(token))):
                        continue
                    yield (token["form"], tokens)

    def _parse_token(self, token: str):
        if len(token.encode("utf-8")) <= self.min_bytes:
            return
        sub_tokens = self.tokenizer.encode(token)
        if len(sub_tokens) <= self.min_tokens:
            return
        if self.max_tokens != 0 and self.max_tokens <= len(sub_tokens):
            return
        return sub_tokens
