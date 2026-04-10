#!/usr/bin/env python3
import regex as re
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers import PreTokenizedString
from transformers import AutoTokenizer

en_text = "It's always 5 o'clock in New Orleans"
zh_text = "火"


# straight from https://github.com/openai/gpt-2/blob/master/src/encoder.py
def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


BYTE_ENC = bytes_to_unicode()


def pre_tok(s: str):
    b = s.encode("utf-8")
    return "".join(BYTE_ENC[x] for x in b)


# straight from https://github.com/openai/gpt-2/blob/master/src/encoder.py
pat = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def print_info(text: str):
    print(text)
    print("pretok chunks:", pat.findall(text))
    print(
        "pretok enc:",
        [pre_tok(x) for x in pat.findall(text)],
    )
    print(
        "pretok hex:",
        [pre_tok(x).encode("utf-8").hex(" ").upper() for x in pat.findall(text)],
    )


print_info(en_text)
print_info(zh_text)
