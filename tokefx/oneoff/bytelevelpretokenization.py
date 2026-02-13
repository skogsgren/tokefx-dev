#!/usr/bin/env python3
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers import PreTokenizedString

pts = PreTokenizedString("可以跟你借一個火嗎？")
pretok = ByteLevel(add_prefix_space=False, use_regex=False)
pretok.pre_tokenize(pts)

for split, offsets, _ in pts.get_splits():
    print(repr(split), offsets)
