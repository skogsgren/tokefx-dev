from icecream import ic
from transformers import AutoTokenizer

tokenizers = [
    "CohereLabs/aya-expanse-8b",
    "gpt2",
    "google/byt5-small",
    "facebook/xglm-564M",
    "Qwen/Qwen2.5-0.5B",
]
vocab_sizes = {}

for spec in tokenizers:
    tokenizer = AutoTokenizer.from_pretrained(spec)
    vocab_sizes[spec] = tokenizer.vocab_size
ic(vocab_sizes)
