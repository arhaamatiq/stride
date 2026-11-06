#!/usr/bin/env python3
"""Builds a small byte-level BPE tokenizer.json with the Llama-3 structure.

Same split regex, ByteLevel pre-tokenizer and decoder, and Llama special token
names as the real tokenizer, but about 1000 ids so it fits the tiny test model
(vocab 1024). Server tests use it; test_tokenizer.py also checks it against HF
with ignore_merges off, which the real file does not exercise.
"""
import argparse
import glob
import os
import sysconfig

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
SPECIALS = [
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
    "<|eom_id|>",
    "<|python_tag|>",
]


def corpus():
    stdlib = sysconfig.get_paths()["stdlib"]
    with open(os.path.join(stdlib, "pydoc_data", "topics.py"), encoding="utf-8") as f:
        text = f.read()
    yield from (text[i : i + 2000] for i in range(0, 400_000, 2000))
    for path in sorted(glob.glob(os.path.join(stdlib, "*.py")))[:40]:
        with open(path, encoding="utf-8", errors="replace") as f:
            yield f.read()[:20_000]
    yield "café naïve 日本語 \U0001f600 ½ 12345 67890"


def build(out_dir: str, vocab_size: int = 1000, ignore_merges: bool = False) -> str:
    tok = Tokenizer(models.BPE(ignore_merges=ignore_merges))
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(PATTERN), "isolated", False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size - len(SPECIALS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(corpus(), trainer)
    tok.add_special_tokens(SPECIALS)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "tokenizer.json")
    tok.save(path)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--ignore-merges", action="store_true")
    args = parser.parse_args()
    print(build(args.out, args.vocab_size, args.ignore_merges))
