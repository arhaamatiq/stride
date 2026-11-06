#!/usr/bin/env python3
"""Reference path: HF transformers with BF16 weights promoted to FP32 activations.

FP32 is the reference because BF16 activations round every intermediate to an
8-bit mantissa; FP32 rounding noise is about 2^16 times smaller, so it measures
engine errors rather than quantization noise. PyTorch is used only here, never in
the runtime path.

Writes JSONL (greedy tokens) and a raw FP32 logits dump in the same format
stride_cli produces, so the two can be compared directly.
"""
import argparse
import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM


@torch.no_grad()
def run(model, tokens, max_new, dump_steps, device):
    ids = torch.tensor([tokens], device=device)
    out = model(input_ids=ids, use_cache=True)
    past = out.past_key_values
    generated, dumps = [], []
    logits = out.logits[0, -1]
    for step in range(max_new):
        if step < dump_steps:
            dumps.append(logits.float().cpu().numpy())
        token = int(torch.argmax(logits).item())
        generated.append(token)
        if step + 1 == max_new:
            break
        out = model(input_ids=torch.tensor([[token]], device=device), past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1]
    return generated, dumps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dump-logits", default=None)
    parser.add_argument("--dump-steps", type=int, default=1)
    parser.add_argument("--max-new", type=int, default=16)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64", "bfloat16"])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    dtype = {"float32": torch.float32, "float64": torch.float64, "bfloat16": torch.bfloat16}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, attn_implementation="eager")
    model.to(args.device).eval()

    dump = open(args.dump_logits, "wb") if args.dump_logits else None
    with open(args.input) as f, open(args.output, "w") as out:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            tokens, dumps = run(model, row["tokens"], args.max_new, args.dump_steps, args.device)
            out.write(json.dumps({"id": row["id"], "tokens": tokens}) + "\n")
            if dump:
                for d in dumps:
                    dump.write(np.asarray(d, dtype=np.float32).tobytes())
    if dump:
        dump.close()


if __name__ == "__main__":
    main()
