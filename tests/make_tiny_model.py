#!/usr/bin/env python3
"""Creates a small random Llama checkpoint with the same structure as Llama 3.2.

The structure matches the real model: 4 query heads per KV head, explicit
head_dim, llama3 RoPE scaling, tied embeddings, BF16 weights. CI uses it so every
correctness test runs without gated weights or a GPU.
"""
import argparse
import json
import os

import torch
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM


def build(out_dir: str, seed: int) -> None:
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=1024,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=3,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=2048,
        rope_theta=500000.0,
        rope_scaling={
            "rope_type": "llama3",
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
        tie_word_embeddings=True,
        attention_bias=False,
        mlp_bias=False,
        rms_norm_eps=1e-5,
        initializer_range=0.2,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = LlamaForCausalLM(cfg)
    # Default init leaves every norm gain at exactly 1.0 and attention nearly silent,
    # which would hide whole classes of bugs. Make every component matter.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "norm" in name:
                param.copy_(1.0 + 0.2 * torch.randn_like(param))
    model = model.to(torch.bfloat16)
    os.makedirs(out_dir, exist_ok=True)
    state = {k: v.clone().contiguous() for k, v in model.state_dict().items() if k != "lm_head.weight"}
    save_file(state, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    build(args.out, args.seed)
