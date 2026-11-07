#!/usr/bin/env python3
"""Batch-1 decode against a memory-bandwidth roofline, with repeats.

The roofline is 1008 GB/s, the RTX 4090's specified peak DRAM bandwidth, divided by the
weight bytes one forward pass must read. It is a bound from a data-sheet figure, not a
counter measurement. KV and activation traffic are not counted, so a real engine cannot
reach 100%. The byte count comes from the safetensors header: every tensor once, which
counts tied embeddings once as the output projection.
"""
import argparse
import json
import os
import statistics
import struct
import subprocess

PEAK_GB_PER_S = 1008.0


def weight_bytes(model_dir):
    with open(os.path.join(model_dir, "model.safetensors"), "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
    return sum(v["data_offsets"][1] - v["data_offsets"][0] for k, v in header.items() if k != "__metadata__")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-new", type=int, default=256)
    args = parser.parse_args()
    bytes_per_pass = weight_bytes(args.model)
    ceiling = PEAK_GB_PER_S * 1e9 / bytes_per_pass
    result = {"peak_gb_per_s": PEAK_GB_PER_S, "peak_is_specified_not_measured": True, "weight_bytes_per_pass": bytes_per_pass, "roofline_tok_per_s": ceiling, "max_new": args.max_new, "paths": {}}
    for path in ("tc", "exact"):
        runs = []
        for _ in range(args.repeats + 1):
            cmd = [args.cli, "--path", path, "--model", args.model, "--input", args.prompt, "--output", "/tmp/roofline.jsonl", "--max-new", str(args.max_new), "--kv-gb", "2"]
            stderr = subprocess.run(cmd, check=True, capture_output=True, text=True).stderr
            runs.append(json.loads(stderr.strip().splitlines()[-1]))
        runs = runs[1:]  # the first run warms clocks and caches
        rate = [r["generated"] / r["run_seconds"] for r in runs]
        result["paths"][path] = {
            "repeats": args.repeats,
            "tok_per_s_mean": statistics.mean(rate),
            "tok_per_s_std": statistics.stdev(rate),
            "tok_per_s_runs": rate,
            "pct_of_roofline": 100.0 * statistics.mean(rate) / ceiling,
            "weight_gb_per_s_implied": statistics.mean(rate) * bytes_per_pass / 1e9,
        }
        print(path, result["paths"][path], flush=True)
    with open(args.report, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
