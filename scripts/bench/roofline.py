#!/usr/bin/env python3
"""Batch-1 decode against the memory-bandwidth roofline, with repeats.

The roofline is peak DRAM bandwidth (1008 GB/s on an RTX 4090) divided by the weight
bytes one forward pass must read; the engine reports that byte count from the actual
matrices it loaded.
"""
import argparse
import json
import statistics
import subprocess

PEAK_GB_PER_S = 1008.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-new", type=int, default=256)
    args = parser.parse_args()
    result = {"peak_gb_per_s": PEAK_GB_PER_S, "max_new": args.max_new, "paths": {}}
    for path in ("tc", "exact"):
        runs = []
        for _ in range(args.repeats + 1):
            cmd = [args.cli, "--path", path, "--model", args.model, "--input", args.prompt, "--output", "/tmp/roofline.jsonl", "--max-new", str(args.max_new), "--kv-gb", "2"]
            stderr = subprocess.run(cmd, check=True, capture_output=True, text=True).stderr
            runs.append(json.loads(stderr.strip().splitlines()[-1]))
        runs = runs[1:]  # the first run warms clocks and caches
        rate = [r["generated"] / r["run_seconds"] for r in runs]
        bytes_per_pass = runs[0]["weight_bytes_per_pass"]
        ceiling = PEAK_GB_PER_S * 1e9 / bytes_per_pass
        result["paths"][path] = {
            "repeats": args.repeats,
            "tok_per_s_mean": statistics.mean(rate),
            "tok_per_s_std": statistics.stdev(rate),
            "tok_per_s_runs": rate,
            "weight_bytes_per_pass": bytes_per_pass,
            "roofline_tok_per_s": ceiling,
            "pct_of_roofline": 100.0 * statistics.mean(rate) / ceiling,
        }
        print(path, result["paths"][path], flush=True)
    with open(args.report, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
