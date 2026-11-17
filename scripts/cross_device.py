#!/usr/bin/env python3
"""Compares exact-path outputs produced on different devices, byte for byte.

Each input is a directory written by `stride_cli --full --dump-logits`. Run the
same prompts on each device, collect the outputs, then point this script at them:
    cross_device.py --report reports/g1/cross_device.json gpu=dirA x86=dirB arm=dirC
"""
import argparse
import hashlib
import json
import os
import sys


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("runs", nargs="+", help="label=directory containing out.jsonl, logits.bin, summary.json")
    args = parser.parse_args()

    runs = {}
    for spec in args.runs:
        label, path = spec.split("=", 1)
        summary = json.load(open(os.path.join(path, "summary.json")))
        rows = [json.loads(l) for l in open(os.path.join(path, "out.jsonl"))]
        runs[label] = {
            "device": summary["device"],
            "host": open(os.path.join(path, "host.txt")).read().strip() if os.path.exists(os.path.join(path, "host.txt")) else "",
            "tokens_per_second": summary["generated"] / summary["run_seconds"],
            "generated_tokens": summary["generated"],
            "logits_bytes": os.path.getsize(os.path.join(path, "logits.bin")),
            "logits_sha256": sha256(os.path.join(path, "logits.bin")),
            "fingerprints": [r["fingerprint"] for r in rows],
        }
    first = next(iter(runs.values()))
    identical = all(r["logits_sha256"] == first["logits_sha256"] and r["fingerprints"] == first["fingerprints"] for r in runs.values())
    report = {"byte_identical": identical, "logit_values_compared": first["logits_bytes"] // 4, "runs": runs}
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print("byte_identical:", identical, "logit values:", first["logits_bytes"] // 4)
    for label, r in runs.items():
        print(f"  {label:6s} {r['device'][:34]:34s} {r['tokens_per_second']:9.1f} tok/s  sha256 {r['logits_sha256'][:16]}")
    sys.exit(0 if identical else 1)


if __name__ == "__main__":
    main()
