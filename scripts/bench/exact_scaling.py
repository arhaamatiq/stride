#!/usr/bin/env python3
"""Exact-path throughput by batch size, on the prompts scripts/bench/offline.py uses."""
import argparse
import json
import os
import statistics
import subprocess
import tempfile

from offline import make_prompts

SIZES = (1, 8, 64, 128)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    tmp = tempfile.mkdtemp()
    prompts = make_prompts(max(SIZES), 256)
    points = []
    for n in SIZES:
        path = os.path.join(tmp, f"p{n}.jsonl")
        with open(path, "w") as f:
            for i, p in enumerate(prompts[:n]):
                f.write(json.dumps({"id": i, "tokens": p}) + "\n")
        cmd = [args.cli, "--path", "exact", "--model", args.model, "--input", path, "--output", os.path.join(tmp, "o.jsonl"), "--max-new", "256",
               "--kv-gb", "16", "--concurrency", str(n), "--max-rows", "2048", "--prefill-chunk", "256", "--max-slots", "256"]
        runs = []
        for r in range(args.repeats + 1):
            d = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stderr.strip().splitlines()[-1])
            if r > 0:
                runs.append(d["generated"] / d["run_seconds"])
        points.append({"batch": n, "tok_per_s": statistics.mean(runs), "std": statistics.stdev(runs), "runs": runs})
        print(points[-1], flush=True)
    with open(args.report, "w") as f:
        json.dump({"path": "exact", "prompt_tokens": 256, "max_new": 256, "repeats": args.repeats, "points": points}, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
