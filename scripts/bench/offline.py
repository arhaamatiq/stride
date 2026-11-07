#!/usr/bin/env python3
"""Raw engine throughput with no HTTP in the path: stride_cli against vLLM's LLM class.

Both get the same token-id prompts, 256 generated tokens with EOS ignored, and greedy
decoding. A first pass for each size warms clocks and is discarded.
"""
import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile

SIZES = (1, 8, 64, 128)


def make_prompts(count, length, seed=21):
    rng = random.Random(seed)
    return [[128000] + [rng.randint(1000, 100000) for _ in range(length - 1)] for _ in range(count)]


def vllm_worker(args):
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.85, enable_prefix_caching=False, seed=0, disable_log_stats=True)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new, ignore_eos=True)
    import time

    prompts = make_prompts(max(SIZES), args.prompt_tokens)
    out = {}
    for n in SIZES:
        reqs = [{"prompt_token_ids": p} for p in prompts[:n]]
        llm.generate(reqs, sp, use_tqdm=False)
        runs = []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            res = llm.generate(reqs, sp, use_tqdm=False)
            dt = time.perf_counter() - t0
            runs.append(sum(len(o.outputs[0].token_ids) for o in res) / dt)
        out[str(n)] = runs
    with open(args.worker_out, "w") as f:
        json.dump(out, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli")
    parser.add_argument("--model", required=True)
    parser.add_argument("--report")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--vllm-python", default=os.path.join(os.environ.get("BENCH_VENV", "/root/venv-vllm"), "bin", "python"))
    parser.add_argument("--worker-out")
    args = parser.parse_args()
    if args.worker_out:
        vllm_worker(args)
        return

    tmp = tempfile.mkdtemp()
    prompts = make_prompts(max(SIZES), args.prompt_tokens)
    stride = {}
    for n in SIZES:
        path = os.path.join(tmp, f"p{n}.jsonl")
        with open(path, "w") as f:
            for i, p in enumerate(prompts[:n]):
                f.write(json.dumps({"id": i, "tokens": p}) + "\n")
        cmd = [args.cli, "--path", "tc", "--model", args.model, "--input", path, "--output", os.path.join(tmp, "o.jsonl"), "--max-new", str(args.max_new),
               "--kv-gb", "16", "--concurrency", str(n), "--max-rows", "2048", "--prefill-chunk", "256", "--max-slots", "256"]
        runs = []
        for r in range(args.repeats + 1):
            d = json.loads(subprocess.run(cmd, check=True, capture_output=True, text=True).stderr.strip().splitlines()[-1])
            if r > 0:
                runs.append(d["generated"] / d["run_seconds"])
        stride[str(n)] = runs
        print("stride", n, runs, flush=True)
    worker_out = os.path.join(tmp, "vllm.json")
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    with open(args.report + ".vllm.log", "w") as log:
        subprocess.run([args.vllm_python, os.path.abspath(__file__), "--model", args.model, "--worker-out", worker_out, "--repeats", str(args.repeats),
                        "--max-new", str(args.max_new), "--prompt-tokens", str(args.prompt_tokens)], check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
    vllm = json.load(open(worker_out))
    points = []
    for n in SIZES:
        a, b = stride[str(n)], vllm[str(n)]
        points.append({"batch": n, "stride_tok_per_s": statistics.mean(a), "stride_std": statistics.stdev(a), "stride_runs": a,
                       "vllm_tok_per_s": statistics.mean(b), "vllm_std": statistics.stdev(b), "vllm_runs": b, "stride_pct_of_vllm": 100.0 * statistics.mean(a) / statistics.mean(b)})
        print(points[-1], flush=True)
    with open(args.report, "w") as f:
        json.dump({"prompt_tokens": args.prompt_tokens, "max_new": args.max_new, "repeats": args.repeats, "points": points}, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    sys.exit(main())
