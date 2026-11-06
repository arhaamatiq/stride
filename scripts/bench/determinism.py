#!/usr/bin/env python3
"""Does a server's output for a fixed request change with load?

Sends the same temperature-0, fixed-seed requests first one at a time, then again
at increasing concurrency, and compares each response against its own sequential
baseline. Any difference can only come from what else was running at the same time.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests"))
from g1_real import prompts  # noqa: E402


def post(port, name, prompt, max_tokens):
    body = {
        "model": name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 0,
        "ignore_eos": True,
    }
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    return d["choices"][0]["message"]["content"], d.get("system_fingerprint")


def first_difference(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, choices=["stride", "vllm"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--levels", default="8,32,64,128")
    parser.add_argument("--stride-bin", default="/root/stride/build/stride_server")
    parser.add_argument("--stride-path", default="tc")
    parser.add_argument("--vllm-bin", default="/root/venv-vllm/bin/vllm")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    pin = subprocess.check_output([sys.executable, os.path.join(here, "..", "verify_weights.py"), "--dir", args.model], text=True).strip()
    name = "llama-3.2-1b"
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if args.engine == "stride":
        cmd = [args.stride_bin, "--model", args.model, "--host", "127.0.0.1", "--port", str(args.port), "--path", args.stride_path, "--kv-gb", "16", "--max-seqs", "256", "--max-rows", "2048", "--prefill-chunk", "256"]
    else:
        cmd = [args.vllm_bin, "serve", args.model, "--dtype", "bfloat16", "--max-model-len", "2048", "--no-enable-prefix-caching", "--seed", "0",
               "--gpu-memory-utilization", "0.85", "--served-model-name", name, "--host", "127.0.0.1", "--port", str(args.port), "--disable-log-stats", "--uvicorn-log-level", "warning"]
    log = open(args.report + ".server.log", "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, preexec_fn=os.setsid)
    try:
        for _ in range(300):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{args.port}/v1/models", timeout=2)
                break
            except Exception:
                if proc.poll() is not None:
                    raise SystemExit("server exited")
                time.sleep(1)

        texts = prompts()
        baseline = [post(args.port, name, t, args.max_tokens) for t in texts]
        repeat = [post(args.port, name, t, args.max_tokens) for t in texts]
        report = {
            "engine": args.engine,
            "server_command": cmd,
            "pinned_weights": pin,
            "prompts": len(texts),
            "max_tokens": args.max_tokens,
            "temperature": 0.0,
            "sequential_repeat_differs": sum(a[0] != b[0] for a, b in zip(baseline, repeat)),
            "levels": [],
        }
        print(f"{args.engine}: sequential rerun differs on {report['sequential_repeat_differs']}/{len(texts)}", flush=True)
        for level in [int(x) for x in args.levels.split(",")]:
            # Enough requests to hold the target concurrency; each prompt is checked every time it runs.
            count = max(len(texts), level * 4)
            order = [i % len(texts) for i in range(count)]
            with ThreadPoolExecutor(level) as pool:
                out = list(pool.map(lambda i: post(args.port, name, texts[i], args.max_tokens), order))
            differs = [(i, o) for i, o in zip(order, out) if o[0] != baseline[i][0]]
            prompts_changed = len({i for i, _ in differs})
            divergence = sorted(first_difference(baseline[i][0], o[0]) for i, o in differs)
            row = {
                "concurrency": level,
                "requests": count,
                "requests_differing": len(differs),
                "requests_differing_pct": 100.0 * len(differs) / count,
                "prompts_changed": prompts_changed,
                "prompts_changed_pct": 100.0 * prompts_changed / len(texts),
                "first_difference_char_p50": divergence[len(divergence) // 2] if divergence else None,
            }
            if args.engine == "stride":
                row["fingerprints_differing"] = sum(o[1] != baseline[i][1] for i, o in zip(order, out))
            report["levels"].append(row)
            print(f"{args.engine} c={level}: {len(differs)}/{count} requests differ, {prompts_changed}/{len(texts)} prompts changed", flush=True)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=30)
        except Exception:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    time.sleep(3)


if __name__ == "__main__":
    main()
