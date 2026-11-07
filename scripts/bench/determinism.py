#!/usr/bin/env python3
"""Does a server's output for a fixed request change with load?

Sends the same temperature-0, fixed-seed requests first one at a time, then again
at increasing concurrency, and compares each response against its own sequential
baseline. Any difference can only come from what else was running at the same time.
Per-request hashes are kept, so the counts can be recomputed from the report.
"""
import argparse
import hashlib
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
    text = d["choices"][0]["message"]["content"]
    return {
        "sha": hashlib.sha256(text.encode()).hexdigest()[:24],
        "fingerprint": d.get("system_fingerprint"),
        "completion_tokens": d.get("usage", {}).get("completion_tokens"),
        "prompt_tokens": d.get("usage", {}).get("prompt_tokens"),
    }


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, choices=["stride", "vllm"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--levels", default="8,32,64,128")
    parser.add_argument("--stride-bin", default=os.path.join(here, "..", "..", "build", "stride_server"))
    parser.add_argument("--stride-path", default="tc")
    parser.add_argument("--kv-gb", default="16")
    parser.add_argument("--vllm-bin", default=os.path.join(os.environ.get("BENCH_VENV", "/root/venv-vllm"), "bin", "vllm"))
    args = parser.parse_args()

    pin = subprocess.check_output([sys.executable, os.path.join(here, "..", "verify_weights.py"), "--dir", args.model], text=True).strip()
    name = "llama-3.2-1b"
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if args.engine == "stride":
        cmd = [args.stride_bin, "--model", args.model, "--host", "127.0.0.1", "--port", str(args.port), "--path", args.stride_path, "--kv-gb", args.kv_gb,
               "--max-seqs", "256", "--max-rows", "2048", "--prefill-chunk", "256"]
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
            "sequential_repeat_differs": sum(a["sha"] != b["sha"] for a, b in zip(baseline, repeat)),
            "baseline": baseline,
            "levels": [],
        }
        print(f"{args.engine}: sequential rerun differs on {report['sequential_repeat_differs']}/{len(texts)}", flush=True)
        for level in [int(x) for x in args.levels.split(",")]:
            # Enough requests to hold the target concurrency; each prompt is checked every time it runs.
            count = max(len(texts), level * 4)
            order = [i % len(texts) for i in range(count)]
            with ThreadPoolExecutor(level) as pool:
                out = list(pool.map(lambda i: post(args.port, name, texts[i], args.max_tokens), order))
            differs = [i for i, o in zip(order, out) if o["sha"] != baseline[i]["sha"]]
            row = {
                "concurrency": level,
                "requests": count,
                "requests_differing": len(differs),
                "requests_differing_pct": 100.0 * len(differs) / count,
                "prompts_changed": len(set(differs)),
                "prompts_changed_pct": 100.0 * len(set(differs)) / len(texts),
                "responses": [{"prompt": i, **o} for i, o in zip(order, out)],
            }
            if args.engine == "stride":
                row["fingerprints_differing"] = sum(o["fingerprint"] != baseline[i]["fingerprint"] for i, o in zip(order, out))
                with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/stats", timeout=5) as r:
                    row["stats_after"] = json.load(r)
            report["levels"].append(row)
            print(f"{args.engine} c={level}: {len(differs)}/{count} requests differ, {row['prompts_changed']}/{len(texts)} prompts changed, stats {row.get('stats_after', {}).get('preemptions')}", flush=True)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=30)
        except Exception:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=1)
        f.write("\n")
    time.sleep(3)


if __name__ == "__main__":
    main()
