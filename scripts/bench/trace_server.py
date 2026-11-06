#!/usr/bin/env python3
"""Captures an Nsight Systems trace of stride_server under AIPerf load and reports
where each step's wall time goes: per-kernel GPU time, gaps between kernels, and
host time between steps.
"""
import argparse
import json
import os
import signal
import sqlite3
import subprocess
import time
import urllib.request


def analyze(sqlite_path):
    db = sqlite3.connect(sqlite_path)
    names = dict(db.execute("select id,value from StringIds").fetchall())
    rows = db.execute("select start,end,demangledName from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
    short = lambda n: names[n].split("kimpl::")[-1].split(">")[0].split("<")[0].split(",")[0].strip('"')
    steps, cur = [], []
    for s, e, n in rows:
        k = short(n)
        if k in ("TcEmbed", "Embed") and cur:
            steps.append(cur)
            cur = []
        cur.append((s, e, k))
    if cur:
        steps.append(cur)
    # The trace also covers startup and shutdown; keep steps with real work between them.
    per_kernel, busy, spans, host = {}, 0, 0, 0
    used = steps[len(steps) // 10 : -len(steps) // 10 or None]
    prev_end = None
    for st in used:
        for s, e, k in st:
            per_kernel[k] = per_kernel.get(k, 0) + (e - s)
            busy += e - s
        spans += st[-1][1] - st[0][0]
        if prev_end is not None:
            host += st[0][0] - prev_end
        prev_end = st[-1][1]
    n = max(len(used), 1)
    return {
        "steps": n,
        "gpu_busy_us_per_step": busy / n / 1e3,
        "step_span_us": spans / n / 1e3,
        "between_steps_us": host / n / 1e3,
        "wall_us_per_step": (spans + host) / n / 1e3,
        "kernel_us_per_step": {k: round(v / n / 1e3, 1) for k, v in sorted(per_kernel.items(), key=lambda x: -x[1])},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--requests", type=int, default=128)
    parser.add_argument("--isl", type=int, default=256)
    parser.add_argument("--osl", type=int, default=256)
    parser.add_argument("--port", type=int, default=8124)
    parser.add_argument("--server-flag", action="append", default=[])
    parser.add_argument("--stride-bin", default="/root/stride/build/stride_server")
    parser.add_argument("--aiperf-bin", default="/root/venv-vllm/bin/aiperf")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rep = os.path.join(args.out, "server_trace")
    env = dict(os.environ, PATH="/usr/local/cuda/bin:" + os.environ["PATH"])
    server = [args.stride_bin, "--model", args.model, "--host", "127.0.0.1", "--port", str(args.port), "--path", "tc", "--kv-gb", "16",
              "--max-seqs", "256", "--max-rows", "2048", "--prefill-chunk", "256", *args.server_flag]
    proc = subprocess.Popen(["nsys", "profile", "--trace=cuda", "--cuda-memory-usage=false", "--sample=none", "--cpuctxsw=none", "-o", rep, "--force-overwrite=true", *server],
                            stdout=open(rep + ".log", "w"), stderr=subprocess.STDOUT, env=env, preexec_fn=os.setsid)
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=2)
            break
        except Exception:
            time.sleep(1)
    cmd = [args.aiperf_bin, "profile", "--model", "llama-3.2-1b", "--url", f"http://127.0.0.1:{args.port}", "--endpoint-type", "chat", "--streaming",
           "--tokenizer", args.model, "--concurrency", str(args.concurrency), "--request-count", str(args.requests),
           "--prompt-input-tokens-mean", str(args.isl), "--prompt-input-tokens-stddev", "32", "--prompt-output-tokens-mean", str(args.osl), "--prompt-output-tokens-stddev", "0",
           "--extra-inputs", "ignore_eos:true", "--extra-inputs", "temperature:0.0", "--random-seed", "77", "--artifact-dir", os.path.join(args.out, "aiperf"),
           "--ui", "none", "--no-server-metrics", "--no-auto-plot"]
    subprocess.run(cmd, check=True, stdout=open(rep + "_aiperf.log", "w"), stderr=subprocess.STDOUT)
    stats = json.load(urllib.request.urlopen(f"http://127.0.0.1:{args.port}/stats", timeout=5))
    # Stop the server only, so nsys writes its report.
    subprocess.run(["pkill", "-TERM", "-x", "stride_server"], check=False)
    proc.wait(timeout=240)
    subprocess.run(["nsys", "stats", "--force-export=true", "--report", "cuda_gpu_kern_sum", "--format", "csv", "--output", os.path.join(args.out, "kernels"), rep + ".nsys-rep"],
                   env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    result = analyze(rep + ".sqlite")
    result["server_stats"] = stats
    summary = json.load(open(os.path.join(args.out, "aiperf", "profile_export_aiperf.json")))
    result["aiperf_output_tok_per_s"] = summary["output_token_throughput"]["avg"]
    result["aiperf_itl_p50_ms"] = summary["inter_token_latency"]["p50"]
    result["aiperf_ttft_p50_ms"] = summary["time_to_first_token"]["p50"]
    with open(os.path.join(args.out, "trace_summary.json"), "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "server_stats"}, indent=1))
    print(json.dumps(stats)[:600])


if __name__ == "__main__":
    main()
