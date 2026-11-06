#!/usr/bin/env python3
"""AIPerf concurrency sweep against stride or vLLM on the OpenAI chat endpoint.

Fairness rules, enforced here rather than by convention:
  * both engines load the same verified model directory, with network fetches off;
  * prefix caching is off in vLLM, and stride has none, so a repeated prompt cannot
    benefit either side;
  * warmup is a separate AIPerf run with a different seed, so warmup prompts are
    disjoint from measured prompts (AIPerf issue #1656) and warmup is excluded;
  * the same prompt and output length distributions and the same seeds are used;
  * every point is repeated, and all raw AIPerf exports are kept.
"""
import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

METRICS = {
    "output_token_throughput": "output_tok_per_s",
    "time_to_first_token": "ttft_ms",
    "inter_token_latency": "itl_ms",
    "request_latency": "request_latency_ms",
}


def wait_ready(port, proc, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise SystemExit(f"server exited with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as r:
                if r.status == 200:
                    return time.time() - t0
        except Exception:
            time.sleep(1)
    raise SystemExit("server did not become ready")


def start_engine(args, log):
    if args.engine == "stride":
        cmd = [args.stride_bin, "--model", args.model, "--host", "127.0.0.1", "--port", str(args.port), "--path", args.stride_path,
               "--kv-gb", str(args.kv_gb), "--max-seqs", "256", "--max-rows", str(args.max_rows), "--prefill-chunk", str(args.prefill_chunk),
               *args.stride_flag]
    else:
        cmd = [args.vllm_bin, "serve", args.model, "--dtype", "bfloat16", "--max-model-len", "2048", "--no-enable-prefix-caching", "--seed", "0",
               "--gpu-memory-utilization", "0.85", "--served-model-name", args.served_name, "--host", "127.0.0.1", "--port", str(args.port),
               "--disable-log-stats", "--uvicorn-log-level", "warning"]
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, preexec_fn=os.setsid)
    return cmd, proc


def run_aiperf(args, concurrency, requests, seed, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cmd = [args.aiperf_bin, "profile", "--model", args.served_name, "--url", f"http://127.0.0.1:{args.port}", "--endpoint-type", "chat", "--streaming",
           "--tokenizer", args.model, "--concurrency", str(concurrency), "--request-count", str(requests),
           "--prompt-input-tokens-mean", str(args.isl), "--prompt-input-tokens-stddev", str(args.isl_stddev),
           "--prompt-output-tokens-mean", str(args.osl), "--prompt-output-tokens-stddev", "0",
           "--extra-inputs", "ignore_eos:true", "--extra-inputs", "temperature:0.0", "--random-seed", str(seed),
           "--artifact-dir", out_dir, "--ui", "none", "--no-server-metrics", "--no-auto-plot"]
    with open(os.path.join(out_dir, "aiperf.log"), "w") as log:
        # AIPerf 0.13 mishandles a local tokenizer path under HF_HUB_OFFLINE. The path
        # below is a verified local directory, so nothing is fetched without the flag.
        env = {k: v for k, v in os.environ.items() if k not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
    with open(os.path.join(out_dir, "command.json"), "w") as f:
        json.dump(cmd, f)
    return json.load(open(os.path.join(out_dir, "profile_export_aiperf.json")))


def pick(summary, name, field):
    node = summary.get(name) or summary.get("records", {}).get(name)
    return None if node is None else node.get(field)


class GpuMemory(threading.Thread):
    """Samples GPU memory so the report can state a measured peak."""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak = 0
        self.stop_flag = False

    def run(self):
        while not self.stop_flag:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
            if out:
                self.peak = max(self.peak, int(out.splitlines()[0]))
            time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, choices=["stride", "vllm"])
    parser.add_argument("--label", default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--concurrency", default="1,4,16,64,128")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--isl", type=int, default=256)
    parser.add_argument("--isl-stddev", type=int, default=32)
    parser.add_argument("--osl", type=int, default=256)
    parser.add_argument("--min-requests", type=int, default=64)
    parser.add_argument("--requests-per-concurrency", type=int, default=6)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--served-name", default="llama-3.2-1b")
    parser.add_argument("--stride-bin", default="/root/stride/build/stride_server")
    parser.add_argument("--stride-path", default="tc")
    parser.add_argument("--stride-flag", action="append", default=[])
    parser.add_argument("--kv-gb", type=float, default=16.0)
    parser.add_argument("--max-rows", type=int, default=2048)
    parser.add_argument("--prefill-chunk", type=int, default=256)
    parser.add_argument("--vllm-bin", default="/root/venv-vllm/bin/vllm")
    parser.add_argument("--aiperf-bin", default="/root/venv-vllm/bin/aiperf")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    pin = subprocess.check_output([sys.executable, os.path.join(here, "..", "verify_weights.py"), "--dir", args.model], text=True).strip()
    os.makedirs(args.out, exist_ok=True)
    label = args.label or args.engine

    with open(os.path.join(args.out, f"{label}_server.log"), "w") as log:
        cmd, proc = start_engine(args, log)
        memory = GpuMemory()
        memory.start()
        try:
            ready = wait_ready(args.port, proc)
            points = []
            for concurrency in [int(c) for c in args.concurrency.split(",")]:
                requests = max(args.min_requests, concurrency * args.requests_per_concurrency)
                # Warmup prompts come from a different seed, so they never repeat in the measurement.
                run_aiperf(args, concurrency, max(concurrency * 2, 8), 900000 + concurrency, os.path.join(args.out, f"{label}_c{concurrency}_warmup"))
                reps = []
                for r in range(args.repeats):
                    summary = run_aiperf(args, concurrency, requests, 1000 + r, os.path.join(args.out, f"{label}_c{concurrency}_r{r}"))
                    row = {"repeat": r}
                    for name, key in METRICS.items():
                        row[key] = {"avg": pick(summary, name, "avg"), "p50": pick(summary, name, "p50"), "p99": pick(summary, name, "p99")}
                    row["errors"] = pick(summary, "error_request_count", "avg") or 0
                    row["requests_ok"] = pick(summary, "request_count", "avg")
                    reps.append(row)
                    print(f"{label} c={concurrency} r={r}: {row['output_tok_per_s']['avg']} tok/s ttft_p50={row['ttft_ms']['p50']} itl_p50={row['itl_ms']['p50']} errors={row['errors']}", flush=True)
                tput = [x["output_tok_per_s"]["avg"] for x in reps]
                points.append({
                    "concurrency": concurrency,
                    "requests_per_repeat": requests,
                    "output_tok_per_s_mean": statistics.mean(tput),
                    "output_tok_per_s_std": statistics.stdev(tput) if len(tput) > 1 else 0.0,
                    "repeats": reps,
                })
        finally:
            memory.stop_flag = True
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=30)
            except Exception:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        time.sleep(3)

    result = {
        "engine": args.engine,
        "label": label,
        "server_command": cmd,
        "pinned_weights": pin,
        "isl_mean": args.isl,
        "isl_stddev": args.isl_stddev,
        "osl": args.osl,
        "temperature": 0.0,
        "ignore_eos": True,
        "prefix_caching": False,
        "server_ready_seconds": ready,
        "gpu_memory_peak_mib": memory.peak,
        "points": points,
    }
    with open(os.path.join(args.out, f"{label}_summary.json"), "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
