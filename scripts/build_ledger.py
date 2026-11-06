#!/usr/bin/env python3
"""Builds docs/resume/METRICS.jsonl from raw artifacts under reports/.

Every value is read or computed here from a committed artifact; nothing is typed in.
Missing artifacts produce no row and a warning, so a gap cannot pass as a result.
Row schema: id, value, unit, gate, method, command, commit, gpu, date, artifact_path.
"""
import argparse
import json
import os
import re
import statistics
import sys


class Ledger:
    def __init__(self, root, commit, gpu, date):
        self.root, self.commit, self.gpu, self.date = root, commit, gpu, date
        self.rows, self.missing = [], []

    def load(self, path):
        full = os.path.join(self.root, path)
        if not os.path.exists(full):
            self.missing.append(path)
            return None
        with open(full) as f:
            return json.load(f)

    def add(self, id_, value, unit, gate, method, command, artifact):
        self.rows.append({"id": id_, "value": value, "unit": unit, "gate": gate, "method": method, "command": command,
                          "commit": self.commit, "gpu": self.gpu, "date": self.date, "artifact_path": artifact})


def parse_env(root):
    path = os.path.join(root, "reports/env/environment.txt")
    if not os.path.exists(path):
        return "unknown", "unknown", "unknown", {}
    text = open(path).read()
    commit = re.search(r"^commit (\S+)", text, re.M)
    date = re.search(r"^(\w{3} \w{3} +\d+ [\d:]+ \w+ \d{4})", text, re.M)
    gpu = re.search(r"^(NVIDIA [^,]+), ([\d.]+), (\d+ MiB)", text, re.M)
    info = {"driver": gpu.group(2) if gpu else None, "vram": gpu.group(3) if gpu else None}
    for key, pat in (("vllm", r"vllm (\S+) torch (\S+)"), ("aiperf", r"aiperf[ ,v]*([\d.]+)"), ("cuda_driver", r"CUDA Version: ([\d.]+)"), ("nvcc", r"Cuda compiler driver.*?([\d.]+),"), ("cpu", r"model name\s*:\s*(.+)")):
        m = re.search(pat, text)
        info[key] = m.group(1) if m else None
    return commit.group(1) if commit else "unknown", gpu.group(1) if gpu else "unknown", date.group(1) if date else "unknown", info


def sweep_point(summary, concurrency):
    for p in summary["points"]:
        if p["concurrency"] == concurrency:
            return p
    return None


def mean_field(point, key, field):
    return statistics.mean(r[key][field] for r in point["repeats"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    args = parser.parse_args()
    root = os.path.abspath(args.root)
    commit, gpu, date, env = parse_env(root)
    L = Ledger(root, commit, gpu, date)

    # Environment and the pinned snapshot.
    export = os.path.join(root, "reports/g3/aiperf/stride_c1_r0/profile_export_aiperf.json")
    if os.path.exists(export):
        env["aiperf"] = json.load(open(export)).get("aiperf_version")
    for key, value in env.items():
        if value:
            L.add(f"env.{key}", value, "", "G0", "recorded at the start of run_all.sh", "scripts/run_all.sh", "reports/env/environment.txt")
    lock = L.load("weights.lock.json")
    if lock:
        L.add("weights.repo", lock["repo"], "", "G0", "pinned snapshot", "scripts/pin_weights.py", "weights.lock.json")
        L.add("weights.commit", lock["commit"], "", "G0", "pinned snapshot", "scripts/pin_weights.py", "weights.lock.json")
        for f in lock["files"]:
            L.add(f"weights.sha256.{f['path']}", f["sha256"], "sha256", "G0", f"computed locally over {f['bytes']} bytes", "scripts/verify_weights.py", "weights.lock.json")

    # G1.
    g1 = L.load("reports/g1/summary.json")
    if g1:
        cmd = "python3 tests/g1_real.py"
        art = "reports/g1/summary.json"
        for name, r in g1["greedy"].items():
            L.add(f"g1.greedy.{name}.pct", r["token_match_pct"], "%", "G1", f"{r['token_match']}/{r['token_total']} tokens, 100 prompts x 256", cmd, art)
            L.add(f"g1.greedy.{name}.identical_sequences", r["sequences_identical"], "of 100", "G1", "whole-sequence equality", cmd, art)
        for name, r in g1["logits"].items():
            L.add(f"g1.logits.{name}.max_abs_diff", r["max_abs_diff"], "logit units", "G1", f"{r['vectors']} logit vectors", cmd, art)
            L.add(f"g1.logits.{name}.vectors", r["vectors"], "vectors", "G1", "logit vectors compared", cmd, art)
            L.add(f"g1.logits.{name}.p99_vector_max_abs_diff", r["p99_vector_max_abs_diff"], "logit units", "G1", f"{r['vectors']} logit vectors", cmd, art)
        for path, n in g1["fingerprints_equal_c1_vs_c32"].items():
            L.add(f"g1.fingerprints_equal_c1_vs_c32.{path}", n, "of 100", "G1", "shuffled lanes, shuffled KV pages, prefill chunk 13", cmd, art)
    xd = L.load("reports/g1/cross_device.json")
    if xd:
        L.add("g1.cross_device.byte_identical", xd["byte_identical"], "bool", "G1", "SHA256 of exact-path logits across devices", "scripts/cross_device.py", "reports/g1/cross_device.json")
        L.add("g1.cross_device.logit_values", xd["logit_values_compared"], "FP32 values", "G1", "8 prompts x 32 steps x vocab", "scripts/cross_device.py", "reports/g1/cross_device.json")
        L.add("g1.cross_device.devices", len(xd["runs"]), "devices", "G1", ", ".join(r["host"] or r["device"] for r in xd["runs"].values()), "scripts/cross_device.py", "reports/g1/cross_device.json")

    # Same commit and weights on a second physical RTX 4090.
    for path in ("tc", "exact"):
        here, there = os.path.join(root, f"reports/g1/stride_{path}_c1.jsonl"), os.path.join(root, f"reports/g1/host2/host2_{path}.jsonl")
        if os.path.exists(here) and os.path.exists(there):
            a = {r["id"]: r for r in map(json.loads, open(here))}
            b = {r["id"]: r for r in map(json.loads, open(there))}
            equal = sum(a[i]["fingerprint"] == b[i]["fingerprint"] and a[i]["tokens"] == b[i]["tokens"] for i in a)
            L.add(f"g1.cross_host.{path}.identical", equal, f"of {len(a)} prompts", "G1", "fingerprints and tokens, second RTX 4090 in another data center; see reports/g1/host2/host2_env.txt", "stride_cli on both hosts", f"reports/g1/host2/host2_{path}.jsonl")

    # G2 and the nondeterminism study.
    for engine in ("stride", "vllm"):
        det = L.load(f"reports/g2/determinism_{engine}.json")
        if not det:
            continue
        art = f"reports/g2/determinism_{engine}.json"
        cmd = f"python3 scripts/bench/determinism.py --engine {engine}"
        L.add(f"g2.determinism.{engine}.sequential_rerun_differs", det["sequential_repeat_differs"], f"of {det['prompts']} prompts", "G2", "two sequential passes", cmd, art)
        L.add(f"g2.determinism.{engine}.total_requests_differing", sum(lv["requests_differing"] for lv in det["levels"]), "requests", "G2", "all concurrency levels", cmd, art)
        L.add(f"g2.determinism.{engine}.total_requests", sum(lv["requests"] for lv in det["levels"]), "requests", "G2", "all concurrency levels", cmd, art)
        for lv in det["levels"]:
            c = lv["concurrency"]
            L.add(f"g2.determinism.{engine}.c{c}.requests_differing_pct", lv["requests_differing_pct"], "%", "G2", f"{lv['requests_differing']}/{lv['requests']} requests differ from the sequential baseline", cmd, art)
            L.add(f"g2.determinism.{engine}.c{c}.requests_differing", lv["requests_differing"], "requests", "G2", "compared with the sequential baseline", cmd, art)
            L.add(f"g2.determinism.{engine}.c{c}.requests", lv["requests"], "requests", "G2", "requests sent at this concurrency", cmd, art)
            L.add(f"g2.determinism.{engine}.c{c}.prompts_changed_pct", lv["prompts_changed_pct"], "%", "G2", f"{lv['prompts_changed']}/{det['prompts']} prompts changed at least once", cmd, art)
            if "fingerprints_differing" in lv:
                L.add(f"g2.determinism.{engine}.c{c}.fingerprints_differing", lv["fingerprints_differing"], f"of {lv['requests']}", "G2", "system_fingerprint compared with the sequential baseline", cmd, art)

    # G3.
    roof = L.load("reports/g3/roofline.json")
    if roof:
        for path, r in roof["paths"].items():
            cmd = "python3 scripts/bench/roofline.py"
            art = "reports/g3/roofline.json"
            L.add(f"g3.batch1.{path}.tok_per_s", r["tok_per_s_mean"], "tok/s", "G3", f"mean of {r['repeats']} runs, std {r['tok_per_s_std']:.2f}", cmd, art)
            L.add(f"g3.batch1.{path}.tok_per_s_std", r["tok_per_s_std"], "tok/s", "G3", f"std over {r['repeats']} runs", cmd, art)
            L.add(f"g3.batch1.{path}.roofline_tok_per_s", r["roofline_tok_per_s"], "tok/s", "G3", f"{roof['peak_gb_per_s']} GB/s / {r['weight_bytes_per_pass']} weight bytes", cmd, art)
            L.add(f"g3.batch1.{path}.pct_of_roofline", r["pct_of_roofline"], "%", "G3", f"{roof['peak_gb_per_s']} GB/s / {r['weight_bytes_per_pass']} weight bytes = {r['roofline_tok_per_s']:.1f} tok/s", cmd, art)
    stride = L.load("reports/g3/aiperf/stride_summary.json")
    vllm = L.load("reports/g3/aiperf/vllm_summary.json")
    for name, s in (("stride", stride), ("vllm", vllm)):
        if not s:
            continue
        art = f"reports/g3/aiperf/{name}_summary.json"
        cmd = f"python3 scripts/bench/sweep.py --engine {name}"
        L.add(f"g3.aiperf.{name}.gpu_memory_peak_mib", s["gpu_memory_peak_mib"], "MiB", "G3", "nvidia-smi sampled at 2 Hz across the sweep", cmd, art)
        for p in s["points"]:
            c = p["concurrency"]
            n = len(p["repeats"])
            L.add(f"g3.aiperf.{name}.c{c}.output_tok_per_s", p["output_tok_per_s_mean"], "tok/s", "G3", f"AIPerf, mean of {n} repeats, std {p['output_tok_per_s_std']:.1f}", cmd, art)
            L.add(f"g3.aiperf.{name}.c{c}.output_tok_per_s_std", p["output_tok_per_s_std"], "tok/s", "G3", f"AIPerf, std over {n} repeats", cmd, art)
            for key, label in (("ttft_ms", "ttft"), ("itl_ms", "itl"), ("request_latency_ms", "request_latency")):
                L.add(f"g3.aiperf.{name}.c{c}.{label}_p50_ms", mean_field(p, key, "p50"), "ms", "G3", f"AIPerf, mean of {n} repeats", cmd, art)
                L.add(f"g3.aiperf.{name}.c{c}.{label}_p99_ms", mean_field(p, key, "p99"), "ms", "G3", f"AIPerf, mean of {n} repeats", cmd, art)
            L.add(f"g3.aiperf.{name}.c{c}.errors", sum(r["errors"] for r in p["repeats"]), "requests", "G2", f"AIPerf error count over {n} repeats", cmd, art)
    if stride:
        L.add("isl", stride["isl_mean"], "tokens", "G3", "AIPerf prompt length mean", "python3 scripts/bench/sweep.py", "reports/g3/aiperf/stride_summary.json")
        L.add("isl_stddev", stride["isl_stddev"], "tokens", "G3", "AIPerf prompt length std", "python3 scripts/bench/sweep.py", "reports/g3/aiperf/stride_summary.json")
        L.add("osl", stride["osl"], "tokens", "G3", "AIPerf output length", "python3 scripts/bench/sweep.py", "reports/g3/aiperf/stride_summary.json")
    if stride and vllm:
        for p in stride["points"]:
            v = sweep_point(vllm, p["concurrency"])
            if v:
                pct = 100.0 * p["output_tok_per_s_mean"] / v["output_tok_per_s_mean"]
                L.add(f"g3.aiperf.pct_of_vllm.c{p['concurrency']}", pct, "%", "G3", "stride mean / vLLM mean, same instance and settings", "python3 scripts/bench/sweep.py", "reports/g3/aiperf/stride_summary.json")
    trace = L.load("reports/g3/server_c64/trace_summary.json")
    if trace:
        L.add("g3.trace.c64.between_steps_us", trace["between_steps_us"], "us/step", "G3", "Nsight Systems: gap between steps under AIPerf load", "python3 scripts/bench/trace_server.py", "reports/g3/server_c64/trace_summary.json")
        L.add("g3.trace.c64.wall_us_per_step", trace["wall_us_per_step"], "us/step", "G3", "Nsight Systems under AIPerf load", "python3 scripts/bench/trace_server.py", "reports/g3/server_c64/trace_summary.json")
    memread = os.path.join(root, "reports/g3/memread.csv")
    if os.path.exists(memread):
        for line in open(memread):
            if line.startswith("lm_head") and ",A thread-per-row," in line:
                L.add("g3.memread.lm_head.thread_per_row_gb_s", float(line.strip().split(",")[-1]), "GB/s", "G3", "tools/bench_memread.cu, median of 12", "tools/bench_memread", "reports/g3/memread.csv")
            if line.startswith("lm_head") and ",B block-per-row interleaved,32," in line:
                L.add("g3.memread.lm_head.interleaved_gb_s", float(line.strip().split(",")[-1]), "GB/s", "G3", "tools/bench_memread.cu, median of 12", "tools/bench_memread", "reports/g3/memread.csv")
    probe = os.path.join(root, "reports/g3/tc_probe.txt")
    if os.path.exists(probe):
        m = re.search(r"Q1 row/lane permutation: (\d+) / (\d+)", open(probe).read())
        if m:
            L.add("g3.tc_probe.position_invariant_elements", int(m.group(1)), f"of {m.group(2)}", "G3", "tools/tc_probe.cu: mma results under row and lane permutation", "tools/tc_probe", "reports/g3/tc_probe.txt")

    # G4.
    off = L.load("reports/g4/micro_1x1_summary.json")
    on = L.load("reports/g4/micro_2x2_summary.json")
    tile = L.load("reports/g4/norm_tile_summary.json")
    if off and on:
        for p in on["points"]:
            c = p["concurrency"]
            base = sweep_point(off, c)
            cmd = "python3 scripts/bench/sweep.py --label micro_1x1 / micro_2x2"
            art = "reports/g4/micro_2x2_summary.json"
            L.add(f"g4.micro.c{c}.throughput_gain_pct", 100.0 * (p["output_tok_per_s_mean"] / base["output_tok_per_s_mean"] - 1.0), "%", "G4", f"2x2 vs 1x1, mean of {len(p['repeats'])} repeats: {p['output_tok_per_s_mean']:.0f} vs {base['output_tok_per_s_mean']:.0f} tok/s", cmd, art)
            a, b = mean_field(p, "itl_ms", "p50"), mean_field(base, "itl_ms", "p50")
            L.add(f"g4.micro.c{c}.itl_p50_cut_pct", 100.0 * (1.0 - a / b), "%", "G4", f"{b:.2f} ms to {a:.2f} ms", cmd, art)
            a, b = mean_field(p, "itl_ms", "p99"), mean_field(base, "itl_ms", "p99")
            L.add(f"g4.micro.c{c}.itl_p99_cut_pct", 100.0 * (1.0 - a / b), "%", "G4", f"{b:.2f} ms to {a:.2f} ms", cmd, art)
            a, b = mean_field(p, "ttft_ms", "p50"), mean_field(base, "ttft_ms", "p50")
            L.add(f"g4.micro.c{c}.ttft_p50_cut_pct", 100.0 * (1.0 - a / b), "%", "G4", f"{b:.1f} ms to {a:.1f} ms", cmd, art)
    if on and tile:
        for p in on["points"]:
            c = p["concurrency"]
            base = sweep_point(tile, c)
            L.add(f"g4.norm_lane.c{c}.throughput_gain_pct", 100.0 * (p["output_tok_per_s_mean"] / base["output_tok_per_s_mean"] - 1.0), "%", "G4", f"per-lane vs per-tile norm: {p['output_tok_per_s_mean']:.0f} vs {base['output_tok_per_s_mean']:.0f} tok/s", "python3 scripts/bench/sweep.py --label norm_tile", "reports/g4/norm_tile_summary.json")
    if os.path.exists(os.path.join(root, "reports/g4/bits_auto.jsonl")):
        a = open(os.path.join(root, "reports/g4/bits_auto.jsonl")).read()
        for name in ("bits_1x1", "bits_normtile"):
            path = os.path.join(root, f"reports/g4/{name}.jsonl")
            if os.path.exists(path):
                L.add(f"g4.regression.{name}.identical", a == open(path).read(), "bool", "G4", "outputs and fingerprints compared with the optimized build, 64 prompts", "scripts/run_all.sh ab_bits", f"reports/g4/{name}.jsonl")

    # G5.
    for name, label in (("mutation_cpu_tiny", "cpu_tiny"), ("mutation_real_exact", "real_exact"), ("mutation_real_tc", "real_tc")):
        m = L.load(f"reports/g5/{name}.json")
        if not m:
            continue
        art = f"reports/g5/{name}.json"
        cmd = "python3 tests/mutations.py"
        L.add(f"g5.{label}.caught", m["caught"], f"of {m['mutants']}", "G5", f"missed: {m['missed']}", cmd, art)
        L.add(f"g5.{label}.mutants", m["mutants"], "mutants", "G5", "injected bugs", cmd, art)
        L.add(f"g5.{label}.catch_rate_pct", m["catch_rate_pct"], "%", "G5", "any oracle, crash or hang", cmd, art)
        L.add(f"g5.{label}.caught_by_tolerance", m["caught_by_oracle"].get("tolerance", 0), f"of {m['mutants']}", "G5", f"tolerance {m['tolerance']} and token match", cmd, art)
        L.add(f"g5.{label}.caught_only_by_bit_exact", len(m["caught_only_by_bit_exact_oracles"]), f"of {m['mutants']}", "G5", "invariance or golden oracle only", cmd, art)
        L.add(f"g5.{label}.clean_preemptions", m["clean"]["preemptions"], "preemptions", "G5", "replay path exercised in the clean run", cmd, art)
        L.add(f"g5.{label}.clean_max_logit_diff", m["clean"]["max_logit_diff"], "logit units", "G5", "clean build against the reference", cmd, art)

    # Source counts.
    fused = ["Qkv", "GemmResidual", "Mlp13", "LmGreedy", "TcQkv", "TcGemmResidual", "TcMlp13", "TcLmGreedy", "AttentionPage"]
    source = open(os.path.join(root, "src/kernels.cu")).read() + open(os.path.join(root, "src/kernels_tc.cu")).read() + open(os.path.join(root, "src/kernel_common.h")).read()
    present = [k for k in fused if re.search(rf"struct {k}\b", source)]
    L.add("g3.fused_kernels", len(present), "kernels", "G3", ", ".join(present) + "; each merges at least two model operations, see docs/ARCHITECTURE.md", "grep in scripts/build_ledger.py", "docs/ARCHITECTURE.md")

    out = os.path.join(root, "docs/resume/METRICS.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        for row in L.rows:
            f.write(json.dumps(row) + "\n")
    print(f"{len(L.rows)} rows; missing artifacts: {L.missing}", file=sys.stderr)


if __name__ == "__main__":
    main()
