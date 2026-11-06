#!/usr/bin/env python3
"""Builds docs/resume/METRICS.jsonl from raw artifacts under reports/.

Every value is read or computed here from a committed artifact; nothing is typed in.
Each row's commit, device and date come from the record of the stage that produced its
artifact (reports/env/stages.jsonl), so a stale artifact shows its real provenance. A
missing artifact produces no row and a warning, so a gap cannot pass as a result.
Row schema: id, value, unit, gate, method, command, commit, gpu, date, artifact_path.
"""
import argparse
import glob
import json
import os
import re
import statistics
import sys


class Ledger:
    def __init__(self, root):
        self.root = root
        self.rows, self.missing = [], []
        self.stages = {}
        path = os.path.join(root, "reports/env/stages.jsonl")
        if os.path.exists(path):
            for line in open(path):
                record = json.loads(line)
                self.stages[record["stage"]] = record  # the last record wins

    def load(self, path):
        full = os.path.join(self.root, path)
        if not os.path.exists(full):
            self.missing.append(path)
            return None
        with open(full) as f:
            return json.load(f)

    def add(self, stage, id_, value, unit, gate, method, command, artifact):
        record = self.stages.get(stage)
        if record is None:
            self.missing.append(f"provenance for stage {stage}")
            record = {"commit": "unknown", "device": "unknown", "finished_utc": "unknown"}
        self.rows.append({"id": id_, "value": value, "unit": unit, "gate": gate, "method": method, "command": command,
                          "commit": record["commit"], "gpu": record["device"], "date": record["finished_utc"], "artifact_path": artifact})


def parse_env(root):
    files = sorted(glob.glob(os.path.join(root, "reports/env/environment_*.txt")), key=os.path.getmtime)
    if not files:
        return {}, None
    path = files[-1]
    text = open(path).read()
    info = {}
    gpu = re.search(r"^(NVIDIA [^,]+), ([\d.]+), (\d+ MiB)", text, re.M)
    if gpu:
        info.update(gpu=gpu.group(1), driver=gpu.group(2), vram=gpu.group(3))
    for key, pat in (("vllm", r"vllm (\S+) torch"), ("vllm_torch", r"vllm \S+ torch (\S+)"), ("cuda_driver", r"CUDA Version: ([\d.]+)"),
                     ("toolkit", r"release [\d.]+, V([\d.]+)"), ("cpu", r"model name\s*:\s*(.+)"), ("provider", r"^provider (.+)$"), ("gpu_uuid", r"(GPU-[0-9a-f-]+)"), ("reference_torch", r"reference torch (\S+)")):
        m = re.search(pat, text, re.M)
        info[key] = m.group(1) if m else None
    return info, os.path.relpath(path, root)


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
    L = Ledger(root)

    # Environment and the pinned snapshot.
    env, env_path = parse_env(root)
    for key, value in env.items():
        if value:
            L.add("build", f"env.{key}", value, "", "G0", "recorded by run_all.sh", "scripts/run_all.sh", env_path)
    export = os.path.join(root, "reports/g3/aiperf/stride_c1_r0/profile_export_aiperf.json")
    if os.path.exists(export):
        L.add("sweep_stride", "env.aiperf", json.load(open(export)).get("aiperf_version"), "", "G0", "from the AIPerf export", "scripts/bench/sweep.py", "reports/g3/aiperf/stride_c1_r0/profile_export_aiperf.json")
    lock = L.load("weights.lock.json")
    if lock:
        L.add("pin", "weights.repo", lock["repo"], "", "G0", "pinned snapshot", "scripts/pin_weights.py", "weights.lock.json")
        L.add("pin", "weights.commit", lock["commit"], "", "G0", "pinned snapshot", "scripts/pin_weights.py", "weights.lock.json")
        for f in lock["files"]:
            L.add("pin", f"weights.sha256.{f['path']}", f["sha256"], "sha256", "G0", f"computed locally over {f['bytes']} bytes", "scripts/verify_weights.py", "weights.lock.json")
            L.add("pin", f"weights.bytes.{f['path']}", f["bytes"], "bytes", "G0", "file size", "scripts/verify_weights.py", "weights.lock.json")

    # G1.
    g1 = L.load("reports/g1/summary.json")
    if g1:
        cmd = "python3 tests/g1_real.py"
        art = "reports/g1/summary.json"
        for name, r in g1["greedy"].items():
            L.add("g1", f"g1.greedy.{name}.pct", r["token_match_pct"], "%", "G1", f"{r['token_match']}/{r['token_total']} tokens, 100 prompts x 256", cmd, art)
            L.add("g1", f"g1.greedy.{name}.identical_sequences", r["sequences_identical"], "of 100", "G1", "whole-sequence equality", cmd, art)
        for name, r in g1["logits"].items():
            L.add("g1", f"g1.logits.{name}.max_abs_diff", r["max_abs_diff"], "logit units", "G1", f"{r['vectors']} logit vectors", cmd, art)
            L.add("g1", f"g1.logits.{name}.vectors", r["vectors"], "vectors", "G1", "logit vectors compared", cmd, art)
            L.add("g1", f"g1.logits.{name}.p99_vector_max_abs_diff", r["p99_vector_max_abs_diff"], "logit units", "G1", f"{r['vectors']} logit vectors", cmd, art)
        for path, n in g1["fingerprints_equal_c1_vs_c32"].items():
            L.add("g1", f"g1.fingerprints_equal_c1_vs_c32.{path}", n, "of 100", "G1", "shuffled lanes, shuffled KV pages, prefill chunk 13", cmd, art)
    prompts_path = os.path.join(root, "reports/g1/prompts.jsonl")
    if os.path.exists(prompts_path):
        lengths = [len(json.loads(l)["tokens"]) for l in open(prompts_path)]
        for name, value in (("min", min(lengths)), ("max", max(lengths)), ("count", len(lengths))):
            L.add("g1", f"g1.prompt_tokens.{name}", value, "tokens", "G1", "token-id prompts used for G1 and G2", cmd if g1 else "", "reports/g1/prompts.jsonl")
    xd = L.load("reports/g1/cross_device.json")
    if xd:
        cmd = "scripts/run_all.sh, then scripts/cross_device.py with an ARM leg"
        L.add("cross_device_arm", "g1.cross_device.byte_identical", xd["byte_identical"], "bool", "G1", "SHA-256 of exact-path logits across devices", cmd, "reports/g1/cross_device.json")
        L.add("cross_device_arm", "g1.cross_device.logit_values", xd["logit_values_compared"], "FP32 values", "G1", "8 prompts x 32 steps x vocab", cmd, "reports/g1/cross_device.json")
        L.add("cross_device_arm", "g1.cross_device.devices", len(xd["runs"]), "devices", "G1", ", ".join(r["host"] or r["device"] for r in xd["runs"].values()), cmd, "reports/g1/cross_device.json")

    # The same commit and weights on a second physical RTX 4090.
    for path in ("tc", "exact"):
        here, there = os.path.join(root, f"reports/g1/stride_{path}_c1.jsonl"), os.path.join(root, f"reports/g1/host2/host2_{path}.jsonl")
        if os.path.exists(here) and os.path.exists(there):
            a = {r["id"]: r for r in map(json.loads, open(here))}
            b = {r["id"]: r for r in map(json.loads, open(there))}
            equal = sum(a[i]["fingerprint"] == b[i]["fingerprint"] and a[i]["tokens"] == b[i]["tokens"] for i in a)
            L.add("cross_host", f"g1.cross_host.{path}.identical", equal, f"of {len(a)} prompts", "G1", "fingerprints and tokens; host details and timings in reports/g1/host2", "stride_cli on both hosts", f"reports/g1/host2/host2_{path}.jsonl")

    # G2 and the load study. Counts are recomputed from the per-request hashes.
    for label, stage in (("stride", "det_stride"), ("vllm", "det_vllm"), ("stride_preempt", "det_preempt")):
        det = L.load(f"reports/g2/determinism_{label}.json")
        if not det:
            continue
        art = f"reports/g2/determinism_{label}.json"
        cmd = "python3 scripts/bench/determinism.py"
        base = det["baseline"]
        L.add(stage, f"g2.determinism.{label}.sequential_rerun_differs", det["sequential_repeat_differs"], f"of {det['prompts']} prompts", "G2", "two sequential passes", cmd, art)
        total_differs, total = 0, 0
        for lv in det["levels"]:
            c = lv["concurrency"]
            differs = [r["prompt"] for r in lv["responses"] if r["sha"] != base[r["prompt"]]["sha"]]
            total_differs += len(differs)
            total += len(lv["responses"])
            pct = 100.0 * len(differs) / len(lv["responses"])
            L.add(stage, f"g2.determinism.{label}.c{c}.requests_differing_pct", pct, "%", "G2", f"{len(differs)}/{len(lv['responses'])} requests differ from the sequential baseline", cmd, art)
            L.add(stage, f"g2.determinism.{label}.c{c}.requests_differing", len(differs), "requests", "G2", "text hash compared with the sequential baseline", cmd, art)
            L.add(stage, f"g2.determinism.{label}.c{c}.requests", len(lv["responses"]), "requests", "G2", "requests sent at this concurrency", cmd, art)
            L.add(stage, f"g2.determinism.{label}.c{c}.prompts_changed", len(set(differs)), f"of {det['prompts']}", "G2", "prompts that changed at least once", cmd, art)
            if "fingerprints_differing" in lv:
                L.add(stage, f"g2.determinism.{label}.c{c}.fingerprints_differing", lv["fingerprints_differing"], f"of {lv['requests']}", "G2", "system_fingerprint compared with the sequential baseline", cmd, art)
            if "stats_after" in lv:
                L.add(stage, f"g2.determinism.{label}.c{c}.preemptions_so_far", lv["stats_after"].get("preemptions"), "preemptions", "G2", "server /stats after this level", cmd, art)
        L.add(stage, f"g2.determinism.{label}.total_requests_differing", total_differs, "requests", "G2", "all concurrency levels", cmd, art)
        L.add(stage, f"g2.determinism.{label}.total_requests", total, "requests", "G2", "all concurrency levels", cmd, art)
        if total:
            L.add(stage, f"g2.determinism.{label}.total_differing_pct", 100.0 * total_differs / total, "%", "G2", "all concurrency levels", cmd, art)

    # G3.
    roof = L.load("reports/g3/roofline.json")
    if roof:
        cmd = "python3 scripts/bench/roofline.py"
        art = "reports/g3/roofline.json"
        L.add("roofline", "g3.roofline.tok_per_s", roof["roofline_tok_per_s"], "tok/s", "G3", f"{roof['peak_gb_per_s']} GB/s specified peak / {roof['weight_bytes_per_pass']} weight bytes", cmd, art)
        L.add("roofline", "g3.roofline.peak_gb_per_s", roof["peak_gb_per_s"], "GB/s", "G3", "RTX 4090 specified peak, not measured", cmd, art)
        L.add("roofline", "g3.roofline.weight_bytes_per_pass", roof["weight_bytes_per_pass"], "bytes", "G3", "from the safetensors header", cmd, art)
        for path, r in roof["paths"].items():
            L.add("roofline", f"g3.batch1.{path}.tok_per_s", r["tok_per_s_mean"], "tok/s", "G3", f"mean of {r['repeats']} runs", cmd, art)
            L.add("roofline", f"g3.batch1.{path}.tok_per_s_std", r["tok_per_s_std"], "tok/s", "G3", f"std over {r['repeats']} runs", cmd, art)
            L.add("roofline", f"g3.batch1.{path}.pct_of_roofline", r["pct_of_roofline"], "%", "G3", "tok/s over the roofline; KV and activation traffic not counted", cmd, art)
    stride = L.load("reports/g3/aiperf/stride_summary.json")
    vllm = L.load("reports/g3/aiperf/vllm_summary.json")
    for name, s in (("stride", stride), ("vllm", vllm)):
        if not s:
            continue
        stage = f"sweep_{name}"
        art = f"reports/g3/aiperf/{name}_summary.json"
        cmd = f"python3 scripts/bench/sweep.py --engine {name}"
        L.add(stage, f"g3.aiperf.{name}.gpu_memory_peak_mib", s["gpu_memory_peak_mib"], "MiB", "G3", "nvidia-smi sampled at 2 Hz across the sweep", cmd, art)
        for p in s["points"]:
            c = p["concurrency"]
            n = len(p["repeats"])
            L.add(stage, f"g3.aiperf.{name}.c{c}.output_tok_per_s", p["output_tok_per_s_mean"], "tok/s", "G3", f"AIPerf, mean of {n} repeats", cmd, art)
            L.add(stage, f"g3.aiperf.{name}.c{c}.output_tok_per_s_std", p["output_tok_per_s_std"], "tok/s", "G3", f"AIPerf, std over {n} repeats", cmd, art)
            for key, label in (("ttft_ms", "ttft"), ("itl_ms", "itl"), ("request_latency_ms", "request_latency")):
                L.add(stage, f"g3.aiperf.{name}.c{c}.{label}_p50_ms", mean_field(p, key, "p50"), "ms", "G3", f"AIPerf, mean of {n} repeats", cmd, art)
                L.add(stage, f"g3.aiperf.{name}.c{c}.{label}_p99_ms", mean_field(p, key, "p99"), "ms", "G3", f"AIPerf, mean of {n} repeats", cmd, art)
            L.add(stage, f"g3.aiperf.{name}.c{c}.errors", sum(r["errors"] for r in p["repeats"]), "requests", "G3", f"AIPerf error count over {n} repeats", cmd, art)
            exports = [json.load(open(x)) for x in glob.glob(os.path.join(root, f"reports/g3/aiperf/{name}_c{c}_r*/profile_export_aiperf.json"))]
            usage = [e["usage_prompt_tokens"]["avg"] for e in exports if e.get("usage_prompt_tokens")]
            if usage:
                L.add(stage, f"g3.aiperf.{name}.c{c}.prompt_tokens_on_wire", statistics.mean(usage), "tokens", "G3", "server-reported prompt tokens, including the chat template", cmd, art)
        L.add(stage, f"g3.aiperf.{name}.isl_requested", s["isl_mean"], "tokens", "G3", "AIPerf prompt length mean, before the chat template", cmd, art)
        L.add(stage, f"g3.aiperf.{name}.isl_stddev", s["isl_stddev"], "tokens", "G3", "AIPerf prompt length std", cmd, art)
        L.add(stage, f"g3.aiperf.{name}.osl", s["osl"], "tokens", "G3", "AIPerf output length", cmd, art)
    if stride and vllm:
        for p in stride["points"]:
            v = sweep_point(vllm, p["concurrency"])
            if v:
                pct = 100.0 * p["output_tok_per_s_mean"] / v["output_tok_per_s_mean"]
                L.add("sweep_vllm", f"g3.aiperf.pct_of_vllm.c{p['concurrency']}", pct, "%", "G3", "HTTP serving; stride mean / vLLM mean on one instance, vLLM on its default scheduler settings", "python3 scripts/bench/sweep.py", "reports/g3/aiperf/stride_summary.json")
    offline = L.load("reports/g3/offline.json")
    if offline:
        cmd = "python3 scripts/bench/offline.py"
        L.add("offline", "g3.offline.prompt_tokens", offline["prompt_tokens"], "tokens", "G3", "token-id prompts", cmd, "reports/g3/offline.json")
        L.add("offline", "g3.offline.repeats", offline["repeats"], "repeats", "G3", "per batch size", cmd, "reports/g3/offline.json")
        for p in offline["points"]:
            b = p["batch"]
            L.add("offline", f"g3.offline.b{b}.stride_tok_per_s", p["stride_tok_per_s"], "tok/s", "G3", f"no HTTP; mean of {offline['repeats']}", cmd, "reports/g3/offline.json")
            L.add("offline", f"g3.offline.b{b}.vllm_tok_per_s", p["vllm_tok_per_s"], "tok/s", "G3", f"no HTTP; mean of {offline['repeats']}", cmd, "reports/g3/offline.json")
            L.add("offline", f"g3.offline.b{b}.pct_of_vllm", p["stride_pct_of_vllm"], "%", "G3", "no HTTP", cmd, "reports/g3/offline.json")
    trace = L.load("reports/g3/server_c64/trace_summary.json")
    if trace:
        cmd = "python3 scripts/bench/trace_server.py"
        art = "reports/g3/server_c64/trace_summary.json"
        L.add("nsys_server", "g3.trace.c64.between_steps_us", trace["between_steps_us"], "us/step", "G3", "Nsight Systems: GPU idle between steps under AIPerf load", cmd, art)
        L.add("nsys_server", "g3.trace.c64.wall_us_per_step", trace["wall_us_per_step"], "us/step", "G3", "Nsight Systems under AIPerf load", cmd, art)
    memread = os.path.join(root, "reports/g3/memread.csv")
    if os.path.exists(memread):
        for line in open(memread):
            parts = line.strip().split(",")
            if line.startswith("lm_head") and parts[1] == "A thread-per-row":
                L.add("memread", "g3.memread.lm_head.thread_per_row_gb_s", float(parts[-1]), "GB/s", "G3", "median of 12 reads", "build/bench_memread", "reports/g3/memread.csv")
            if line.startswith("lm_head") and parts[1] == "B block-per-row interleaved" and parts[2] == "32":
                L.add("memread", "g3.memread.lm_head.interleaved_gb_s", float(parts[-1]), "GB/s", "G3", "median of 12 reads", "build/bench_memread", "reports/g3/memread.csv")
    probe = os.path.join(root, "reports/g3/tc_probe.txt")
    if os.path.exists(probe):
        m = re.search(r"Q1 row/lane permutation: (\d+) / (\d+)", open(probe).read())
        if m:
            L.add("tc_probe", "g3.tc_probe.position_invariant_elements", int(m.group(1)), f"of {m.group(2)}", "G3", "mma results under row and lane permutation", "build/tc_probe", "reports/g3/tc_probe.txt")

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
            L.add("ab_micro_on", f"g4.micro.c{c}.throughput_gain_pct", 100.0 * (p["output_tok_per_s_mean"] / base["output_tok_per_s_mean"] - 1.0), "%", "G4", f"2x2 vs 1x1, mean of {len(p['repeats'])} repeats: {p['output_tok_per_s_mean']:.0f} vs {base['output_tok_per_s_mean']:.0f} tok/s", cmd, art)
            L.add("ab_micro_on", f"g4.micro.c{c}.tok_per_s_2x2", p["output_tok_per_s_mean"], "tok/s", "G4", "A/B arm", cmd, art)
            L.add("ab_micro_off", f"g4.micro.c{c}.tok_per_s_1x1", base["output_tok_per_s_mean"], "tok/s", "G4", "A/B arm", cmd, "reports/g4/micro_1x1_summary.json")
            for key, label, unit_fmt in (("itl_ms", "itl", ".2f"), ("ttft_ms", "ttft", ".1f")):
                for field in ("p50", "p99"):
                    a, b = mean_field(p, key, field), mean_field(base, key, field)
                    L.add("ab_micro_on", f"g4.micro.c{c}.{label}_{field}_cut_pct", 100.0 * (1.0 - a / b), "%", "G4", f"{b:{unit_fmt}} ms to {a:{unit_fmt}} ms", cmd, art)
    if on and tile:
        for p in on["points"]:
            c = p["concurrency"]
            base = sweep_point(tile, c)
            L.add("ab_norm_tile", f"g4.norm_lane.c{c}.throughput_gain_pct", 100.0 * (p["output_tok_per_s_mean"] / base["output_tok_per_s_mean"] - 1.0), "%", "G4", f"per-lane vs per-tile norm: {p['output_tok_per_s_mean']:.0f} vs {base['output_tok_per_s_mean']:.0f} tok/s", "python3 scripts/bench/sweep.py --label norm_tile", "reports/g4/norm_tile_summary.json")
    if os.path.exists(os.path.join(root, "reports/g4/bits_auto.jsonl")):
        a = open(os.path.join(root, "reports/g4/bits_auto.jsonl")).read()
        for name in ("bits_1x1", "bits_normtile"):
            path = os.path.join(root, f"reports/g4/{name}.jsonl")
            if os.path.exists(path):
                L.add("ab_bits", f"g4.regression.{name}.identical", a == open(path).read(), "bool", "G4", "outputs and fingerprints compared with the optimized build, 100 prompts x 64 tokens", "scripts/run_all.sh ab_bits", f"reports/g4/{name}.jsonl")

    # G5.
    for name, label, stage in (("mutation_cpu_tiny", "cpu_tiny", "mut_cpu"), ("mutation_real_exact", "real_exact", "mut_exact"), ("mutation_real_tc", "real_tc", "mut_tc")):
        m = L.load(f"reports/g5/{name}.json")
        if not m:
            continue
        art = f"reports/g5/{name}.json"
        cmd = "python3 tests/mutations.py"
        tolerance = f"tolerance {m['tolerance']}" + (" and token match" if m["token_oracle"] else ", no token oracle")
        L.add(stage, f"g5.{label}.caught", m["caught"], f"of {m['mutants']}", "G5", f"missed: {m['missed']}", cmd, art)
        L.add(stage, f"g5.{label}.mutants", m["mutants"], "mutants", "G5", f"{m['functional_mutants']} functional, {len(m['rounding_only_mutants'])} rounding-only", cmd, art)
        L.add(stage, f"g5.{label}.functional_mutants", m["functional_mutants"], "mutants", "G5", "mutants that change behaviour, not only rounding", cmd, art)
        L.add(stage, f"g5.{label}.functional_caught", m["functional_caught"], f"of {m['functional_mutants']}", "G5", "any oracle, crash or hang", cmd, art)
        L.add(stage, f"g5.{label}.rounding_only_mutants", len(m["rounding_only_mutants"]), "mutants", "G5", str(m["rounding_only_mutants"]), cmd, art)
        L.add(stage, f"g5.{label}.catch_rate_pct", m["catch_rate_pct"], "%", "G5", "any oracle, crash or hang", cmd, art)
        L.add(stage, f"g5.{label}.caught_by_tolerance", m["caught_by_oracle"].get("tolerance", 0), f"of {m['mutants']}", "G5", f"{tolerance}, one request at a time on the full-logits path", cmd, art)
        L.add(stage, f"g5.{label}.functional_caught_by_tolerance", m["functional_caught_by_tolerance"], f"of {m['functional_mutants']}", "G5", tolerance, cmd, art)
        L.add(stage, f"g5.{label}.caught_only_by_crash_or_hang", len(m["caught_only_by_crash_or_hang"]), "mutants", "G5", str(m["caught_only_by_crash_or_hang"]), cmd, art)
        L.add(stage, f"g5.{label}.clean_preemptions", m["clean"]["preemptions"], "preemptions", "G5", "replay path exercised in the clean run", cmd, art)
        L.add(stage, f"g5.{label}.clean_max_logit_diff", m["clean"]["max_logit_diff"], "logit units", "G5", "clean build against the reference", cmd, art)

    # Source count. Each listed kernel merges at least two model operations and is launched at runtime.
    fused = ["Qkv", "GemmResidual", "Mlp13", "LmGreedy", "TcQkv", "TcGemmResidual", "TcMlp13", "TcLmGreedy", "AttentionPage"]
    source = open(os.path.join(root, "src/kernels.cu")).read() + open(os.path.join(root, "src/kernels_tc.cu")).read() + open(os.path.join(root, "src/kernel_common.h")).read()
    present = [k for k in fused if re.search(rf"struct {k}\b", source)]
    L.add("build", "g3.fused_kernels", len(present), "kernels", "G3", ", ".join(present) + "; see docs/ARCHITECTURE.md", "grep in scripts/build_ledger.py", "docs/ARCHITECTURE.md")

    out = os.path.join(root, "docs/resume/METRICS.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        for row in L.rows:
            f.write(json.dumps(row) + "\n")
    print(f"{len(L.rows)} rows; missing: {sorted(set(L.missing))}", file=sys.stderr)


if __name__ == "__main__":
    main()
