#!/usr/bin/env python3
"""Gate G5: bug-injection suite. Each mutant is one realistic bug compiled into the
STRIDE_MUTATION build and selected at run time with --bug.

A mutant counts as caught if any oracle trips or the run crashes or hangs:
  tolerance    logits differ from the PyTorch reference beyond the tolerance, or a
               greedy token differs. This is what a conventional harness checks.
  invariance   outputs change between concurrency 1 and a shuffled concurrent run.
  golden       fingerprints differ from the clean build, down to a single bit.
The report keeps the per-oracle results, because the interesting number is how many
bugs only the bit-exact oracles can see.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter

import numpy as np

# id: (where, description)
MUTANTS = {
    101: ("embed", "sign flip on one embedding element"),
    102: ("norm", "RMSNorm partial sums combined in reverse order"),
    103: ("norm", "mean divides by dim - 1"),
    104: ("norm", "epsilon dropped from RMSNorm"),
    105: ("norm", "learned norm gain ignored for one dim"),
    111: ("rope", "RoPE uses position + 1"),
    112: ("rope", "sign error in one RoPE rotation term"),
    113: ("kv", "K and V swapped on append"),
    121: ("attention", "query cannot see its own position"),
    122: ("attention", "score scale dropped"),
    123: ("attention", "softmax denominator summed in reverse order"),
    124: ("attention", "value mix reads K instead of V"),
    125: ("kv", "value mix reads the previous KV page"),
    126: ("attention", "query heads swapped inside a KV group"),
    131: ("gemm", "partial sums combined in reverse order"),
    132: ("gemm", "residual connection dropped"),
    141: ("mlp", "SiLU skipped"),
    142: ("mlp", "SiLU applied to the wrong projection"),
    151: ("sampling", "argmax picks the minimum"),
    201: ("scheduler", "last row of a prefill chunk gets the wrong position"),
    202: ("scheduler", "preempted sequence resumes without replaying its KV"),
    203: ("scheduler", "decode row feeds the previous token"),
    204: ("scheduler", "logits routed to the wrong sequence after a lane shuffle"),
    205: ("kv", "freed block returned to the pool twice"),
    206: ("kv", "new block not synced to the device table"),
    207: ("scheduler", "generated-token count off by one"),
    301: ("engine", "all layers read KV from layer zero"),
    302: ("engine", "final norm uses a layer norm's weights"),
    303: ("engine", "logit lanes compacted from the wrong row"),
}


def run_cli(cli, model, prompts, output, bug, timeout, extra, summary=None):
    cmd = [cli, "--model", model, "--input", prompts, "--output", output, "--bug", str(bug), *extra]
    try:
        r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=timeout, text=True)
    except subprocess.TimeoutExpired:
        return "hang"
    if r.returncode != 0:
        return "crash"
    if summary is not None:
        summary.update(json.loads(r.stderr.strip().splitlines()[-1]))
    return "ok"


def read_rows(path):
    return {r["id"]: r for r in map(json.loads, open(path))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", required=True, help="stride_cli built with STRIDE_MUTATION=ON")
    parser.add_argument("--model", required=True, help="tiny model directory")
    parser.add_argument("--reference", required=True, help="reference jsonl from tests/reference.py")
    parser.add_argument("--reference-logits", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--max-new", type=int, default=32)
    parser.add_argument("--dump-steps", type=int, default=4)
    parser.add_argument("--path", default="exact", choices=["exact", "tc"])
    parser.add_argument("--tolerance", type=float, default=2e-5)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--kv-gb", default="0.25")
    parser.add_argument("--tight-blocks", default="16", help="KV blocks in the run that must preempt")
    parser.add_argument("--tight-max-new", default="200", help="tokens per prompt in the preemption run")
    parser.add_argument("--max-rows", default="256")
    parser.add_argument("--no-token-oracle", action="store_true", help="BF16 path: token-level agreement with an FP32 reference is not expected")
    parser.add_argument("--min-catch-rate", type=float, default=0.0, help="exit non-zero below this percentage")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    tmp = tempfile.mkdtemp()
    vocab = json.load(open(os.path.join(args.model, "config.json")))["vocab_size"]
    base = ["--max-new", str(args.max_new), "--path", args.path, "--max-rows", args.max_rows]
    ref_rows = read_rows(args.reference)
    ref_logits = np.fromfile(args.reference_logits, dtype=np.float32)

    def evaluate(bug):
        solo, solo_bin, mixed, tight = (os.path.join(tmp, f"{bug}_{n}") for n in ("solo.jsonl", "solo.bin", "mixed.jsonl", "tight.jsonl"))
        status = run_cli(args.cli, args.model, args.prompts, solo, bug, args.timeout,
                         [*base, "--kv-gb", args.kv_gb, "--concurrency", "1", "--full", "--dump-logits", solo_bin, "--dump-steps", str(args.dump_steps)])
        if status != "ok":
            return {"status": status}
        status = run_cli(args.cli, args.model, args.prompts, mixed, bug, args.timeout,
                         [*base, "--kv-gb", args.kv_gb, "--concurrency", "24", "--row-shuffle", "99", "--kv-shuffle", "5", "--prefill-chunk", "7", "--arrival-seed", "3"])
        if status != "ok":
            return {"status": status}
        # Long generations in a small KV arena force preemption and replay; the
        # solo run at the same length is the baseline for that comparison.
        long_solo = os.path.join(tmp, f"{bug}_long.jsonl")
        long_base = ["--max-new", args.tight_max_new, "--path", args.path, "--max-rows", args.max_rows]
        status = run_cli(args.cli, args.model, args.prompts, long_solo, bug, args.timeout, [*long_base, "--kv-gb", args.kv_gb, "--concurrency", "1"])
        if status != "ok":
            return {"status": status}
        tight_summary = {}
        status = run_cli(args.cli, args.model, args.prompts, tight, bug, args.timeout,
                         [*long_base, "--concurrency", "24", "--prefill-chunk", "9", "--kv-blocks", args.tight_blocks, "--max-slots", "24"],
                         tight_summary)
        if status != "ok":
            return {"status": status}
        logits = np.fromfile(solo_bin, dtype=np.float32)
        rows, mixed_rows, tight_rows, long_rows = read_rows(solo), read_rows(mixed), read_rows(tight), read_rows(long_solo)
        tokens_ok = args.no_token_oracle or all(rows[i]["tokens"] == ref_rows[i]["tokens"] for i in ref_rows)
        max_diff = float(np.abs(logits - ref_logits).max()) if logits.size == ref_logits.size else float("inf")
        return {
            "status": "ok",
            "fingerprints": {i: rows[i]["fingerprint"] for i in rows},
            "tokens_match_reference": tokens_ok,
            "max_logit_diff": max_diff,
            "invariant_mixed": all(rows[i]["fingerprint"] == mixed_rows[i]["fingerprint"] for i in rows),
            "invariant_tight": all(long_rows[i]["fingerprint"] == tight_rows[i]["fingerprint"] for i in rows),
            "preemptions": tight_summary.get("preemptions", 0),
        }

    clean = evaluate(0)
    if clean.get("status") != "ok":
        sys.exit(f"clean build failed: {clean}")
    if not (clean["invariant_mixed"] and clean["invariant_tight"]):
        sys.exit(f"clean build is not invariant: {clean}")
    if clean["preemptions"] == 0:
        sys.exit("the tight KV run did not preempt, so the replay path is untested; lower --tight-blocks")
    tolerance_ok = clean["max_logit_diff"] <= args.tolerance and clean["tokens_match_reference"]
    if not tolerance_ok:
        sys.exit(f"clean build fails the tolerance oracle: {clean['max_logit_diff']} > {args.tolerance}")

    results = {}
    for bug, (where, what) in MUTANTS.items():
        r = evaluate(bug)
        if r["status"] != "ok":
            caught_by = [r["status"]]
            detail = {}
        else:
            caught_by = []
            if not r["tokens_match_reference"] or r["max_logit_diff"] > args.tolerance:
                caught_by.append("tolerance")
            if not (r["invariant_mixed"] and r["invariant_tight"]):
                caught_by.append("invariance")
            if r["fingerprints"] != clean["fingerprints"]:
                caught_by.append("golden")
            detail = {"max_logit_diff": r["max_logit_diff"], "tokens_match_reference": r["tokens_match_reference"]}
        results[bug] = {"where": where, "what": what, "caught": bool(caught_by), "caught_by": caught_by, **detail}
        print(f"{bug:4d} {where:10s} {'CAUGHT' if caught_by else 'MISSED'} {','.join(caught_by):32s} {what}", flush=True)

    caught = sum(r["caught"] for r in results.values())
    by_oracle = Counter(o for r in results.values() for o in r["caught_by"])
    only_exact = [b for b, r in results.items() if r["caught"] and "tolerance" not in r["caught_by"] and not set(r["caught_by"]) & {"crash", "hang"}]
    report = {
        "path": args.path,
        "tolerance": args.tolerance,
        "clean": {k: clean[k] for k in ("max_logit_diff", "tokens_match_reference", "invariant_mixed", "invariant_tight", "preemptions")},
        "clean_within_tolerance": tolerance_ok,
        "mutants": len(results),
        "caught": caught,
        "catch_rate_pct": 100.0 * caught / len(results),
        "caught_by_oracle": dict(by_oracle),
        "caught_only_by_bit_exact_oracles": only_exact,
        "missed": [b for b, r in results.items() if not r["caught"]],
        "results": results,
    }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"caught {caught}/{len(results)}; tolerance oracle alone {by_oracle.get('tolerance', 0)}; only bit-exact {len(only_exact)}; missed {report['missed']}")
    if report["catch_rate_pct"] < args.min_catch_rate:
        sys.exit(f"catch rate {report['catch_rate_pct']:.1f}% is below the required {args.min_catch_rate}%")


if __name__ == "__main__":
    main()
