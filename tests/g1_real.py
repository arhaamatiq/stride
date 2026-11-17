#!/usr/bin/env python3
"""Gate G1: differential correctness against HF transformers on the pinned weights.

Runs the PyTorch reference and both stride kernel paths on 100 prompts for 256
tokens each, all from one verified model directory, and writes reports/g1. Raw
outputs stay on disk; their SHA256 digests go in the summary.

Reference choices are recorded, not assumed:
  hf_fp32  BF16 weights, FP32 activations. The primary reference: it measures
           engine errors rather than BF16 rounding noise.
  hf_bf16  BF16 weights and activations, as vLLM serves them. Comparing it with
           hf_fp32 gives the noise floor of BF16 itself, which is the documented
           tolerance the tensor-core path is judged against.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

TOPICS = [
    "how a GPU schedules compute work", "the French Revolution", "photosynthesis", "binary search trees",
    "the causes of inflation", "how vaccines train the immune system", "plate tectonics", "the Roman aqueducts",
    "TCP congestion control", "the life of Marie Curie", "black holes", "how compilers perform register allocation",
    "the water cycle", "supply and demand", "the printing press", "quantum entanglement",
    "how airplanes generate lift", "the Silk Road", "garbage collection in programming languages", "coral reefs",
    "the theory of evolution", "how batteries store energy", "the Apollo program", "cache coherence protocols",
    "the history of jazz",
]
STYLES = [
    "Explain {t} to a curious teenager.",
    "Write a detailed essay about {t}.",
    "List ten surprising facts about {t}, one per line.",
    "Write a short story that involves {t}.",
]


def prompts():
    out = [s.format(t=t) for t in TOPICS for s in STYLES]
    assert len(out) == 100
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd, log):
    print("+", " ".join(cmd), flush=True)
    with open(log, "w") as f:
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)


def load_tokens(path):
    return {r["id"]: r for r in map(json.loads, open(path))}


def compare(a_path, b_path):
    a, b = load_tokens(a_path), load_tokens(b_path)
    match = total = identical = 0
    first_div = []
    for i in a:
        ta, tb = a[i]["tokens"], b[i]["tokens"]
        n = min(len(ta), len(tb))
        m = sum(x == y for x, y in zip(ta[:n], tb[:n]))
        match += m
        total += max(len(ta), len(tb))
        identical += ta == tb
        div = next((k for k in range(n) if ta[k] != tb[k]), None)
        if div is not None:
            first_div.append(div)
    return {
        "token_match": match,
        "token_total": total,
        "token_match_pct": 100.0 * match / total,
        "sequences_identical": identical,
        "sequences": len(a),
        "first_divergence_steps": sorted(first_div),
    }


def logit_stats(a_path, b_path, vocab):
    a = np.fromfile(a_path, dtype=np.float32)
    b = np.fromfile(b_path, dtype=np.float32)
    assert a.size == b.size and a.size % vocab == 0
    d = np.abs(a - b)
    per_vector_max = d.reshape(-1, vocab).max(axis=1)
    # Rank agreement at the head of the distribution, which is what decoding uses.
    ra, rb = a.reshape(-1, vocab), b.reshape(-1, vocab)
    top1 = float(np.mean(ra.argmax(axis=1) == rb.argmax(axis=1)))
    return {
        "vectors": int(a.size / vocab),
        "max_abs_diff": float(d.max()),
        "p50_vector_max_abs_diff": float(np.percentile(per_vector_max, 50)),
        "p99_vector_max_abs_diff": float(np.percentile(per_vector_max, 99)),
        "rms_diff": float(np.sqrt(np.mean(d.astype(np.float64) ** 2))),
        "logit_rms": float(np.sqrt(np.mean(a.astype(np.float64) ** 2))),
        "top1_agreement": top1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cli", required=True)
    parser.add_argument("--work", required=True, help="directory for raw outputs")
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--dump-steps", type=int, default=4)
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(args.work, exist_ok=True)
    os.makedirs(args.report, exist_ok=True)
    os.environ["HF_HUB_OFFLINE"] = "1"

    pin = subprocess.check_output(
        [sys.executable, os.path.join(here, "..", "scripts", "verify_weights.py"), "--dir", args.model], text=True
    ).strip()
    print(pin, flush=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    pfile = os.path.join(args.work, "prompts.jsonl")
    with open(pfile, "w") as f:
        for i, p in enumerate(prompts()):
            ids = tok.apply_chat_template(
                [{"role": "user", "content": p}], add_generation_prompt=True, date_string="06 Nov 2026"
            )
            if not isinstance(ids, list):
                ids = ids["input_ids"]
            f.write(json.dumps({"id": i, "tokens": ids}) + "\n")

    vocab = json.load(open(os.path.join(args.model, "config.json")))["vocab_size"]
    w = lambda n: os.path.join(args.work, n)
    ref = os.path.join(here, "reference.py")
    common = ["--model", args.model, "--input", pfile, "--max-new", str(args.max_new), "--dump-steps", str(args.dump_steps)]

    for dtype, name in (("float32", "hf_fp32"), ("bfloat16", "hf_bf16")):
        if not os.path.exists(w(name + ".done")):
            run([sys.executable, ref, *common, "--dtype", dtype, "--device", "cuda", "--output", w(name + ".jsonl"),
                 "--dump-logits", w(name + ".bin")], w(name + ".log"))
            open(w(name + ".done"), "w").close()

    engine_runs = {
        "stride_exact_c1": ["--path", "exact", "--concurrency", "1", "--full"],
        "stride_tc_c1": ["--path", "tc", "--concurrency", "1", "--full"],
        "stride_exact_c32": ["--path", "exact", "--concurrency", "32", "--row-shuffle", "7", "--kv-shuffle", "9", "--prefill-chunk", "13"],
        "stride_tc_c32": ["--path", "tc", "--concurrency", "32", "--row-shuffle", "7", "--kv-shuffle", "9", "--prefill-chunk", "13"],
    }
    for name, flags in engine_runs.items():
        dump = ["--dump-logits", w(name + ".bin")] if "--full" in flags else []
        run([args.cli, "--model", args.model, "--input", pfile, "--output", w(name + ".jsonl"), "--max-new",
             str(args.max_new), "--dump-steps", str(args.dump_steps), "--kv-gb", "6", "--max-rows", "512", *dump, *flags],
            w(name + ".log"))

    summary = {
        "pinned_weights": pin,
        "prompts": 100,
        "tokens_per_prompt": args.max_new,
        "logit_vectors_per_prompt": args.dump_steps,
        "greedy": {
            "stride_exact_vs_hf_fp32": compare(w("stride_exact_c1.jsonl"), w("hf_fp32.jsonl")),
            "stride_tc_vs_hf_fp32": compare(w("stride_tc_c1.jsonl"), w("hf_fp32.jsonl")),
            "stride_tc_vs_hf_bf16": compare(w("stride_tc_c1.jsonl"), w("hf_bf16.jsonl")),
            "hf_bf16_vs_hf_fp32_noise_floor": compare(w("hf_bf16.jsonl"), w("hf_fp32.jsonl")),
            "stride_exact_c1_vs_c32_shuffled": compare(w("stride_exact_c1.jsonl"), w("stride_exact_c32.jsonl")),
            "stride_tc_c1_vs_c32_shuffled": compare(w("stride_tc_c1.jsonl"), w("stride_tc_c32.jsonl")),
        },
        "logits": {
            "stride_exact_vs_hf_fp32": logit_stats(w("stride_exact_c1.bin"), w("hf_fp32.bin"), vocab),
            "stride_tc_vs_hf_fp32": logit_stats(w("stride_tc_c1.bin"), w("hf_fp32.bin"), vocab),
            "stride_tc_vs_hf_bf16": logit_stats(w("stride_tc_c1.bin"), w("hf_bf16.bin"), vocab),
            "hf_bf16_vs_hf_fp32_noise_floor": logit_stats(w("hf_bf16.bin"), w("hf_fp32.bin"), vocab),
        },
        "fingerprints_equal_c1_vs_c32": {},
        "raw_sha256": {},
    }
    for path in ("exact", "tc"):
        a, b = load_tokens(w(f"stride_{path}_c1.jsonl")), load_tokens(w(f"stride_{path}_c32.jsonl"))
        # --full hashes the same value bits, so c1 and c32 fingerprints are comparable.
        summary["fingerprints_equal_c1_vs_c32"][path] = sum(a[i]["fingerprint"] == b[i]["fingerprint"] for i in a)
    for name in os.listdir(args.work):
        if name.endswith((".jsonl", ".bin")):
            summary["raw_sha256"][name] = sha256(w(name))

    with open(os.path.join(args.report, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    for name in ("stride_exact_c1", "stride_tc_c1", "hf_fp32", "hf_bf16"):
        subprocess.run(["cp", w(name + ".jsonl"), os.path.join(args.report, name + ".jsonl")], check=True)
    print(json.dumps({k: v for k, v in summary.items() if k in ("greedy", "fingerprints_equal_c1_vs_c32")}, indent=1)[:3000], flush=True)


if __name__ == "__main__":
    main()
