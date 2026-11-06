"""CPU correctness tests against the PyTorch reference, on a small random model.

Run from the repository root after building: pytest tests/test_correctness.py
Environment: STRIDE_CLI (default build-cpu/stride_cli), STRIDE_TINY (default tests/data/tiny).
"""
import json
import os
import random
import subprocess
import sys

import numpy as np
import pytest

CLI = os.environ.get("STRIDE_CLI", "build-cpu/stride_cli")
TINY = os.environ.get("STRIDE_TINY", "tests/data/tiny")
MAX_NEW = 32
DUMP_STEPS = 4
# Largest FP32 logit difference allowed against the reference. The clean engine is
# well inside it; see reports/g5 for the measured value.
TOLERANCE = 2e-4


def read(path):
    return {r["id"]: r for r in map(json.loads, open(path))}


def cli(tmp_path, name, *flags):
    out = tmp_path / f"{name}.jsonl"
    # A later --max-new overrides the default.
    r = subprocess.run([CLI, "--model", TINY, "--input", f"{TINY}/prompts.jsonl", "--output", str(out), "--max-new", str(MAX_NEW), *flags],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    return read(out), json.loads(r.stderr.strip().splitlines()[-1])


@pytest.fixture(scope="session", autouse=True)
def model():
    os.makedirs(TINY, exist_ok=True)
    if not os.path.exists(f"{TINY}/ref.bin"):
        subprocess.run([sys.executable, "tests/make_tiny_model.py", "--out", TINY], check=True)
        random.seed(7)
        with open(f"{TINY}/prompts.jsonl", "w") as f:
            for i in range(24):
                n = random.randint(1, 40)
                f.write(json.dumps({"id": i, "tokens": [random.randint(3, 1023) for _ in range(n)]}) + "\n")
        subprocess.run([sys.executable, "tests/reference.py", "--model", TINY, "--input", f"{TINY}/prompts.jsonl", "--output", f"{TINY}/ref.jsonl",
                        "--dump-logits", f"{TINY}/ref.bin", "--dump-steps", str(DUMP_STEPS), "--max-new", str(MAX_NEW)], check=True)


def test_matches_reference(tmp_path):
    rows, _ = cli(tmp_path, "solo", "--concurrency", "1", "--kv-gb", "0.25", "--full", "--dump-logits", str(tmp_path / "solo.bin"), "--dump-steps", str(DUMP_STEPS))
    ref = read(f"{TINY}/ref.jsonl")
    for i in ref:
        assert rows[i]["tokens"] == ref[i]["tokens"], f"prompt {i} diverged"
    a = np.fromfile(tmp_path / "solo.bin", dtype=np.float32)
    b = np.fromfile(f"{TINY}/ref.bin", dtype=np.float32)
    assert a.size == b.size
    assert np.abs(a - b).max() <= TOLERANCE


@pytest.mark.parametrize("flags", [
    ("--concurrency", "24", "--kv-gb", "0.25"),
    ("--concurrency", "24", "--kv-gb", "0.25", "--row-shuffle", "99"),
    ("--concurrency", "24", "--kv-gb", "0.25", "--kv-shuffle", "5"),
    ("--concurrency", "24", "--kv-gb", "0.25", "--prefill-chunk", "1"),
    ("--concurrency", "7", "--kv-gb", "0.25", "--prefill-chunk", "7", "--arrival-seed", "3"),
])
def test_bit_identical_under_any_schedule(tmp_path, flags):
    solo, _ = cli(tmp_path, "solo", "--concurrency", "1", "--kv-gb", "0.25")
    mixed, _ = cli(tmp_path, "mixed", *flags)
    for i in solo:
        assert solo[i]["fingerprint"] == mixed[i]["fingerprint"], f"prompt {i} changed under {flags}"


def test_preemption_replay_is_bit_exact(tmp_path):
    solo, _ = cli(tmp_path, "solo", "--concurrency", "1", "--kv-gb", "0.25", "--max-new", "200")
    tight, stats = cli(tmp_path, "tight", "--concurrency", "24", "--kv-blocks", "16", "--max-slots", "24", "--prefill-chunk", "9", "--max-new", "200")
    assert stats["preemptions"] > 0, "arena was large enough that nothing was preempted"
    for i in solo:
        assert solo[i]["fingerprint"] == tight[i]["fingerprint"]


def test_sampling_is_keyed_on_seed_and_position(tmp_path):
    a, _ = cli(tmp_path, "a", "--concurrency", "1", "--kv-gb", "0.25", "--temperature", "0.8", "--top-p", "0.9", "--seed", "11")
    b, _ = cli(tmp_path, "b", "--concurrency", "8", "--max-rows", "64", "--kv-gb", "0.25", "--temperature", "0.8", "--top-p", "0.9", "--seed", "11", "--row-shuffle", "4")
    c, _ = cli(tmp_path, "c", "--concurrency", "1", "--kv-gb", "0.25", "--temperature", "0.8", "--top-p", "0.9", "--seed", "12")
    for i in a:
        assert a[i]["tokens"] == b[i]["tokens"]
    assert any(a[i]["tokens"] != c[i]["tokens"] for i in a), "seed had no effect"
