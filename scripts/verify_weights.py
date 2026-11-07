#!/usr/bin/env python3
"""Fails unless a model directory is byte-identical to weights.lock.json.

scripts/run_all.sh calls this at the start, and the G1, load-study and sweep stages call
it again and record the printed digest, so the engine, the reference and vLLM can be
shown to have loaded the same files.
"""
import argparse
import hashlib
import json
import os
import sys


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--lock", default=os.path.join(here, "..", "weights.lock.json"))
    args = parser.parse_args()

    with open(args.lock) as f:
        lock = json.load(f)
    bad = 0
    combined = hashlib.sha256()
    for entry in lock["files"]:
        path = os.path.join(args.dir, entry["path"])
        digest = sha256(path) if os.path.exists(path) else "missing"
        if digest != entry["sha256"]:
            print(f"MISMATCH {entry['path']}: {digest} != {entry['sha256']}", file=sys.stderr)
            bad += 1
        combined.update(digest.encode())
    expected = {e["path"] for e in lock["files"]}
    extra = set(os.listdir(args.dir)) - expected
    extra = {x for x in extra if not x.startswith(".")}
    if extra:
        print(f"UNEXPECTED FILES in {args.dir}: {sorted(extra)}", file=sys.stderr)
        bad += 1
    if bad:
        sys.exit(1)
    print(f"{lock['repo']}@{lock['commit']} snapshot_sha256={combined.hexdigest()} dir={os.path.realpath(args.dir)}")


if __name__ == "__main__":
    main()
