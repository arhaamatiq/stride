#!/usr/bin/env python3
"""Fails if a file contains a token-shaped string. Prints where, never the value.

Profilers and logs can capture the process environment: nsys embeds it in .nsys-rep
files. scripts/run_all.sh scans its output, and the pre-push check runs this too.
"""
import gzip
import os
import re
import sys

PATTERNS = {
    "runpod": rb"rpa_[A-Za-z0-9]{20,}",
    "github": rb"gh[pousr]_[A-Za-z0-9]{20,}",
    "huggingface": rb"hf_[A-Za-z0-9]{20,}",
    "aws": rb"AKIA[0-9A-Z]{16}",
    "generic_assignment": rb"(?:API_KEY|AUTH_TOKEN|ACCESS_TOKEN|SECRET_KEY|PASSWORD)=[^\s\"\x00]{8,}",
}
SKIP_DIRS = {"models", "__pycache__"}


def main():
    roots = sys.argv[1:] or ["."]
    found = 0
    for root in roots:
        for directory, subdirs, files in os.walk(root):
            subdirs[:] = [d for d in subdirs if d not in SKIP_DIRS and (not d.startswith(".") or d == ".github") and not d.startswith("build") and d != "data"]
            for name in files:
                path = os.path.join(directory, name)
                if os.path.abspath(path) == os.path.abspath(__file__):
                    continue
                try:
                    with open(path, "rb") as f:
                        blob = f.read()
                    if name.endswith(".gz"):
                        blob = gzip.decompress(blob)
                except OSError:
                    continue
                for kind, pattern in PATTERNS.items():
                    count = len(re.findall(pattern, blob))
                    if count:
                        print(f"{path}: {count} {kind} match(es)", file=sys.stderr)
                        found += 1
    if found:
        sys.exit(f"{found} secret finding(s)")
    print("no secrets found")


if __name__ == "__main__":
    main()
