#!/usr/bin/env python3
"""Masks secrets inside .nsys-rep files.

Nsight Systems stores an environment block in its reports. On container hosts that
block can hold API keys even when the profiled process ran with an empty environment.
Values are overwritten with the same number of bytes, which keeps the file layout valid;
callers should confirm the report still opens with `nsys stats`.
"""
import re
import sys

PATTERNS = [
    rb"rpa_[A-Za-z0-9]{20,}",
    rb"gh[pousr]_[A-Za-z0-9]{20,}",
    rb"hf_[A-Za-z0-9]{20,}",
    rb"(?<=API_KEY=)[^\s\"\x00-\x1f]{8,}",
    rb"(?<=TOKEN=)[^\s\"\x00-\x1f]{8,}",
]


def main():
    total = 0
    for path in sys.argv[1:]:
        with open(path, "rb") as f:
            blob = f.read()
        count = 0
        for pattern in PATTERNS:
            blob, n = re.subn(pattern, lambda m: b"X" * len(m.group(0)), blob)
            count += n
        with open(path, "wb") as f:
            f.write(blob)
        print(f"{path}: {count} value(s) masked")
        total += count
    return 0


if __name__ == "__main__":
    sys.exit(main())
