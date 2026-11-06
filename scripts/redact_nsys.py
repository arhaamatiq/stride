#!/usr/bin/env python3
"""Masks secrets and network details inside .nsys-rep files.

Nsight Systems stores an environment block in its reports. On container hosts that
block can hold API keys and the SSH client's address, even when the profiled process
ran with an empty environment. Values are overwritten with the same number of bytes,
which keeps the file layout valid; callers should confirm the report still opens with
`nsys stats`.
"""
import re
import sys

PATTERNS = [
    rb"rpa_[A-Za-z0-9]{20,}",
    rb"gh[pousr]_[A-Za-z0-9]{20,}",
    rb"hf_[A-Za-z0-9]{20,}",
    rb"(?<=API_KEY=)[^\s\"\x00-\x1f]{8,}",
    rb"(?<=TOKEN=)[^\s\"\x00-\x1f]{8,}",
    # Network identity: every IPv4-shaped string, and the SSH key material.
    rb"(?<![0-9.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9.])",
    rb"(?<=ssh-ed25519 )[A-Za-z0-9+/=]+",
    rb"[A-Za-z0-9._-]+@[A-Za-z0-9._-]+\.local",
]
# Keys whose whole value should go, up to the next control character.
VALUE_KEYS = rb"(?:SSH_CLIENT|SSH_CONNECTION|CONNECTION|PUBLIC_IP|PUBLIC_KEY|TCP_PORT_22|POD_ID|POD_HOSTNAME)="


def mask(match):
    return match.group(1) + b"X" * len(match.group(2))


def main():
    for path in sys.argv[1:]:
        with open(path, "rb") as f:
            blob = f.read()
        size = len(blob)
        count = 0
        for pattern in PATTERNS:
            blob, n = re.subn(pattern, lambda m: b"X" * len(m.group(0)), blob)
            count += n
        blob, n = re.subn(rb"(" + VALUE_KEYS + rb")([^\x00-\x1f]{1,120})", mask, blob)
        count += n
        assert len(blob) == size
        with open(path, "wb") as f:
            f.write(blob)
        print(f"{path}: {count} value(s) masked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
