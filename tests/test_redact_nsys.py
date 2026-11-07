#!/usr/bin/env python3
"""scripts/redact_nsys.py masks values split across LZ4 tokens, without changing the frame layout."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import redact_nsys  # noqa: E402


def length_bytes(n):
    out = b""
    while n >= 255:
        out += b"\xff"
        n -= 255
    return out + bytes([n])


def sequence(literal, offset=0, match=0):
    extra = match - 4 if match else 0
    out = bytes([min(len(literal), 15) << 4 | min(extra, 15)])
    if len(literal) >= 15:
        out += length_bytes(len(literal) - 15)
    out += literal
    if match:
        out += offset.to_bytes(2, "little")
        if extra >= 15:
            out += length_bytes(extra - 15)
    return out


def varint(n):
    out = b""
    while n >= 128:
        out += bytes([n & 127 | 128])
        n >>= 7
    return out + bytes([n])


def report():
    # RUNPOD_PUBLIC_IP=100.100.23.45 is split: its second octet is a back-reference to its first.
    block = sequence(b"PATH=/usr/bin\0", 14, 56) + sequence(b"RUNPOD_PUBLIC_IP=100.", 4, 4) + sequence(b"23.45\0RUNPOD_POD_ID=abcdef012345\0SSH_CLIENT=10.1.2.3 5555 22\0DONE=1\0")
    decoded = b"PATH=/usr/bin\0" * 5 + b"RUNPOD_PUBLIC_IP=100.100.23.45\0RUNPOD_POD_ID=abcdef012345\0SSH_CLIENT=10.1.2.3 5555 22\0DONE=1\0"
    header = b"\x0a\x10plain header\0SSH_CONNECTION=10.1.2.3 5555 172.16.0.2 22\x0a"
    return header + varint(len(block)) + varint(len(decoded)) + block + b"tail", decoded


def decoded_text(data):
    (start, end), = redact_nsys.find_frames(data)
    return bytes(redact_nsys.lz4_decode(data, start, end)[0])


def test_decoder_reads_split_values():
    data, decoded = report()
    assert decoded_text(data) == decoded
    assert redact_nsys.findings(data)["ipv4"] == 4


def test_masks_values_split_across_tokens():
    data, decoded = report()
    masked, _ = redact_nsys.redact(data)
    assert len(masked) == len(data)
    assert redact_nsys.find_frames(masked) == redact_nsys.find_frames(data)
    text = decoded_text(masked)
    assert len(text) == len(decoded)
    for value in (b"100.", b"23.45", b"abcdef012345", b"10.1.2.3", b"5555"):
        assert value not in text and value not in masked
    assert b"PATH=/usr/bin\0" in text and b"DONE=1\0" in text
    assert not any(v for k, v in redact_nsys.findings(masked).items() if k != "frames")


def test_masking_is_stable():
    data, _ = report()
    once, _ = redact_nsys.redact(data)
    assert redact_nsys.redact(once)[0] == once
