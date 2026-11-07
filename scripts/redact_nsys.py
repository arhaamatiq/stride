#!/usr/bin/env python3
"""Masks secrets and network details inside .nsys-rep files.

Nsight Systems stores an environment block in its reports. On container hosts that
block can hold API keys and the SSH client's address, even when the profiled process
ran with an empty environment.

The block appears twice: once as plain text in the report header, and again inside
LZ4-compressed frames, where a value is split across literal runs and back-references
and a text search only sees fragments. So this script decodes every frame, finds the
sensitive ranges in the decoded text, and overwrites the compressed bytes that produce
them. The file's size and frame structure do not change.

Usage: redact_nsys.py FILE [FILE...]        mask in place
       redact_nsys.py --check FILE [FILE...] exit 1 if anything sensitive remains
"""
import re
import sys

MASK = ord("X")
IPV4 = re.compile(rb"(?<![0-9.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9.])")
KEYS = (
    rb"SSH_CLIENT|SSH_CONNECTION|HOSTNAME|[A-Z0-9_]*PUBLIC_IP|[A-Z0-9_]*PUBLIC_KEY|[A-Z0-9_]*TCP_PORT_[0-9]+"
    rb"|[A-Z0-9_]*POD_ID|[A-Z0-9_]*POD_HOSTNAME|[A-Z0-9_]*API_KEY|[A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD)[A-Z0-9_]*"
)
ENV_VALUE = re.compile(rb"(?<![A-Za-z0-9_])(?:" + KEYS + rb")=([^\x00-\x1f\x7f-\xff]{1,512})")
# The same variables also appear as protobuf key/value pairs: KEY, then field 2 with a length byte.
KV_VALUE = re.compile(rb"(?<![A-Za-z0-9_])(?:" + KEYS + rb")\x12([\x01-\x7f])")
# An SSH_CLIENT or SSH_CONNECTION value, with or without its addresses already masked.
SSH_TUPLE = re.compile(
    rb"(?:X{7,}|(?:[0-9]{1,3}\.){3}[0-9]{1,3}) ([0-9]{1,5})(?: (?:X{7,}|(?:[0-9]{1,3}\.){3}[0-9]{1,3}))? ([0-9]{1,5})(?![0-9])"
)
TOKENS = re.compile(rb"rpa_[A-Za-z0-9]{8,}|gh[pousr]_[A-Za-z0-9]{8,}|hf_[A-Za-z0-9]{8,}|AKIA[0-9A-Z]{16}")
SSH_KEY = re.compile(rb"ssh-(?:ed25519|rsa) ([A-Za-z0-9+/=]{16,})")
USER_HOST = re.compile(rb"[A-Za-z0-9._-]+@[A-Za-z0-9._-]+\.local")


def varint(data, i):
    value = shift = 0
    while True:
        b = data[i]
        i += 1
        value |= (b & 0x7F) << shift
        shift += 7
        if not b & 0x80:
            return value, i
        if shift > 35:
            raise ValueError("varint too long")


def lz4_sequences(src, start, end):
    """Yields (literal_start, literal_len, offset_pos, offset, match_len) for one LZ4 block."""
    i = start
    while i < end:
        token = src[i]
        i += 1
        lit = token >> 4
        if lit == 15:
            while True:
                b = src[i]
                i += 1
                lit += b
                if b != 255:
                    break
        lit_start = i
        i += lit
        if i > end:
            raise ValueError("literal run past end of block")
        if i == end:
            yield lit_start, lit, None, 0, 0
            return
        offset = src[i] | (src[i + 1] << 8)
        offset_pos = i
        i += 2
        match = token & 15
        if match == 15:
            while True:
                b = src[i]
                i += 1
                match += b
                if b != 255:
                    break
        yield lit_start, lit, offset_pos, offset, match + 4
    raise ValueError("block did not end on literals")


def lz4_decode(src, start, end):
    """Returns (output, provenance, matches). provenance[i] is the file offset of the literal
    behind output byte i. matches lists (output_start, length, offset_pos, offset)."""
    out = bytearray()
    prov = []
    matches = []
    for lit_start, lit, offset_pos, offset, match in lz4_sequences(src, start, end):
        out += src[lit_start:lit_start + lit]
        prov.extend(range(lit_start, lit_start + lit))
        if offset_pos is None:
            break
        if offset == 0 or offset > len(out):
            raise ValueError("match reaches before start of block")
        begin = len(out)
        for k in range(match):
            out.append(out[begin - offset + k])
            prov.append(prov[begin - offset + k])
        matches.append((begin, match, offset_pos, offset))
    return out, prov, matches


def try_frame(data, i, limit):
    """Returns (block_start, block_end) if a valid [compressed][uncompressed][LZ4] frame starts at i."""
    try:
        compressed, j = varint(data, i)
        uncompressed, j = varint(data, j)
    except (ValueError, IndexError):
        return None
    if not 16 <= compressed <= limit - j or not compressed <= uncompressed <= 1 << 26:
        return None
    try:
        total = 0
        for _, lit, offset_pos, offset, match in lz4_sequences(data, j, j + compressed):
            total += lit
            if offset_pos is None:
                break
            if offset == 0 or offset > total:
                return None
            total += match
    except (ValueError, IndexError):
        return None
    return (j, j + compressed) if total == uncompressed else None


def find_frames(data):
    """Finds every LZ4 frame. Frames chain directly; plain protobuf can sit between chains."""
    frames = []
    i = 0
    n = len(data)
    while i < n:
        frame = try_frame(data, i, n)
        if frame:
            frames.append(frame)
            i = frame[1]
        else:
            i += 1
    return frames


def sensitive_ranges(text, literals):
    ranges = []
    for m in ENV_VALUE.finditer(text):
        ranges.append(m.span(1))
    for m in KV_VALUE.finditer(text):
        ranges.append((m.end(), min(len(text), m.end() + m.group(1)[0])))
    for m in SSH_TUPLE.finditer(text):
        ranges.extend((m.span(1), m.span(2)))
    for rx in (IPV4, TOKENS, USER_HOST):
        for m in rx.finditer(text):
            ranges.append(m.span(0))
    for m in SSH_KEY.finditer(text):
        ranges.append(m.span(1))
    for value in literals:
        start = text.find(value)
        while start != -1:
            ranges.append((start, start + len(value)))
            start = text.find(value, start + 1)
    return ranges


def harvest(text):
    """Values worth searching for elsewhere in the report: long ones, so short ones cannot clobber data."""
    found = set()
    for m in ENV_VALUE.finditer(text):
        for part in re.split(rb"[ X]+", m.group(1)):
            if len(part) >= 8:
                found.add(part)
    return found


def redact(data):
    """Masks until a pass changes nothing: masks written by one pass give the next pass runs to point at."""
    while True:
        masked, stats = redact_pass(data)
        if masked == data:
            return masked, stats
        data = masked


def redact_pass(data):
    data = bytearray(data)
    frames = find_frames(bytes(data))
    plain_parts = []
    cursor = 0
    for start, end in frames:
        plain_parts.append((cursor, start))
        cursor = end
    plain_parts.append((cursor, len(data)))

    decoded = [lz4_decode(data, start, end) for start, end in frames]
    literals = set()
    for a, b in plain_parts:
        literals |= harvest(bytes(data[a:b]))
    for out, _, _ in decoded:
        literals |= harvest(bytes(out))
    literals = sorted(literals, key=len, reverse=True)

    stats = {"frames": len(frames), "plain_bytes": 0, "literal_bytes": 0, "retargeted": 0, "collateral": 0, "unresolved": 0}
    for a, b in plain_parts:
        text = bytes(data[a:b])
        for s, e in sensitive_ranges(text, literals):
            for k in range(a + s, a + e):
                if data[k] != MASK:
                    data[k] = MASK
                    stats["plain_bytes"] += 1

    for out, prov, matches in decoded:
        marked = bytearray(len(out))
        for s, e in sensitive_ranges(bytes(out), literals):
            marked[s:e] = b"\1" * (e - s)
        # A literal is masked when its first appearance is itself sensitive.
        first = {}
        for pos, source in enumerate(prov):
            first.setdefault(source, pos)
        for pos in range(len(out)):
            if marked[pos] and marked[first[prov[pos]]] and data[prov[pos]] != MASK:
                data[prov[pos]] = MASK
                stats["literal_bytes"] += 1
        # What is left was copied from text that is not sensitive. Point those matches at a run of masks.
        current = bytearray(len(out))
        for pos, source in enumerate(prov):
            current[pos] = data[source]
        for begin, length, offset_pos, offset in matches:
            span = range(begin, begin + length)
            if not any(marked[p] and current[p] != MASK for p in span):
                continue
            needle = bytes([MASK]) * length
            lo = max(0, begin - 65535)
            hit = bytes(current[lo:begin]).rfind(needle)
            if hit == -1:
                stats["unresolved"] += 1
                continue
            new_offset = begin - (lo + hit)
            data[offset_pos] = new_offset & 255
            data[offset_pos + 1] = new_offset >> 8
            stats["collateral"] += sum(1 for p in span if not marked[p] and current[p] != MASK)
            stats["retargeted"] += 1
            for p in span:
                current[p] = MASK
    return bytes(data), stats


def unmasked(value):
    """The part of a value that is not mask. A masked separator can run a value into the next env entry."""
    value = re.sub(rb"[A-Za-z_][A-Za-z0-9_]*=", b"", value)
    return re.sub(rb"X+", b"", value)


def findings(data):
    """Counts what is still readable, in plain sections and in decoded frames."""
    frames = find_frames(data)
    texts = []
    cursor = 0
    for start, end in frames:
        texts.append(data[cursor:start])
        texts.append(bytes(lz4_decode(data, start, end)[0]))
        cursor = end
    texts.append(data[cursor:])
    counts = {"frames": len(frames), "ipv4": 0, "env_value": 0, "token": 0, "ssh_key": 0, "user_host": 0}
    for text in texts:
        counts["ipv4"] += len(IPV4.findall(text))
        counts["token"] += len(TOKENS.findall(text))
        counts["user_host"] += len(USER_HOST.findall(text))
        counts["ssh_key"] += sum(1 for m in SSH_KEY.finditer(text) if set(m.group(1)) - {MASK})
        counts["env_value"] += sum(1 for m in ENV_VALUE.finditer(text) if unmasked(m.group(1)))
    return counts


def main():
    args = sys.argv[1:]
    check = args[:1] == ["--check"]
    failed = 0
    for path in args[1:] if check else args:
        with open(path, "rb") as f:
            blob = f.read()
        if check:
            counts = findings(blob)
            print(f"{path}: {counts}")
            failed += any(v for k, v in counts.items() if k != "frames")
            continue
        masked, stats = redact(blob)
        assert len(masked) == len(blob)
        with open(path, "wb") as f:
            f.write(masked)
        print(f"{path}: {stats}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
