#!/usr/bin/env python3
"""Renders README.md and docs/resume/BULLETS.md from the metrics ledger.

Templates reference ledger rows as {{row.id|format}}, for example
{{g3.aiperf.stride.c64.output_tok_per_s|,.0f}}. A reference to a missing row is a hard
error, so a document can only state a number the ledger holds. Bullets over the length
limit are also an error.
"""
import argparse
import json
import os
import re
import sys

BULLET_LIMIT = 104


def render(text, metrics, source):
    def sub(match):
        key, _, fmt = match.group(1).partition("|")
        if key not in metrics:
            sys.exit(f"{source}: no ledger row named {key!r}")
        value = metrics[key]
        # floor keeps a claim from rounding up; floor, adds thousands separators.
        if fmt == "floor":
            return str(int(value))
        if fmt == "floor,":
            return f"{int(value):,}"
        return format(value, fmt) if fmt else str(value)

    return re.sub(r"\{\{([^{}]+)\}\}", sub, text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    args = parser.parse_args()
    root = os.path.abspath(args.root)
    with open(os.path.join(root, "docs/resume/METRICS.jsonl")) as f:
        metrics = {r["id"]: r["value"] for r in map(json.loads, f)}

    for template, output in (("docs/templates/README.md", "README.md"), ("docs/templates/BULLETS.md", "docs/resume/BULLETS.md")):
        with open(os.path.join(root, template)) as f:
            rendered = render(f.read(), metrics, template)
        with open(os.path.join(root, output), "w") as f:
            f.write(rendered)
        if output.endswith("BULLETS.md"):
            for line in rendered.splitlines():
                m = re.match(r"\d+\. (.*)", line)
                if m and len(m.group(1)) > BULLET_LIMIT:
                    sys.exit(f"bullet is {len(m.group(1))} characters, over {BULLET_LIMIT}: {m.group(1)}")
                if m:
                    print(f"{len(m.group(1)):3d} {m.group(1)}")


if __name__ == "__main__":
    main()
