#!/usr/bin/env python3
"""Appends a provenance record for a stage run outside scripts/run_all.sh."""
import argparse
import datetime
import json

parser = argparse.ArgumentParser()
parser.add_argument("--file", required=True)
parser.add_argument("--stage", required=True)
parser.add_argument("--commit", required=True)
parser.add_argument("--device", required=True)
args = parser.parse_args()
with open(args.file, "a") as f:
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    f.write(json.dumps({"stage": args.stage, "commit": args.commit, "device": args.device, "finished_utc": now}) + "\n")
