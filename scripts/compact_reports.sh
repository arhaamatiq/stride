#!/bin/bash
# Reduces a reports directory before committing: compresses per-request AIPerf exports
# and removes files that are regenerable or too large. Nothing the ledger reads is removed.
#   scripts/compact_reports.sh reports
set -eu
cd "${1:?usage: compact_reports.sh <reports dir>}"
find . -name inputs.json -delete
find . -type d -name logs -prune -exec rm -rf {} +
find . -name profile_export_aiperf.csv -delete
find . -name .aiperf_results_ready.json -delete
find . -name profile_export.jsonl -exec gzip -9f {} +
find . -name "*.sqlite" -delete
find . -name ".done_*" -delete
find . -path "*/xdev/*" -name logits.bin -delete
rm -rf .work_g1
