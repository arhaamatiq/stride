#!/bin/bash
# Runs both kernel paths on a second GPU host with the same commit, weights and prompts.
# Run from the repository root on that host, with G1's prompts copied to $1. Writes host
# identity, per-run timings and outputs to reports/g1/host2. Afterward, record the stage
# with scripts/record_stage.py.
#   scripts/cross_host.sh /path/to/prompts.jsonl <git commit>
set -u
PROMPTS=${1:?usage: cross_host.sh <prompts.jsonl> <git commit>}
SOURCE_COMMIT=${2:?usage: cross_host.sh <prompts.jsonl> <git commit>}
DC=$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | grep -E '^RUNPOD_DC_ID=' | cut -d= -f2)
for v in $(env | cut -d= -f1 | grep -E 'KEY|TOKEN|SECRET|PASSWORD|RUNPOD'); do unset "$v"; done
export PATH=/usr/local/cuda/bin:$PATH
OUT=reports/g1/host2
mkdir -p "$OUT"
M=${MODEL_DIR:-models/llama32-1b}
COMMIT=$(python3 -c "import json;print(json.load(open('weights.lock.json'))['commit'])")
cmake -S . -B build -G Ninja -DSTRIDE_CUDA=ON -DSTRIDE_SERVER=OFF -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build build > /dev/null
python3 scripts/pin_weights.py --dest "$M" --lock /tmp/lock.json --commit "$COMMIT" > /dev/null
{
  date -u
  echo "commit $SOURCE_COMMIT"
  echo "provider RunPod Secure Cloud, data center ${DC:-unknown}"
  hostname
  nvidia-smi -L
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  grep -m1 "model name" /proc/cpuinfo
  python3 scripts/verify_weights.py --dir "$M"
} > "$OUT/host2_env.txt"
for path in tc exact; do
  build/stride_cli --path "$path" --model "$M" --input "$PROMPTS" --output "$OUT/host2_$path.jsonl" --max-new 256 --concurrency 1 --kv-gb 6 --max-rows 512 2> "$OUT/host2_$path.stderr.txt"
done
nvcc -O2 --gpu-architecture=sm_89 tools/ncu_trivial.cu -o /tmp/ncu_trivial
{ grep RmProfilingAdminOnly /proc/driver/nvidia/params; grep CapEff /proc/self/status; ncu --metrics gpu__time_duration.sum --csv /tmp/ncu_trivial 2>&1 | grep -E 'ERR_|gpu__time|PROF'; } > "$OUT/ncu_probe.txt" 2>&1
cat "$OUT/host2_env.txt" | cut -c1-110
tail -qn 1 "$OUT"/host2_*.stderr.txt | cut -c1-100
cut -c1-110 "$OUT/ncu_probe.txt"
