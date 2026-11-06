#!/bin/bash
# Reproduces every number in docs/resume/METRICS.jsonl on one RTX 4090 host. Each stage
# writes raw artifacts under $OUT and is skipped if its marker exists, so an interrupted
# run resumes. Benchmarks run one at a time, never alongside anything else on the GPU.
set -u
COMMIT=${1:?usage: run_all.sh <git commit>}
# nsys embeds the process environment in its reports, and reports get committed.
for v in $(env | cut -d= -f1 | grep -E 'KEY|TOKEN|SECRET|PASSWORD|RUNPOD'); do unset "$v"; done
ROOT=$(cd "$(dirname "$0")/.." && pwd)
M=${MODEL_DIR:-/root/models/llama32-1b}
OUT=${OUT_DIR:-/workspace/reports}
WORK=${WORK_DIR:-/workspace/g1}
export PATH=/usr/local/cuda/bin:$PATH
cd "$ROOT"
mkdir -p "$OUT"/{g1,g2,g3,g4,g5,env}

stage() { # name command...
  local name=$1; shift
  if [ -f "$OUT/.done_$name" ]; then echo "[skip] $name"; return 0; fi
  echo "[$(date -u +%H:%M:%S)] start $name"
  if "$@" > "$OUT/log_$name.txt" 2>&1; then
    touch "$OUT/.done_$name"; echo "[$(date -u +%H:%M:%S)] done  $name"
  else
    echo "[$(date -u +%H:%M:%S)] FAILED $name (see log_$name.txt)"
  fi
}

python3 scripts/verify_weights.py --dir "$M" | tee "$OUT/env/weights_verified.txt"
{
  echo "commit $COMMIT"; date -u
  nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.memory,clocks.max.graphics,pstate --format=csv
  nvidia-smi | sed -n 3p
  nvcc --version | sed -n 1,2p
  ncu --version | head -1; nsys --version
  grep -m1 "model name" /proc/cpuinfo; nproc; cat /sys/fs/cgroup/cpu.max 2>/dev/null
  grep RmProfilingAdminOnly /proc/driver/nvidia/params
  /root/venv-vllm/bin/python -c "import vllm,torch,transformers;print('vllm',vllm.__version__,'torch',torch.__version__,'transformers',transformers.__version__)" 2>/dev/null | tail -1
  /root/venv-vllm/bin/aiperf --version 2>/dev/null | tail -1
  python3 -c "import torch,transformers;print('reference torch',torch.__version__,'transformers',transformers.__version__)"
  uname -r
} > "$OUT/env/environment.txt" 2>&1

build() {
  cmake -S . -B build -G Ninja -DSTRIDE_CUDA=ON -DSTRIDE_SERVER=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build &&
  cmake -S . -B build-mut -G Ninja -DSTRIDE_CUDA=ON -DSTRIDE_SERVER=OFF -DSTRIDE_MUTATION=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build-mut
}
stage build build

# G1: differential correctness. The PyTorch runs are reused when already present.
stage g1 python3 tests/g1_real.py --model "$M" --cli "$ROOT/build/stride_cli" --work "$WORK" --report "$OUT/g1"

# G3: batch-1 roofline, then Nsight Systems traces.
python3 - "$WORK/prompts.jsonl" /tmp/one_prompt.jsonl <<'PY'
import sys
open(sys.argv[2], "w").write(open(sys.argv[1]).readline())
PY
stage roofline python3 scripts/bench/roofline.py --cli build/stride_cli --model "$M" --prompt /tmp/one_prompt.jsonl --report "$OUT/g3/roofline.json"
nsys_b1() {
  for path in tc exact; do
    nsys profile --trace=cuda --cuda-memory-usage=false --sample=none --cpuctxsw=none -o "$OUT/g3/batch1_$path" --force-overwrite=true build/stride_cli --path $path --model "$M" --input /tmp/one_prompt.jsonl --output /tmp/o.jsonl --max-new 256 --kv-gb 2 &&
    nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv --output "$OUT/g3/batch1_${path}_kernels" "$OUT/g3/batch1_$path.nsys-rep"
    rm -f "$OUT/g3/batch1_$path.sqlite"
  done
}
stage nsys_b1 nsys_b1
stage nsys_server python3 scripts/bench/trace_server.py --model "$M" --out "$OUT/g3/server_c64"

# G2 and the nondeterminism study: same requests alone and under load, on both engines.
stage det_stride python3 scripts/bench/determinism.py --engine stride --model "$M" --report "$OUT/g2/determinism_stride.json" --port 8125 --levels 4,16,64,128
stage det_vllm python3 scripts/bench/determinism.py --engine vllm --model "$M" --report "$OUT/g2/determinism_vllm.json" --port 8126 --levels 4,16,64,128

# G3: AIPerf sweep on both engines, same instance, same settings.
stage sweep_stride python3 scripts/bench/sweep.py --engine stride --model "$M" --out "$OUT/g3/aiperf" --port 8127
stage sweep_vllm python3 scripts/bench/sweep.py --engine vllm --model "$M" --out "$OUT/g3/aiperf" --port 8128

# G4: register-blocked microkernel A/B, plus the per-lane norm, both bit-identical by construction.
stage ab_micro_off python3 scripts/bench/sweep.py --engine stride --label micro_1x1 --model "$M" --out "$OUT/g4" --concurrency 64,128 --port 8129 --stride-flag=--micro-r --stride-flag=1 --stride-flag=--micro-lt --stride-flag=1
stage ab_micro_on python3 scripts/bench/sweep.py --engine stride --label micro_2x2 --model "$M" --out "$OUT/g4" --concurrency 64,128 --port 8130 --stride-flag=--micro-r --stride-flag=2 --stride-flag=--micro-lt --stride-flag=2
stage ab_norm_tile python3 scripts/bench/sweep.py --engine stride --label norm_tile --model "$M" --out "$OUT/g4" --concurrency 64,128 --port 8131 --stride-flag=--norm-tile
ab_bits() {
  head -n 64 "$WORK/prompts.jsonl" > /tmp/ab.jsonl; cat /tmp/ab.jsonl > /tmp/ab_prompts.jsonl
  build/stride_cli --path tc --model "$M" --input /tmp/ab_prompts.jsonl --output "$OUT/g4/bits_auto.jsonl" --max-new 64 --kv-gb 8 --concurrency 64 --max-rows 2048 --prefill-chunk 256 &&
  build/stride_cli --path tc --model "$M" --input /tmp/ab_prompts.jsonl --output "$OUT/g4/bits_1x1.jsonl" --max-new 64 --kv-gb 8 --concurrency 64 --max-rows 2048 --prefill-chunk 256 --micro-r 1 --micro-lt 1 &&
  build/stride_cli --path tc --model "$M" --input /tmp/ab_prompts.jsonl --output "$OUT/g4/bits_normtile.jsonl" --max-new 64 --kv-gb 8 --concurrency 64 --max-rows 2048 --prefill-chunk 256 --norm-tile &&
  cmp "$OUT/g4/bits_auto.jsonl" "$OUT/g4/bits_1x1.jsonl" && cmp "$OUT/g4/bits_auto.jsonl" "$OUT/g4/bits_normtile.jsonl"
}
stage ab_bits ab_bits

# G5: bug injection on real weights, both paths.
mut_ref() {
  head -n 8 "$WORK/prompts.jsonl" > /tmp/mut_prompts.jsonl
  python3 tests/reference.py --model "$M" --input /tmp/mut_prompts.jsonl --output /tmp/mut_ref.jsonl --dump-logits /tmp/mut_ref.bin --dump-steps 4 --max-new 32 --dtype float32 --device cuda &&
  python3 tests/reference.py --model "$M" --input /tmp/mut_prompts.jsonl --output /tmp/mut_ref1.jsonl --dump-logits /tmp/mut_ref1.bin --dump-steps 1 --max-new 1 --dtype float32 --device cuda
}
stage mut_ref mut_ref
stage mut_exact python3 tests/mutations.py --cli build-mut/stride_cli --model "$M" --reference /tmp/mut_ref.jsonl --reference-logits /tmp/mut_ref.bin --prompts /tmp/mut_prompts.jsonl --path exact --tolerance 2e-4 --kv-gb 2 --tight-blocks 16 --timeout 300 --report "$OUT/g5/mutation_real_exact.json"
stage mut_tc python3 tests/mutations.py --cli build-mut/stride_cli --model "$M" --reference /tmp/mut_ref1.jsonl --reference-logits /tmp/mut_ref1.bin --prompts /tmp/mut_prompts.jsonl --path tc --no-token-oracle --dump-steps 1 --tolerance 0.6 --kv-gb 2 --tight-blocks 16 --timeout 300 --report "$OUT/g5/mutation_real_tc.json"

python3 scripts/redact_nsys.py "$OUT"/g3/*.nsys-rep "$OUT"/g3/server_c64/*.nsys-rep
python3 scripts/scan_secrets.py "$OUT" || echo "SECRET FOUND IN $OUT: do not commit these reports"
echo "[$(date -u +%H:%M:%S)] all stages finished"
ls -la "$OUT"/.done_* | wc -l
