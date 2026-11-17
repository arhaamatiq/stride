#!/bin/bash
# Regenerates the benchmark evidence on one RTX 4090 host. Each stage writes raw artifacts
# under $OUT_DIR, appends a provenance record, and is skipped if its marker exists, so an
# interrupted run resumes. Benchmarks run one at a time, never alongside anything else on
# the GPU. Two pieces cannot run here and are recorded separately: the ARM leg of
# cross_device needs an ARM machine, and cross_host needs a second GPU host. See the README.
#
#   OUT_DIR=$PWD/reports MODEL_DIR=/path/to/model BENCH_VENV=/path/to/venv scripts/run_all.sh <git commit>
#
# BENCH_VENV holds vLLM and AIPerf. The system Python needs torch and transformers, and
# nvcc, nsys and ncu must be on PATH. Run scripts/compact_reports.sh on OUT_DIR before
# committing; it drops regenerable files and compresses per-request exports.
set -u
COMMIT=${1:?usage: run_all.sh <git commit>}
DC=$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | grep -E '^RUNPOD_DC_ID=' | cut -d= -f2)
# nsys embeds the process environment in its reports, and reports get committed.
for v in $(env | cut -d= -f1 | grep -E 'KEY|TOKEN|SECRET|PASSWORD|RUNPOD'); do unset "$v"; done
ROOT=$(cd "$(dirname "$0")/.." && pwd)
M=${MODEL_DIR:-/root/models/llama32-1b}
OUT=${OUT_DIR:-$ROOT/reports}
WORK=${WORK_DIR:-$OUT/.work_g1}
export BENCH_VENV=${BENCH_VENV:-/root/venv-vllm}
export PATH=/usr/local/cuda/bin:$PATH
BENCH=(--stride-bin "$ROOT/build/stride_server")
cd "$ROOT"
mkdir -p "$OUT"/{g1,g2,g3,g4,g5,env}
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | sed 's/NVIDIA GeForce //')

stage() { # name device command...
  local name=$1 device=$2; shift 2
  if [ -f "$OUT/.done_$name" ]; then echo "[skip] $name"; return 0; fi
  echo "[$(date -u +%H:%M:%S)] start $name"
  if "$@" > "$OUT/log_$name.txt" 2>&1; then
    touch "$OUT/.done_$name"
    python3 scripts/record_stage.py --file "$OUT/env/stages.jsonl" --stage "$name" --commit "$COMMIT" --device "$device"
    echo "[$(date -u +%H:%M:%S)] done  $name"
  else
    echo "[$(date -u +%H:%M:%S)] FAILED $name (see log_$name.txt)"
  fi
}

python3 scripts/verify_weights.py --dir "$M" | tee "$OUT/env/weights_verified_${COMMIT:0:12}.txt"
{
  echo "commit $COMMIT"; date -u
  echo "provider RunPod Secure Cloud, data center ${DC:-unknown}"
  nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.memory,clocks.max.graphics,pstate --format=csv
  nvidia-smi -L
  nvidia-smi | sed -n 3p
  nvcc --version | grep release
  ncu --version | head -1; nsys --version
  grep -m1 "model name" /proc/cpuinfo; nproc; cat /sys/fs/cgroup/cpu.max 2>/dev/null
  grep RmProfilingAdminOnly /proc/driver/nvidia/params
  "$BENCH_VENV/bin/python" -c "import vllm,torch,transformers;print('vllm',vllm.__version__,'torch',torch.__version__,'transformers',transformers.__version__)" 2>/dev/null | tail -1
  python3 -c "import torch,transformers;print('reference torch',torch.__version__,'transformers',transformers.__version__)"
  uname -r
} > "$OUT/env/environment_${COMMIT:0:12}.txt" 2>&1

build() {
  cmake -S . -B build -G Ninja -DSTRIDE_CUDA=ON -DSTRIDE_SERVER=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build &&
  cmake -S . -B build-mut -G Ninja -DSTRIDE_CUDA=ON -DSTRIDE_SERVER=OFF -DSTRIDE_MUTATION=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build-mut &&
  cmake -S . -B build-cpu -G Ninja -DSTRIDE_SERVER=OFF -DCMAKE_BUILD_TYPE=Release && cmake --build build-cpu &&
  cmake -S . -B build-cpu-mut -G Ninja -DSTRIDE_SERVER=OFF -DSTRIDE_MUTATION=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build-cpu-mut
}
stage build "$GPU" build

# Hardware probes behind the kernel design, and whether GPU counters are available.
ncu_probe() {
  nvcc -O2 --gpu-architecture=sm_89 tools/ncu_trivial.cu -o /tmp/ncu_trivial &&
  { grep RmProfilingAdminOnly /proc/driver/nvidia/params; grep CapEff /proc/self/status; ncu --metrics gpu__time_duration.sum --csv /tmp/ncu_trivial 2>&1 | grep -E 'ERR_|gpu__time|PROF'; true; } | tee "$OUT/env/ncu_probe.txt"
}
stage ncu_probe "$GPU" ncu_probe
stage memread "$GPU" bash -c "build/bench_memread > '$OUT/g3/memread.csv'"
stage tc_probe "$GPU" bash -c "build/tc_probe > '$OUT/g3/tc_probe.txt'"
stage cublas_probe "$GPU" bash -c "python3 scripts/bench/cublas_probe.py > '$OUT/g3/cublas_probe.txt'"

# G1: differential correctness. The PyTorch runs are reused when already present.
stage g1 "$GPU" python3 tests/g1_real.py --model "$M" --cli "$ROOT/build/stride_cli" --work "$WORK" --report "$OUT/g1"
cp "$WORK/prompts.jsonl" "$OUT/g1/prompts.jsonl" 2>/dev/null

# Exact path, GPU and x86 legs. The pool is capped because a cgroup quota can sit far below the host's thread count.
cross_device() {
  local x="$OUT/g1/xdev"; mkdir -p "$x/gpu" "$x/x86"
  head -n 8 "$WORK/prompts.jsonl" > "$x/prompts.jsonl"
  local flags=(--path exact --model "$M" --input "$x/prompts.jsonl" --dump-steps 32 --max-new 32 --full --concurrency 1 --kv-gb 1)
  build/stride_cli "${flags[@]}" --output "$x/gpu/out.jsonl" --dump-logits "$x/gpu/logits.bin" 2> "$x/gpu/stderr.txt" &&
  STRIDE_CPU_THREADS=${STRIDE_CPU_THREADS:-48} build-cpu/stride_cli "${flags[@]}" --output "$x/x86/out.jsonl" --dump-logits "$x/x86/logits.bin" 2> "$x/x86/stderr.txt" &&
  for d in gpu x86; do tail -1 "$x/$d/stderr.txt" > "$x/$d/summary.json"; done
  grep -m1 "model name" /proc/cpuinfo | cut -d: -f2 | sed 's/^ //' > "$x/x86/host.txt"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "$x/gpu/host.txt"
  cmp "$x/gpu/logits.bin" "$x/x86/logits.bin"
}
stage cross_device "$GPU and CPU" cross_device

# G3: batch-1 roofline, then Nsight Systems traces.
head -n 1 "$WORK/prompts.jsonl" > /tmp/one_prompt.jsonl
stage roofline "$GPU" python3 scripts/bench/roofline.py --cli build/stride_cli --model "$M" --prompt /tmp/one_prompt.jsonl --report "$OUT/g3/roofline.json"
nsys_b1() {
  for path in tc exact; do
    nsys profile --trace=cuda --cuda-memory-usage=false --sample=none --cpuctxsw=none -o "$OUT/g3/batch1_$path" --force-overwrite=true build/stride_cli --path $path --model "$M" --input /tmp/one_prompt.jsonl --output /tmp/o.jsonl --max-new 256 --kv-gb 2 &&
    nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv --output "$OUT/g3/batch1_${path}_kernels" "$OUT/g3/batch1_$path.nsys-rep"
    rm -f "$OUT/g3/batch1_$path.sqlite"
  done
}
stage nsys_b1 "$GPU" nsys_b1
stage nsys_server "$GPU" python3 scripts/bench/trace_server.py --model "$M" --out "$OUT/g3/server_c64" "${BENCH[@]}"

# G2 and the load study: the same requests alone and under load, on both engines. The
# small-arena run forces preemption and replay in the server.
stage det_stride "$GPU" python3 scripts/bench/determinism.py --engine stride --model "$M" --report "$OUT/g2/determinism_stride.json" --port 8125 --levels 4,16,64,128 "${BENCH[@]}"
stage det_vllm "$GPU" python3 scripts/bench/determinism.py --engine vllm --model "$M" --report "$OUT/g2/determinism_vllm.json" --port 8126 --levels 4,16,64,128
stage det_preempt "$GPU" python3 scripts/bench/determinism.py --engine stride --model "$M" --report "$OUT/g2/determinism_stride_preempt.json" --port 8132 --levels 64,128 --kv-gb 0.2 "${BENCH[@]}"

# G3: AIPerf sweep on both engines, same instance, same settings.
stage sweep_stride "$GPU" python3 scripts/bench/sweep.py --engine stride --model "$M" --out "$OUT/g3/aiperf" --port 8127 "${BENCH[@]}"
stage sweep_vllm "$GPU" python3 scripts/bench/sweep.py --engine vllm --model "$M" --out "$OUT/g3/aiperf" --port 8128

# Raw engines with no HTTP in the path, and the exact path at the same batch sizes.
stage offline "$GPU" python3 scripts/bench/offline.py --cli build/stride_cli --model "$M" --report "$OUT/g3/offline.json"
stage exact_scaling "$GPU" python3 scripts/bench/exact_scaling.py --cli build/stride_cli --model "$M" --report "$OUT/g3/exact_scaling.json"

# G4: register-blocked microkernel A/B, plus the per-lane norm, both bit-identical by construction.
stage ab_micro_off "$GPU" python3 scripts/bench/sweep.py --engine stride --label micro_1x1 --model "$M" --out "$OUT/g4" --concurrency 64,128 --port 8129 "${BENCH[@]}" --stride-flag=--micro-r --stride-flag=1 --stride-flag=--micro-lt --stride-flag=1
stage ab_micro_on "$GPU" python3 scripts/bench/sweep.py --engine stride --label micro_2x2 --model "$M" --out "$OUT/g4" --concurrency 64,128 --port 8130 "${BENCH[@]}" --stride-flag=--micro-r --stride-flag=2 --stride-flag=--micro-lt --stride-flag=2
stage ab_norm_tile "$GPU" python3 scripts/bench/sweep.py --engine stride --label norm_tile --model "$M" --out "$OUT/g4" --concurrency 64,128 --port 8131 "${BENCH[@]}" --stride-flag=--norm-tile
ab_bits() {
  local flags=(--path tc --model "$M" --input "$WORK/prompts.jsonl" --max-new 64 --kv-gb 8 --concurrency 64 --max-rows 2048 --prefill-chunk 256)
  build/stride_cli "${flags[@]}" --output "$OUT/g4/bits_auto.jsonl" &&
  build/stride_cli "${flags[@]}" --output "$OUT/g4/bits_1x1.jsonl" --micro-r 1 --micro-lt 1 &&
  build/stride_cli "${flags[@]}" --output "$OUT/g4/bits_normtile.jsonl" --norm-tile &&
  cmp "$OUT/g4/bits_auto.jsonl" "$OUT/g4/bits_1x1.jsonl" && cmp "$OUT/g4/bits_auto.jsonl" "$OUT/g4/bits_normtile.jsonl"
}
stage ab_bits "$GPU" ab_bits

# G5: bug injection on real weights, both paths, and on a small model on the CPU, as in CI.
mut_ref() {
  head -n 8 "$WORK/prompts.jsonl" > /tmp/mut_prompts.jsonl
  python3 tests/reference.py --model "$M" --input /tmp/mut_prompts.jsonl --output /tmp/mut_ref.jsonl --dump-logits /tmp/mut_ref.bin --dump-steps 4 --max-new 32 --dtype float32 --device cuda &&
  python3 tests/reference.py --model "$M" --input /tmp/mut_prompts.jsonl --output /tmp/mut_ref1.jsonl --dump-logits /tmp/mut_ref1.bin --dump-steps 1 --max-new 1 --dtype float32 --device cuda
}
stage mut_ref "$GPU" mut_ref
stage mut_exact "$GPU" python3 tests/mutations.py --cli build-mut/stride_cli --model "$M" --reference /tmp/mut_ref.jsonl --reference-logits /tmp/mut_ref.bin --prompts /tmp/mut_prompts.jsonl --path exact --tolerance 2e-4 --kv-gb 2 --tight-blocks 16 --timeout 300 --report "$OUT/g5/mutation_real_exact.json"
stage mut_tc "$GPU" python3 tests/mutations.py --cli build-mut/stride_cli --model "$M" --reference /tmp/mut_ref1.jsonl --reference-logits /tmp/mut_ref1.bin --prompts /tmp/mut_prompts.jsonl --path tc --no-token-oracle --dump-steps 1 --tolerance 0.6 --kv-gb 2 --tight-blocks 16 --timeout 300 --report "$OUT/g5/mutation_real_tc.json"
mut_cpu() {
  local t=/tmp/stride_tiny; rm -rf "$t"; mkdir -p "$t"
  python3 tests/make_tiny_model.py --out "$t" &&
  python3 - "$t" <<'PY' &&
import json, random, sys
random.seed(7)
with open(sys.argv[1] + "/prompts.jsonl", "w") as f:
    for i in range(24):
        f.write(json.dumps({"id": i, "tokens": [random.randint(3, 1023) for _ in range(random.randint(1, 40))]}) + "\n")
PY
  python3 tests/reference.py --model "$t" --input "$t/prompts.jsonl" --output "$t/ref.jsonl" --dump-logits "$t/ref.bin" --dump-steps 4 --max-new 32 &&
  STRIDE_CPU_THREADS=${STRIDE_CPU_THREADS:-48} python3 tests/mutations.py --cli build-cpu-mut/stride_cli --model "$t" --reference "$t/ref.jsonl" --reference-logits "$t/ref.bin" --prompts "$t/prompts.jsonl" --tolerance 2e-4 --timeout 120 --report "$OUT/g5/mutation_cpu_tiny.json" --min-catch-rate 100
}
stage mut_cpu "CPU" mut_cpu

python3 scripts/redact_nsys.py "$OUT"/g3/*.nsys-rep "$OUT"/g3/server_c64/*.nsys-rep
python3 scripts/scan_secrets.py "$OUT" || echo "SECRET FOUND IN $OUT: do not commit these reports"
echo "[$(date -u +%H:%M:%S)] all stages finished"
