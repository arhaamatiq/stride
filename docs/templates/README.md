# stride

A C++/CUDA LLM inference engine whose greedy outputs did not change under load in any test here. Serves Llama-3.2-1B on an RTX 4090 with paged KV caching, continuous batching and an OpenAI-compatible streaming API, on original kernels with no PyTorch or vLLM in the runtime path.

At concurrency 128, vLLM returned different text for {{g2.determinism.vllm.c128.requests_differing_pct|.2f}}% of temperature-0 requests, compared with the same requests sent one at a time. stride changed {{g2.determinism.stride.total_requests_differing}} of {{g2.determinism.stride.total_requests}} across all load levels, and it reaches {{g3.aiperf.pct_of_vllm.c64|.1f}}% of vLLM's serving throughput at concurrency 64.

## Results

One RTX 4090 host, driver {{env.driver}}, and one pinned set of weight files for every engine comparison. Every number comes from [the ledger](docs/resume/METRICS.jsonl), which records its artifact, commit and device.

**Same request, alone and under load.** {{g1.prompt_tokens.count}} prompts of {{g1.prompt_tokens.min}} to {{g1.prompt_tokens.max}} tokens, 256 output tokens, temperature 0, seed 0, EOS ignored, prefix caching off. Each response is compared with that request's own sequential output.

| Concurrency | stride: requests that changed | vLLM {{env.vllm}}: requests that changed |
| --- | --- | --- |
| 4 | **{{g2.determinism.stride.c4.requests_differing}} / {{g2.determinism.stride.c4.requests}}** | {{g2.determinism.vllm.c4.requests_differing}} / {{g2.determinism.vllm.c4.requests}} ({{g2.determinism.vllm.c4.requests_differing_pct|.2f}}%) |
| 16 | **{{g2.determinism.stride.c16.requests_differing}} / {{g2.determinism.stride.c16.requests}}** | {{g2.determinism.vllm.c16.requests_differing}} / {{g2.determinism.vllm.c16.requests}} ({{g2.determinism.vllm.c16.requests_differing_pct|.2f}}%) |
| 64 | **{{g2.determinism.stride.c64.requests_differing}} / {{g2.determinism.stride.c64.requests}}** | {{g2.determinism.vllm.c64.requests_differing}} / {{g2.determinism.vllm.c64.requests}} ({{g2.determinism.vllm.c64.requests_differing_pct|.2f}}%) |
| 128 | **{{g2.determinism.stride.c128.requests_differing}} / {{g2.determinism.stride.c128.requests}}** | {{g2.determinism.vllm.c128.requests_differing}} / {{g2.determinism.vllm.c128.requests}} ({{g2.determinism.vllm.c128.requests_differing_pct|.2f}}%) |

Run sequentially twice, stride changed {{g2.determinism.stride.sequential_rerun_differs}} and vLLM {{g2.determinism.vllm.sequential_rerun_differs}} of 100 outputs, so these vLLM differences appear only when requests share the engine. That is consistent with batch-dependent numerics as [analyzed by Thinking Machines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/), though this test does not isolate the cause, and it is not a claim that vLLM has a bug. One flipped token changes everything after it, so a 256-token output is a sensitive test.

stride's `system_fingerprint`, which covers the exact bits of every chosen logit, also matched on every request. A separate run with a KV arena small enough to force {{g2.determinism.stride_preempt.c128.preemptions_so_far}} preemptions, replaying {{g2.determinism.stride_preempt.replayed_rows_recomputed|,}} rows, changed {{g2.determinism.stride_preempt.total_requests_differing}} of {{g2.determinism.stride_preempt.total_requests}} requests. Per-request hashes are in [`reports/g2`](reports/g2).

**HTTP serving throughput and latency.** NVIDIA AIPerf {{env.aiperf}} on the OpenAI chat endpoint, streaming, mean of 3 repeats. AIPerf requested {{g3.aiperf.stride.isl_requested}} ± {{g3.aiperf.stride.isl_stddev}} prompt tokens, about {{g3.aiperf.stride.c64.prompt_tokens_on_wire|.0f}} on the wire after the chat template, and {{g3.aiperf.stride.osl}} output tokens. vLLM ran on its default scheduler settings; stride's were chosen on these shapes.

| Concurrency | stride tok/s | vLLM tok/s | stride / vLLM | TTFT p50 ms, stride / vLLM | Inter-token p50 ms, stride / vLLM | Request p99 ms, stride / vLLM |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | {{g3.aiperf.stride.c1.output_tok_per_s|,.0f}} ± {{g3.aiperf.stride.c1.output_tok_per_s_std|,.1f}} | {{g3.aiperf.vllm.c1.output_tok_per_s|,.0f}} ± {{g3.aiperf.vllm.c1.output_tok_per_s_std|,.1f}} | {{g3.aiperf.pct_of_vllm.c1|.1f}}% | {{g3.aiperf.stride.c1.ttft_p50_ms|.0f}} / {{g3.aiperf.vllm.c1.ttft_p50_ms|.0f}} | {{g3.aiperf.stride.c1.itl_p50_ms|.2f}} / {{g3.aiperf.vllm.c1.itl_p50_ms|.2f}} | {{g3.aiperf.stride.c1.request_latency_p99_ms|,.0f}} / {{g3.aiperf.vllm.c1.request_latency_p99_ms|,.0f}} |
| 4 | {{g3.aiperf.stride.c4.output_tok_per_s|,.0f}} ± {{g3.aiperf.stride.c4.output_tok_per_s_std|,.1f}} | {{g3.aiperf.vllm.c4.output_tok_per_s|,.0f}} ± {{g3.aiperf.vllm.c4.output_tok_per_s_std|,.1f}} | {{g3.aiperf.pct_of_vllm.c4|.1f}}% | {{g3.aiperf.stride.c4.ttft_p50_ms|.0f}} / {{g3.aiperf.vllm.c4.ttft_p50_ms|.0f}} | {{g3.aiperf.stride.c4.itl_p50_ms|.2f}} / {{g3.aiperf.vllm.c4.itl_p50_ms|.2f}} | {{g3.aiperf.stride.c4.request_latency_p99_ms|,.0f}} / {{g3.aiperf.vllm.c4.request_latency_p99_ms|,.0f}} |
| 16 | {{g3.aiperf.stride.c16.output_tok_per_s|,.0f}} ± {{g3.aiperf.stride.c16.output_tok_per_s_std|,.1f}} | {{g3.aiperf.vllm.c16.output_tok_per_s|,.0f}} ± {{g3.aiperf.vllm.c16.output_tok_per_s_std|,.1f}} | {{g3.aiperf.pct_of_vllm.c16|.1f}}% | {{g3.aiperf.stride.c16.ttft_p50_ms|.0f}} / {{g3.aiperf.vllm.c16.ttft_p50_ms|.0f}} | {{g3.aiperf.stride.c16.itl_p50_ms|.2f}} / {{g3.aiperf.vllm.c16.itl_p50_ms|.2f}} | {{g3.aiperf.stride.c16.request_latency_p99_ms|,.0f}} / {{g3.aiperf.vllm.c16.request_latency_p99_ms|,.0f}} |
| 64 | {{g3.aiperf.stride.c64.output_tok_per_s|,.0f}} ± {{g3.aiperf.stride.c64.output_tok_per_s_std|,.1f}} | {{g3.aiperf.vllm.c64.output_tok_per_s|,.0f}} ± {{g3.aiperf.vllm.c64.output_tok_per_s_std|,.1f}} | {{g3.aiperf.pct_of_vllm.c64|.1f}}% | {{g3.aiperf.stride.c64.ttft_p50_ms|.0f}} / {{g3.aiperf.vllm.c64.ttft_p50_ms|.0f}} | {{g3.aiperf.stride.c64.itl_p50_ms|.2f}} / {{g3.aiperf.vllm.c64.itl_p50_ms|.2f}} | {{g3.aiperf.stride.c64.request_latency_p99_ms|,.0f}} / {{g3.aiperf.vllm.c64.request_latency_p99_ms|,.0f}} |
| 128 | {{g3.aiperf.stride.c128.output_tok_per_s|,.0f}} ± {{g3.aiperf.stride.c128.output_tok_per_s_std|,.1f}} | {{g3.aiperf.vllm.c128.output_tok_per_s|,.0f}} ± {{g3.aiperf.vllm.c128.output_tok_per_s_std|,.1f}} | {{g3.aiperf.pct_of_vllm.c128|.1f}}% | {{g3.aiperf.stride.c128.ttft_p50_ms|.0f}} / {{g3.aiperf.vllm.c128.ttft_p50_ms|.0f}} | {{g3.aiperf.stride.c128.itl_p50_ms|.2f}} / {{g3.aiperf.vllm.c128.itl_p50_ms|.2f}} | {{g3.aiperf.stride.c128.request_latency_p99_ms|,.0f}} / {{g3.aiperf.vllm.c128.request_latency_p99_ms|,.0f}} |

AIPerf reported {{g3.aiperf.total_errors}} errors across all points on both engines.

**Engines without HTTP.** The serving comparison includes each server's overhead; at large batches vLLM's engine is further ahead without it. `stride_cli` against vLLM's `LLM` class on the same token-id prompts ({{g3.offline.prompt_tokens}} tokens), 256 output tokens, mean of {{g3.offline.repeats}} repeats. vLLM's time still includes its Python front end and default detokenization; `stride_cli` links no tokenizer.

| Batch | stride tok/s | vLLM tok/s | stride / vLLM |
| --- | --- | --- | --- |
| 1 | {{g3.offline.b1.stride_tok_per_s|,.0f}} | {{g3.offline.b1.vllm_tok_per_s|,.0f}} | {{g3.offline.b1.pct_of_vllm|.1f}}% |
| 8 | {{g3.offline.b8.stride_tok_per_s|,.0f}} | {{g3.offline.b8.vllm_tok_per_s|,.0f}} | {{g3.offline.b8.pct_of_vllm|.1f}}% |
| 64 | {{g3.offline.b64.stride_tok_per_s|,.0f}} | {{g3.offline.b64.vllm_tok_per_s|,.0f}} | {{g3.offline.b64.pct_of_vllm|.1f}}% |
| 128 | {{g3.offline.b128.stride_tok_per_s|,.0f}} | {{g3.offline.b128.vllm_tok_per_s|,.0f}} | {{g3.offline.b128.pct_of_vllm|.1f}}% |

**Correctness.** {{g1.prompt_tokens.count}} prompts × 256 tokens against HF transformers on the same files.

| Comparison | Greedy token match | Identical sequences |
| --- | --- | --- |
| stride exact path vs HF, FP32 activations | **{{g1.greedy.stride_exact_vs_hf_fp32.pct|.3f}}%** | {{g1.greedy.stride_exact_vs_hf_fp32.identical_sequences}} / 100 |
| stride serving path vs HF, FP32 activations | {{g1.greedy.stride_tc_vs_hf_fp32.pct|.2f}}% | {{g1.greedy.stride_tc_vs_hf_fp32.identical_sequences}} / 100 |
| HF BF16 vs HF FP32: how far PyTorch's BF16 mode drifts | {{g1.greedy.hf_bf16_vs_hf_fp32_noise_floor.pct|.2f}}% | {{g1.greedy.hf_bf16_vs_hf_fp32_noise_floor.identical_sequences}} / 100 |
| stride exact path, batch 1 vs batch 32 with shuffled lanes and pages | {{g1.greedy.stride_exact_c1_vs_c32_shuffled.pct|.3f}}% | {{g1.greedy.stride_exact_c1_vs_c32_shuffled.identical_sequences}} / 100 |
| stride serving path, batch 1 vs batch 32 with shuffled lanes and pages | {{g1.greedy.stride_tc_c1_vs_c32_shuffled.pct|.3f}}% | {{g1.greedy.stride_tc_c1_vs_c32_shuffled.identical_sequences}} / 100 |

Token match is position-wise on free-running sequences, so one early flip counts every later token as a miss. Exact-path logits stay within {{g1.logits.stride_exact_vs_hf_fp32.max_abs_diff|.2e}} of the FP32 reference across {{g1.logits.stride_exact_vs_hf_fp32.vectors}} vectors.

**Same bytes on different hardware.** On real Llama weights, the exact path produced byte-identical logits on an RTX 4090, an AMD EPYC 7702 and an Apple M1 Pro: {{g1.cross_device.logit_values|,}} FP32 values from 8 prompts × 32 steps, with one SHA-256. The serving path runs only on a GPU, but a second RTX 4090 in another data center matched on {{g1.cross_host.tc.identical}} of 100 prompts at concurrency 1, by fingerprint and tokens over 25,600 generated tokens ([host details](reports/g1/host2)).

**Other measurements**

| Metric | Value |
| --- | --- |
| Batch-1 decode, serving path | {{g3.batch1.tc.tok_per_s|.1f}} ± {{g3.batch1.tc.tok_per_s_std|.1f}} tok/s, **{{g3.batch1.tc.pct_of_roofline|.1f}}%** of a {{g3.roofline.tok_per_s|.1f}} tok/s roofline |
| Batch-1 decode, exact path | {{g3.batch1.exact.tok_per_s|.1f}} ± {{g3.batch1.exact.tok_per_s_std|.1f}} tok/s, {{g3.batch1.exact.pct_of_roofline|.1f}}% of the roofline |
| Register-blocked tensor-core kernels (A/B, 2×2 vs 1×1, identical output bits) | Concurrency 128: inter-token p50 cut {{g4.micro.c128.itl_p50_cut_pct|.1f}}%, throughput up {{g4.micro.c128.throughput_gain_pct|.1f}}%. Concurrency 64: {{g4.micro.c64.itl_p50_cut_pct|.1f}}% and {{g4.micro.c64.throughput_gain_pct|.1f}}%. Below 49 rows per step the automatic rule selects 1×1; not A/B tested below concurrency 64 |
| Per-lane RMSNorm (A/B, identical output bits) | Throughput up {{g4.norm_lane.c64.throughput_gain_pct|.1f}}% at concurrency 64, with a noisy baseline (std {{g4.norm_lane.c64.baseline_std|.0f}} tok/s); up {{g4.norm_lane.c128.throughput_gain_pct|.1f}}% at 128 |
| Exact path against serving path, no HTTP | Batch 64: {{g3.exact.b64.tok_per_s|,.0f}} against {{g3.offline.b64.stride_tok_per_s|,.0f}} tok/s. Batch 128: {{g3.exact.b128.tok_per_s|,.0f}} against {{g3.offline.b128.stride_tok_per_s|,.0f}} tok/s |
| GPU idle between steps at concurrency 64, under Nsight Systems | {{g3.trace.c64.between_steps_us|.0f}} µs of {{g3.trace.c64.wall_us_per_step|,.0f}} µs per step |
| GPU memory peak, stride / vLLM | {{g3.aiperf.stride.gpu_memory_peak_mib|,}} / {{g3.aiperf.vllm.gpu_memory_peak_mib|,}} MiB: each engine's configured allocation, not its minimum |

**Bug injection.** {{g5.real_exact.mutants}} injected bugs in kernels, KV cache and scheduler, of which {{g5.real_exact.functional_mutants}} change behaviour and {{g5.real_exact.rounding_only_mutants}} change only rounding.

| Run | Caught | Functional bugs caught | Functional bugs caught by a tolerance check alone |
| --- | --- | --- | --- |
| Real weights, exact path | {{g5.real_exact.caught}} / {{g5.real_exact.mutants}} | {{g5.real_exact.functional_caught}} / {{g5.real_exact.functional_mutants}} | {{g5.real_exact.functional_caught_by_tolerance}} |
| Real weights, serving path | {{g5.real_tc.caught}} / {{g5.real_tc.mutants}} | {{g5.real_tc.functional_caught}} / {{g5.real_tc.functional_mutants}} | {{g5.real_tc.functional_caught_by_tolerance}} |
| Small random model on CPU, as in CI | {{g5.cpu_tiny.caught}} / {{g5.cpu_tiny.mutants}} | {{g5.cpu_tiny.functional_caught}} / {{g5.cpu_tiny.functional_mutants}} | {{g5.cpu_tiny.functional_caught_by_tolerance}} |

The exact-path count includes {{g5.real_exact.caught_only_by_crash_or_hang}} bug caught only because it crashed the run. The tolerance check runs one request at a time on the full-logits path, so it cannot reach bugs in the fused argmax or in concurrent scheduling; the invariance and golden-bits oracles reach those. On the serving path its threshold is wide (±0.6 on first-step logits, no token check) because of BF16 noise, so it misses more. The rounding-only bugs show the golden oracle is sensitive, not that it finds functional bugs.

## How it works

For any output element, the operations and their order depend only on the matrix shape, never on the batch, the lane, the thread or where KV pages live. Dot products use fixed summation trees, `exp` is hand-written from FMA steps, attention sums pages in position order, and argmax is order-independent. An optimization that only moves work between warps therefore should not change a bit, and the tests check that it does not. On the exact path this follows from the IEEE operations used. On the serving path it depends on how the tensor cores round, so it is established by testing, not guaranteed.

Two kernel families share one scheduler and server:

- **Serving path** (`--path tc`, default on CUDA): tensor-core `mma` tiles with BF16 activations. Fast and stable under load in these tests, but it runs only on a GPU, and it differs from an FP32 reference as BF16 inference generally does.
- **Exact path** (`--path exact`): scalar FMA chains with FP32 activations. Matches PyTorch to FP32 rounding and produces the same bytes on GPU and CPU, but it is slower at large batches; see the table above.

Responses carry `system_fingerprint: stride-<hash>`, chained over every token and the exact bits of its logit, so a client can check that load changed nothing.

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Choices and the measurements behind them: [docs/DECISIONS.md](docs/DECISIONS.md).

## Methodology

- **Weights:** `{{weights.repo}}` at commit `{{weights.commit}}`. The engine, the PyTorch reference and vLLM load one directory, which [`scripts/verify_weights.py`](scripts/verify_weights.py) checks against [`weights.lock.json`](weights.lock.json) before each run.
- **Runs:** the comparison numbers come from two `run_all.sh` passes on one container, judged by process and cache continuity; the GPU UUID was recorded only in the second. Engine source is identical between the passes. The HF reference runs predate the final G1 stage and are reused.
- **Fairness:** same host and driver, though not the same CUDA libraries: stride builds with toolkit {{env.toolkit}}, and vLLM runs on PyTorch {{env.vllm_torch}}. Prefix caching is off in vLLM and absent from stride. Warmup is a separate AIPerf run with a different seed, so its prompts differ from measured prompts (AIPerf issue #1656), and it is excluded. 3 repeats per point.
- **Roofline:** {{g3.roofline.peak_gb_per_s|,.0f}} GB/s, the RTX 4090's specified peak, divided by {{g3.roofline.weight_bytes_per_pass|,}} weight bytes per forward pass from the safetensors header. It is a bound from a data-sheet figure, not a counter measurement, and KV and activation traffic are not counted.
- **Reference:** HF transformers with BF16 weights and FP32 activations. The HF BF16 row above shows why BF16 activations cannot serve as a precise reference.
- **Environment:** {{env.provider}}, {{env.gpu}}, driver {{env.driver}}, {{env.cpu}}. See [`reports/env`](reports/env).

| Weight file | SHA-256 |
| --- | --- |
| `model.safetensors` | `{{weights.sha256.model.safetensors}}` |
| `config.json` | `{{weights.sha256.config.json}}` |
| `generation_config.json` | `{{weights.sha256.generation_config.json}}` |
| `tokenizer.json` | `{{weights.sha256.tokenizer.json}}` |
| `tokenizer_config.json` | `{{weights.sha256.tokenizer_config.json}}` |
| `special_tokens_map.json` | `{{weights.sha256.special_tokens_map.json}}` |
| `chat_template.jinja` | `{{weights.sha256.chat_template.jinja}}` |

## Limitations

- **Misses against the targets.** Batch-1 decode is {{g3.batch1.tc.pct_of_roofline|.1f}}% of the roofline against a target of 85%. Serving throughput at concurrency 64 is {{g3.aiperf.pct_of_vllm.c64|.1f}}% of vLLM, past the 70% target but short of the 90% stretch. At concurrency 128 it is {{g3.aiperf.pct_of_vllm.c128|.1f}}%, and request p99 latency is worse than vLLM's at both of those levels.
- **The serving path does not match FP32 token for token**, at {{g1.greedy.stride_tc_vs_hf_fp32.pct|.2f}}%. It is closer than HF's own BF16 mode at {{g1.greedy.hf_bf16_vs_hf_fp32_noise_floor.pct|.2f}}%, but the 100% result belongs to the exact path, which is too slow at large batches to be the serving default.
- **What the stability result covers.** Greedy decoding, {{g1.prompt_tokens.count}} distinct short prompts, one server configuration, concurrency up to 128, this model, and one RTX 4090 under load; a second RTX 4090 reproduced the outputs at concurrency 1. A different GPU architecture may round differently inside `mma`, which has not been tested. The exact-path byte comparison covers 8 prompts × 32 steps.
- **The weights come from a mirror.** The upstream `meta-llama` repo is gated and masks its file hash, so byte identity with it is not verified. `model.safetensors` has the same size as upstream's. `config.json` is the mirror's own edited version.
- **Nsight Compute reports are missing.** `ncu` fails with `ERR_NVGPUCTRPERM` on the RunPod hosts tried ([one](reports/env/ncu_probe.txt), [two](reports/g1/host2/ncu_probe.txt)), because GPU counters need admin rights the containers lack. Timing comes from Nsight Systems.
- **Nsight Systems reports are redacted.** They embed the container's environment, so secrets and network details are overwritten by [`scripts/redact_nsys.py`](scripts/redact_nsys.py). The scan covers raw bytes, not any compressed sections.
- **One report field reads wrong.** `replayed_rows` was not wired up when the load study ran, so [`determinism_stride_preempt.json`](reports/g2/determinism_stride_preempt.json) shows 0. The number above is recomputed from that report's own row and token counts; the counter has since been fixed, with no effect on arithmetic.
- **Not implemented:** prefix caching, multi-GPU, logprobs and tools. `min_tokens` is accepted only together with `ignore_eos`. Llama-3.1-8B was not attempted. Maximum context is 2048 tokens.

## Reproduce

```bash
# Fetch and check the pinned snapshot.
python3 scripts/pin_weights.py --dest models/llama32-1b --lock /tmp/lock.json --commit {{weights.commit}}
python3 scripts/verify_weights.py --dir models/llama32-1b

# Build and serve on a CUDA machine.
cmake -S . -B build -G Ninja -DSTRIDE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build
build/stride_server --model models/llama32-1b --port 8000 --kv-gb 16 --max-seqs 256 --max-rows 2048 --prefill-chunk 256

# CPU tests, as in CI. No GPU or gated weights needed.
pip install torch transformers safetensors numpy pytest accelerate
cmake -S . -B build-cpu -G Ninja -DSTRIDE_SERVER=OFF
cmake --build build-cpu
pytest tests/test_correctness.py
```

To regenerate the evidence on one RTX 4090, with vLLM and AIPerf installed in `BENCH_VENV`, PyTorch and transformers in the system Python, and `nvcc`, `nsys` and `ncu` on `PATH`:

```bash
OUT_DIR=$PWD/reports MODEL_DIR=$PWD/models/llama32-1b BENCH_VENV=/path/to/venv scripts/run_all.sh $(git rev-parse HEAD)
scripts/compact_reports.sh reports
python3 scripts/build_ledger.py && python3 scripts/render_docs.py
```

Not produced by `run_all.sh`:

- `reports/g1/cross_device.json` and its ARM leg. `run_all.sh` only compares the GPU and x86 logits; [`scripts/cross_device.py`](scripts/cross_device.py) makes the report.
- The second-host run, from [`scripts/cross_host.sh`](scripts/cross_host.sh).
- `weights.lock.json`, from [`scripts/pin_weights.py`](scripts/pin_weights.py).
- Stage records for those, from [`scripts/record_stage.py`](scripts/record_stage.py).
- [`reports/g3/tuning_log.md`](reports/g3/tuning_log.md), hand-collected design experiments used in `docs/DECISIONS.md` but not for any reported result.

The committed first pass predates per-stage records, so its records were backfilled from `reports/run_all.log`. Reports were reduced as in [`scripts/compact_reports.sh`](scripts/compact_reports.sh): AIPerf per-request exports are compressed, and regenerable inputs, logit dumps and stage markers are removed. `run_all.sh` assumes a `/usr/local/cuda` toolkit.

[ATTRIBUTION.md](ATTRIBUTION.md) lists libraries and prior work. Built with Llama. Code is Apache-2.0; the model is under the Llama 3.2 Community License.
