# stride

A C++/CUDA LLM inference engine whose greedy outputs did not change under load in any test here. Serves Llama-3.2-1B on an RTX 4090 with paged KV caching, continuous batching and an OpenAI-compatible streaming API, on original kernels with no PyTorch or vLLM in the runtime path.

At concurrency 128, vLLM returned different text for 85.0% of temperature-0 requests, compared with the same requests sent one at a time. stride changed 0 of 968 across all load levels, and it reaches 87.5% of vLLM's serving throughput at concurrency 64.

## Results

One RTX 4090 instance, driver 595.91.07, and one pinned set of weight files for every engine. Every number comes from [the ledger](docs/resume/METRICS.jsonl), which records its artifact, commit and device.

**Same request, alone and under load.** 100 prompts of 43 to 51 tokens, 256 output tokens, temperature 0, seed 0, EOS ignored, prefix caching off. Each response is compared with that request's own sequential output.

| Concurrency | stride: requests that changed | vLLM 0.32.0: requests that changed |
| --- | --- | --- |
| 4 | **0 / 100** | 82 / 100 (82.0%) |
| 16 | **0 / 100** | 84 / 100 (84.0%) |
| 64 | **0 / 256** | 225 / 256 (87.9%) |
| 128 | **0 / 512** | 435 / 512 (85.0%) |

Run sequentially twice, stride changed 0 and vLLM 0 of 100 outputs, so these vLLM differences appear only when requests share the engine. That is consistent with batch-dependent numerics as [analyzed by Thinking Machines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/), though this test does not isolate the cause, and it is not a claim that vLLM has a bug. One flipped token changes everything after it, so a 256-token output is a sensitive test.

stride's `system_fingerprint`, which covers the exact bits of every chosen logit, also matched on every request. A separate run with a KV arena small enough to force 670 preemptions and replays changed 0 of 768 requests. Per-request hashes are in [`reports/g2`](reports/g2).

**HTTP serving throughput and latency.** NVIDIA AIPerf 0.13.0 on the OpenAI chat endpoint, streaming, mean of 3 repeats. AIPerf requested 256 ± 32 prompt tokens, about 290 on the wire after the chat template, and 256 output tokens. vLLM ran on its default scheduler settings; stride's were chosen on these shapes.

| Concurrency | stride tok/s | vLLM tok/s | stride / vLLM | TTFT p50 ms, stride / vLLM | Inter-token p50 ms, stride / vLLM | Request p99 ms, stride / vLLM |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 302 ± 0.3 | 316 ± 0.8 | 95.5% | 13 / 19 | 3.25 / 3.07 | 849 / 815 |
| 4 | 1,162 ± 1.5 | 1,121 ± 0.5 | 103.6% | 33 / 36 | 3.29 / 3.40 | 890 / 926 |
| 16 | 4,026 ± 6.4 | 3,907 ± 9.4 | 103.1% | 120 / 91 | 3.47 / 3.72 | 1,024 / 1,051 |
| 64 | 9,540 ± 11.8 | 10,907 ± 22.2 | 87.5% | 215 / 219 | 5.83 / 4.98 | 2,012 / 1,531 |
| 128 | 11,814 ± 8.2 | 14,947 ± 39.7 | 79.0% | 222 / 403 | 9.96 / 6.96 | 3,562 / 2,209 |

AIPerf reported 0 errors at every point, on both engines.

**Raw engines, no HTTP.** The serving comparison includes each server's overhead; at large batches vLLM's raw engine is further ahead. Same token-id prompts (256 tokens), 256 output tokens, mean of 3 repeats.

| Batch | stride tok/s | vLLM tok/s | stride / vLLM |
| --- | --- | --- | --- |
| 1 | 306 | 319 | 95.6% |
| 8 | 2,276 | 2,181 | 104.4% |
| 64 | 10,061 | 11,822 | 85.1% |
| 128 | 12,622 | 16,730 | 75.4% |

**Correctness.** 100 prompts × 256 tokens against HF transformers on the same files.

| Comparison | Greedy token match | Identical sequences |
| --- | --- | --- |
| stride exact path vs HF, FP32 activations | **100.000%** | 100 / 100 |
| stride serving path vs HF, FP32 activations | 75.73% | 52 / 100 |
| HF BF16 vs HF FP32, the noise floor of BF16 itself | 38.76% | 6 / 100 |
| stride exact path, batch 1 vs batch 32 with shuffled lanes and pages | 100.000% | 100 / 100 |
| stride serving path, batch 1 vs batch 32 with shuffled lanes and pages | 100.000% | 100 / 100 |

Exact-path logits stay within 5.72e-05 of the FP32 reference across 400 vectors.

**Same bytes on different hardware.** On real Llama weights, the exact path produced byte-identical logits on an RTX 4090, an AMD EPYC 7702 and an Apple M1 Pro: 32,833,536 FP32 values from 8 prompts × 32 steps, with one SHA-256. The serving path does not reproduce on a CPU, but a second RTX 4090 in another data center matched on 100 of 100 prompts, by fingerprint and tokens over 25,600 generated tokens ([host details](reports/g1/host2)).

**Other measurements**

| Metric | Value |
| --- | --- |
| Batch-1 decode, serving path | 310.1 ± 0.3 tok/s, **76.0%** of a 407.8 tok/s roofline |
| Batch-1 decode, exact path | 285.0 ± 0.1 tok/s, 69.9% of the roofline |
| Register-blocked tensor-core kernels (A/B, 2×2 vs 1×1, identical output bits) | Concurrency 128: inter-token p50 cut 33.8%, throughput up 50.2%. Concurrency 64: 16.4% and 25.4%. No effect below 49 rows, where 2×2 is not used |
| Per-lane RMSNorm (A/B, identical output bits) | Throughput up 19.9% at concurrency 64, 11.3% at 128 |
| GPU idle between steps at concurrency 64, under Nsight Systems | 106 µs of 5,884 µs per step |
| GPU memory peak, stride / vLLM | 19,938 / 21,068 MiB: each engine's configured allocation, not its minimum |

**Bug injection.** 29 injected bugs in kernels, KV cache and scheduler, of which 25 change behaviour and 4 change only rounding.

| Run | Caught | Functional bugs caught | Functional bugs caught by a tolerance check alone |
| --- | --- | --- | --- |
| Real weights, exact path | 29 / 29 | 25 / 25 | 19 |
| Real weights, serving path | 29 / 29 | 25 / 25 | 12 |
| Small random model on CPU, as in CI | 29 / 29 | 25 / 25 | 19 |

The tolerance check runs one request at a time on the full-logits path, so it cannot reach bugs in the fused argmax or in concurrent scheduling; the invariance and golden-bits oracles reach those. On the serving path its threshold is wide (±0.6 on first-step logits, no token check) because of BF16 noise, so it misses more. The rounding-only bugs show the golden oracle is sensitive, not that it finds functional bugs.

## How it works

For any output element, the operations and their order depend only on the matrix shape, never on the batch, the lane, the thread or where KV pages live. Dot products use fixed summation trees, `exp` is hand-written from FMA steps, attention sums pages in position order, and argmax is order-independent. An optimization that only moves work between warps therefore should not change a bit, and the tests check that it does not.

Two kernel families share one scheduler and server:

- **Serving path** (`--path tc`, default on CUDA): tensor-core `mma` tiles with BF16 activations. Fast and stable under load in these tests, but not reproducible on a CPU, and it differs from an FP32 reference as all BF16 inference does.
- **Exact path** (`--path exact`): scalar FMA chains with FP32 activations. Matches PyTorch to FP32 rounding and produces the same bytes on GPU and CPU, but it is much slower at large batches.

Responses carry `system_fingerprint: stride-<hash>`, chained over every token and the exact bits of its logit, so a client can check that load changed nothing.

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Choices and the measurements behind them: [docs/DECISIONS.md](docs/DECISIONS.md).

## Methodology

- **Weights:** `unsloth/Llama-3.2-1B-Instruct` at commit `5a8abab4a5d6f164389b1079fb721cfab8d7126c`. The engine, the PyTorch reference and vLLM load one directory, which [`scripts/verify_weights.py`](scripts/verify_weights.py) checks against [`weights.lock.json`](weights.lock.json) before each run.
- **Fairness:** same instance and driver, though not the same CUDA libraries: stride builds with toolkit 12.8.93, and vLLM runs on PyTorch 2.13.0+cu130. Prefix caching is off in vLLM and absent from stride. Warmup is a separate AIPerf run with a different seed, so its prompts differ from measured prompts (AIPerf issue #1656), and it is excluded. 3 repeats per point.
- **Roofline:** 1,008 GB/s, the RTX 4090's specified peak, divided by 2,471,628,800 weight bytes per forward pass from the safetensors header. It is a bound from a data-sheet figure, not a counter measurement, and KV and activation traffic are not counted.
- **Reference:** HF transformers with BF16 weights and FP32 activations. The noise-floor row above shows why BF16 activations cannot serve as a precise reference.
- **Environment:** RunPod Secure Cloud, data center EUR-IS-1, NVIDIA GeForce RTX 4090, driver 595.91.07, AMD EPYC 7702 64-Core Processor. See [`reports/env`](reports/env).

| Weight file | SHA-256 |
| --- | --- |
| `model.safetensors` | `1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f` |
| `config.json` | `dfb67fd8afe73a1c75245824ef9d64a6ba8983025447e3bf76aa1ea57ee46152` |
| `generation_config.json` | `6d4f979915331212d7672c68b22a4ddad9e21ed8126cf2bd1ea6b2b88f595c1c` |
| `tokenizer.json` | `6b9e4e7fb171f92fd137b777cc2714bf87d11576700a1dcd7a399e7bbe39537b` |
| `tokenizer_config.json` | `9ddd255c19fe319c8d4e891163540382e9fbda99f394674f2a929efc47d57458` |
| `special_tokens_map.json` | `94e708c3f5e64acf85bbe5ad01467a1248faadb73e83b41793087ecced586e8f` |
| `chat_template.jinja` | `5816fce10444e03c2e9ee1ef8a4a1ea61ae7e69e438613f3b17b69d0426223a4` |

## Limitations

- **Misses against the targets.** Batch-1 decode is 76.0% of the roofline against a target of 85%. Serving throughput at concurrency 64 is 87.5% of vLLM, past the 70% target but short of the 90% stretch. At concurrency 128 it is 79.0%, and request p99 latency is worse than vLLM's at both of those levels.
- **The serving path does not match FP32 token for token**, at 75.73%. It is closer than HF's own BF16 mode at 38.76%, but the 100% result belongs to the exact path, which is too slow at large batches to be the serving default.
- **What the stability result covers.** Greedy decoding, 100 distinct short prompts, one server configuration, concurrency up to 128, this model, and two RTX 4090s on one driver. A different GPU architecture may round differently inside `mma`, which has not been tested. The exact-path byte comparison covers 8 prompts × 32 steps.
- **The weights come from a mirror.** The upstream `meta-llama` repo is gated and masks its file hash, so byte identity with it is not verified. `model.safetensors` has the same size as upstream's. `config.json` is the mirror's own edited version.
- **Nsight Compute reports are missing.** `ncu` fails with `ERR_NVGPUCTRPERM` on the RunPod hosts tried ([one](reports/env/ncu_probe.txt), [two](reports/g1/host2/ncu_probe.txt)), because GPU counters need admin rights the containers lack. Timing comes from Nsight Systems.
- **Nsight Systems reports are redacted.** They embed the container's environment, so secret values are overwritten by [`scripts/redact_nsys.py`](scripts/redact_nsys.py). The scan covers raw bytes, not any compressed sections.
- **Not implemented:** prefix caching, multi-GPU, logprobs and tools. `min_tokens` is accepted only together with `ignore_eos`. Llama-3.1-8B was not attempted. Maximum context is 2048 tokens.

## Reproduce

```bash
# Fetch and check the pinned snapshot.
python3 scripts/pin_weights.py --dest models/llama32-1b --lock /tmp/lock.json --commit 5a8abab4a5d6f164389b1079fb721cfab8d7126c
python3 scripts/verify_weights.py --dir models/llama32-1b

# Build and serve on a CUDA machine.
cmake -S . -B build -G Ninja -DSTRIDE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build
build/stride_server --model models/llama32-1b --port 8000 --kv-gb 16 --max-seqs 256 --max-rows 2048 --prefill-chunk 256

# CPU tests, as in CI. No GPU or gated weights needed.
cmake -S . -B build-cpu -G Ninja -DSTRIDE_SERVER=OFF
cmake --build build-cpu
pytest tests/test_correctness.py
```

To regenerate the evidence on one RTX 4090, with vLLM and AIPerf installed in `BENCH_VENV` and PyTorch and transformers in the system Python:

```bash
OUT_DIR=$PWD/reports MODEL_DIR=$PWD/models/llama32-1b BENCH_VENV=/path/to/venv scripts/run_all.sh $(git rev-parse HEAD)
python3 scripts/build_ledger.py && python3 scripts/render_docs.py
```

Not produced by `run_all.sh`: the ARM leg of the cross-device comparison ([`scripts/cross_device.py`](scripts/cross_device.py), needs an ARM machine), the second-host run ([`scripts/cross_host.sh`](scripts/cross_host.sh), needs a second GPU host), and the hand-collected design experiments in [`reports/g3/tuning_log.md`](reports/g3/tuning_log.md), which are not used for any reported result. `run_all.sh` assumes a RunPod-style `/usr/local/cuda` toolkit.

[ATTRIBUTION.md](ATTRIBUTION.md) lists libraries and prior work. Built with Llama. Code is Apache-2.0; the model is under the Llama 3.2 Community License.
