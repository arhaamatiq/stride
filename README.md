# stride

A C++/CUDA LLM inference engine whose outputs do not change under load. Serves Llama-3.2-1B on an RTX 4090 with paged KV caching, continuous batching and an OpenAI-compatible streaming API, in original kernels with no PyTorch or vLLM in the runtime path.

At concurrency 128, vLLM returned different text for 85.2% of identical temperature-0 requests, compared with the same requests sent one at a time. stride changed 0 of 968 across all load levels: its output bits depend only on the request.

## Results

RTX 4090, same instance and same pinned weights for every engine. Prompt length 256 ± 32 tokens, 256 output tokens, temperature 0, prefix caching off. Every number links to a raw artifact through [the ledger](docs/resume/METRICS.jsonl).

**Same request, alone vs under load** (100 prompts × 256 tokens, compared with each request's own sequential output)

| Concurrency | stride: requests that changed | vLLM 0.32.0: requests that changed |
| --- | --- | --- |
| 4 | **0 / 100** | 82 / 100 (82.0%) |
| 16 | **0 / 100** | 80 / 100 (80.0%) |
| 64 | **0 / 256** | 222 / 256 (86.7%) |
| 128 | **0 / 512** | 436 / 512 (85.2%) |

Run sequentially twice, neither engine changed any output, so the vLLM differences come from running alongside other requests, not from randomness. stride's `system_fingerprint`, which covers the exact bits of every chosen logit, also matched on every request. This is known behaviour of batch-dependent numerics rather than a vLLM bug; see [Thinking Machines' analysis](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/). A single flipped token changes everything after it, so a 256-token output is a sensitive test.

**Throughput and latency** (NVIDIA AIPerf 0.13.0, mean of 3 repeats)

| Concurrency | stride tok/s | vLLM tok/s | stride / vLLM | TTFT p50 ms, stride / vLLM | Inter-token p50 ms, stride / vLLM | Request p99 ms, stride / vLLM |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 302 ± 0 | 316 ± 1 | 95.5% | 13 / 19 | 3.25 / 3.07 | 849 / 815 |
| 4 | 1,162 ± 2 | 1,121 ± 0 | 103.6% | 33 / 36 | 3.29 / 3.40 | 890 / 926 |
| 16 | 4,026 ± 6 | 3,907 ± 9 | 103.1% | 120 / 91 | 3.47 / 3.72 | 1,024 / 1,051 |
| 64 | 9,540 ± 12 | 10,907 ± 22 | 87.5% | 215 / 219 | 5.83 / 4.98 | 2,012 / 1,531 |
| 128 | 11,814 ± 8 | 14,947 ± 40 | 79.0% | 222 / 403 | 9.96 / 6.96 | 3,562 / 2,209 |

**Correctness** (100 prompts × 256 tokens against HF transformers on the same files)

| Comparison | Greedy token match | Identical sequences |
| --- | --- | --- |
| stride exact path vs HF, FP32 activations | **100.000%** | 100 / 100 |
| stride serving path vs HF, FP32 activations | 75.73% | 52 / 100 |
| HF BF16 vs HF FP32 (the noise floor of BF16 itself) | 38.76% | 6 / 100 |
| stride exact path, batch 1 vs batch 32 with shuffled lanes and pages | 100.000% | 100 / 100 |
| stride serving path, batch 1 vs batch 32 with shuffled lanes and pages | 100.000% | 100 / 100 |

Exact-path logits stay within 5.72e-05 of the FP32 reference across 400 vectors.

**Same bytes on different hardware.** On real Llama weights, the exact path produced byte-identical logits on an RTX 4090, an AMD EPYC 7702 and an Apple M1 Pro: 32,833,536 FP32 values with one SHA-256. The serving path does not reproduce on a CPU, but a second physical RTX 4090 in another data center matched on 100 of 100 prompts, by fingerprint and tokens over 25,600 generated tokens.

**Other measurements**

| Metric | Value |
| --- | --- |
| Batch-1 decode, serving path | 310.1 ± 0.4 tok/s, **76.0%** of the 407.9 tok/s bandwidth roofline |
| Batch-1 decode, exact path | 285.4 ± 0.5 tok/s, 70.0% of the roofline |
| Bug-injection catch rate, real weights | exact path 29 of 29; serving path 29 of 29 |
| Of those, caught by a single-request tolerance check | exact path 19; serving path 12. The rest needed the invariance or golden-bits oracles: some are rounding-level bugs, and some are concurrency bugs that a single-request run never reaches |
| Register-blocked tensor-core kernels (A/B) | inter-token p50 cut 33.8% and throughput up 50.2% at concurrency 128; identical output bits |
| Host time per step at concurrency 64 | 106 µs of 5,884 µs |
| GPU memory peak, stride / vLLM | 19,938 / 21,068 MiB, each engine's configured allocation rather than its minimum |

## How it works

For any output element, the operations and their order depend only on the matrix shape, never on the batch, the lane, the thread or where KV pages live. Dot products use fixed summation trees, `exp` is hand-written from FMA steps, attention sums pages in position order, and argmax is order-independent. An optimization that only moves work between warps therefore cannot change a bit, so it can be verified by comparing bits.

Two kernel families share one scheduler and server:

- **Serving path** (`--path tc`, default): tensor-core `mma` tiles with BF16 activations. Fast, and bit-stable under any load, but not reproducible on a CPU, and it differs from an FP32 reference as all BF16 inference does.
- **Exact path** (`--path exact`): scalar FMA chains with FP32 activations. Matches PyTorch to FP32 rounding and produces the same bytes on GPU and CPU, but it is much slower at large batches.

Responses carry `system_fingerprint: stride-<hash>`, chained over every token and the exact bits of its logit, so a client can verify that load changed nothing.

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Choices and the measurements behind them: [docs/DECISIONS.md](docs/DECISIONS.md).

## Methodology

- **Weights:** `unsloth/Llama-3.2-1B-Instruct` at commit `5a8abab4a5d6f164389b1079fb721cfab8d7126c`. The engine, the PyTorch reference and vLLM load the same verified directory with network fetches off for the engines, and [`scripts/verify_weights.py`](scripts/verify_weights.py) checks it against [`weights.lock.json`](weights.lock.json) before each run.
- **Fairness:** same instance, driver and CUDA; prefix caching off in vLLM and absent from stride; warmup is a separate AIPerf run with a different seed, so its prompts are disjoint from measured prompts (AIPerf issue #1656), and it is excluded; 3 repeats per point.
- **Roofline:** 1008 GB/s divided by the weight bytes one forward pass reads, as measured from the loaded matrices.
- **Reference:** HF transformers with BF16 weights and FP32 activations. See the noise-floor row above for why BF16 activations cannot serve as a precise reference.
- **Environment:** RTX 4090, driver 595.91.07, CUDA 13.2 (driver), toolkit 12.8, AMD EPYC 7702 64-Core Processor, RunPod Secure Cloud. See [reports/env/environment.txt](reports/env/environment.txt).

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

- **Misses against the targets:** batch-1 decode is 76.0% of the roofline against a target of 85%. Throughput at concurrency 64 is 87.5% of vLLM, past the 70% target but short of the 90% stretch. At concurrency 128 it is 79.0%, and p99 request latency is worse than vLLM's at both of those levels.
- **The serving path does not match FP32 token for token**, at 75.73%. It is closer than HF's own BF16 mode at 38.76%, but the 100% result belongs to the exact path, which is too slow at large batches to be the serving default.
- **The weights come from a mirror.** The upstream `meta-llama` repo is gated and masks its file hash, so byte identity with it is not verified; the sizes match.
- **Nsight Compute reports are missing.** `ncu` fails with `ERR_NVGPUCTRPERM` on RunPod, on both data centers tried, because GPU counters need admin rights the containers lack. Timing comes from Nsight Systems.
- **Determinism has limits:** on the serving path, bit-identical results are shown only for this model, on two RTX 4090s with the same driver. A different GPU architecture may round differently inside `mma`, and that has not been tested.
- **Not implemented:** prefix caching, multi-GPU, `min_tokens`, logprobs and tools. Llama-3.1-8B was not attempted. Maximum context is 2048 tokens.

## Reproduce

```bash
# Build. Add -DSTRIDE_CUDA=ON on a CUDA machine; without it you get the CPU backend.
cmake -S . -B build -G Ninja -DSTRIDE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Fetch the pinned snapshot, then check it.
python3 scripts/pin_weights.py --dest models/llama32-1b --lock /tmp/lock.json --commit 5a8abab4a5d6f164389b1079fb721cfab8d7126c
python3 scripts/verify_weights.py --dir models/llama32-1b

# Serve.
build/stride_server --model models/llama32-1b --port 8000 --kv-gb 16 --max-seqs 256 --max-rows 2048 --prefill-chunk 256

# Every number in this README, on one RTX 4090 (expects vLLM and AIPerf in /root/venv-vllm).
MODEL_DIR=$PWD/models/llama32-1b scripts/run_all.sh $(git rev-parse HEAD)
python3 scripts/build_ledger.py && python3 scripts/render_docs.py

# CPU tests, as in CI.
pytest tests/test_correctness.py
```

[ATTRIBUTION.md](ATTRIBUTION.md) lists libraries and prior work. Built with Llama. Code is Apache-2.0; the model is under the Llama 3.2 Community License.
