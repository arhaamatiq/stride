# Decisions

Decisions made during the build, with the reason and, where it exists, the data.

## Project setup

| Decision | Reason |
| --- | --- |
| Spend cap of $30 on GPU time | The brief left the cap blank. About 33 hours at $0.89/hr on RunPod Secure Cloud. |
| RTX 4090 on RunPod Secure Cloud, EUR-IS-1 | Community Cloud had no stock. See `reports/env/environment.txt`. |
| Weights from `unsloth/Llama-3.2-1B-Instruct`, pinned by commit | No Hugging Face token was available for the gated upstream repo; the owner approved this mirror. The commit and the SHA256 of every file are in `weights.lock.json`, and `scripts/verify_weights.py` runs before every engine, reference and vLLM run, so all three load byte-identical files. The upstream repo masks its LFS hash from unauthenticated requests, so byte identity with `meta-llama/Llama-3.2-1B-Instruct` is not verified; the file sizes match. |
| Reference is HF transformers with BF16 weights and FP32 activations | BF16 activations round every intermediate to an 8-bit mantissa. Measured: HF BF16 agrees with HF FP32 on 38.76% of greedy tokens over 100 prompts, so BF16 cannot serve as a precise reference. It is kept as the noise floor. |

## Numerics

| Decision | Reason |
| --- | --- |
| Hand-written `exp`, explicit `fmaf`, IEEE `/` and `sqrt`, `-fmad=false`, `-ffp-contract=off` | Results must not depend on a libm or on compiler contraction. This is what lets the exact path produce the same bytes on an RTX 4090, an x86 CPU and an ARM CPU. |
| Fixed summation trees that depend only on matrix shape | Batch invariance by construction: an output element sees the same sequence of operations regardless of batch, lane, thread or KV placement. |
| Two kernel families | The exact path (FMA, FP32) gives CPU/GPU byte identity and the G1 result. The tensor-core path (mma, BF16) gives batched throughput; scalar FMA cannot get close, because cuBLAS reaches 51 T MAC/s on these matrices at batch 128. |
| Serving defaults to the tensor-core path | At batch 64 the first FMA kernels reached 565 tok/s and the first tile kernels 7,126 (`reports/g3/tuning_log.md`). Its outputs are bit-stable under load but differ from the FP32 reference, as all BF16 inference does. |

## Performance choices driven by measurement

| Decision | Data |
| --- | --- |
| One block per row group, threads interleaved over cache lines | `reports/g3/memread.csv`: on a 525 MB matrix, 489 GB/s with one thread per row, 913 GB/s interleaved. Decode went from 74 to 254 tok/s with no change in output bits. |
| Tile GEMMs on tensor cores | `reports/g3/cublas_probe.txt`, `reports/g3/tc_probe.txt`. An mma result was bit-identical in 1,048,576 of 1,048,576 elements when rows and lanes were permuted, so tiles preserve invariance. |
| RMSNorm with one block per lane | The per-tile version gave a batch of 64 only four warps and spent its time on load latency. See the A/B in `reports/g4`. |
| Paged attention | A single chain over positions cost about 4.4 microseconds per context position per step and was latency-bound. Page results combine in a fixed order, so more blocks read KV at once. |
| Rows-outer block order | Tried lanes-outer to keep x tiles in L1. It was slower: 9,848 against 11,208 tok/s at batch 128. Rejected. |
| Register blocking at 2x2 | 4x2 and 4x4 were slower than 2x2 at every batch tested; likely register pressure, not investigated further. |
| Eight warps per block for Q, K, V, output and MLP down | These matrices have few rows and starve for warps. For MLP down at batch 64: K=1 5,965, K=2 7,675, K=4 8,312, K=8 8,551 tok/s. MLP gate and up stayed at K=1 because K=2 and K=4 did not help. |
| Fused argmax with an atomic maximum | Logits are never written to memory on the greedy path. The maximum of (value, -index) does not depend on update order, so racing blocks still give one deterministic answer. |

## Harness

| Decision | Reason |
| --- | --- |
| Tiny test model has random norm gains and a larger initializer | With default init every norm gain is exactly 1.0 and attention is nearly silent, so several injected bugs changed nothing and could not be caught. |
| Preemption test uses long generations in a small KV arena, and fails if nothing was preempted | The first version's arena was large enough that no preemption ever happened, which hid an injected replay bug. |
| Mutation suite also runs on real weights | Tolerance checks look artificially weak on a random model, so a claim about what only bit-exact oracles catch needs real weights. |
| Serving-path mutation run uses first-step logits and no token oracle | The BF16 path is not expected to agree token for token with an FP32 reference. The tolerance is set from the measured BF16 noise floor. |

## Limits

| Item | Status |
| --- | --- |
| Nsight Compute | `ncu` fails with `ERR_NVGPUCTRPERM` on RunPod: `RmProfilingAdminOnly` is 1 and containers lack `CAP_SYS_ADMIN`. Nsight Systems works and supplies per-kernel timing. Per-kernel hardware counters need a VM where counters are enabled. |
| Stretch: Llama-3.1-8B | Not attempted. First in the cut order. |
| Prefix caching | Not implemented. Off in vLLM for every comparison. |
| `min_tokens`, logprobs, penalties, tools | Not implemented; the server returns 400. |
