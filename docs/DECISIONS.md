# Decisions

Decisions made during the build, with the reason and, where it exists, the data.

## Project setup

| Decision | Reason |
| --- | --- |
| GPU budget of $30 | About 33 hours at $0.89/hr on RunPod Secure Cloud. |
| RTX 4090 on RunPod Secure Cloud | Community Cloud had no stock. Host details are in `reports/env`. |
| Weights from `unsloth/Llama-3.2-1B-Instruct`, pinned by commit | The upstream repo is gated. The commit and the SHA-256 of every file are in `weights.lock.json`, and `scripts/run_all.sh` verifies at the start, and the G1, load-study and sweep stages also verify and record the digest, so all three can be shown to load byte-identical files. The upstream repo masks its LFS hash from unauthenticated requests, so byte identity with `meta-llama/Llama-3.2-1B-Instruct` is not verified. `model.safetensors` matches upstream's size; `config.json` is the mirror's edited version. |
| Reference is HF transformers with BF16 weights and FP32 activations | BF16 activations round every intermediate to an 8-bit mantissa. HF BF16 agrees with HF FP32 on 38.76% of greedy tokens over 100 prompts, position by position, so BF16 cannot serve as a precise reference. It is kept as a measure of how far BF16 drifts. |

## Numerics

| Decision | Reason |
| --- | --- |
| Hand-written `exp`, explicit `fmaf`, IEEE `/` and `sqrt`, `-fmad=false`, `-ffp-contract=off` | Results must not depend on a libm or on compiler contraction. This is what lets the exact path produce the same bytes on an RTX 4090, an x86 CPU and an ARM CPU. |
| Fixed summation trees that depend only on matrix shape | An output element sees the same sequence of operations regardless of batch, lane, thread or KV placement. |
| Two kernel families | The exact path (FMA, FP32) gives CPU/GPU byte identity and the G1 result. The tensor-core path (mma, BF16) gives batched throughput. Scalar FMA cannot get close: cuBLAS in BF16 reaches about 51 T MAC/s on `lm_head` at batch 128 (`reports/g3/cublas_probe.txt`). |
| Serving defaults to the tensor-core path | At batch 64 the first FMA kernels reached 565 tok/s and the first tile kernels 7,126 (`reports/g3/tuning_log.md`). Its outputs were stable under load in every test but differ from the FP32 reference, as BF16 inference generally does. |

## Performance choices driven by measurement

Single-run experiment numbers below are in `reports/g3/tuning_log.md`.

| Decision | Data |
| --- | --- |
| One block per row group, threads interleaved over cache lines | `reports/g3/memread.csv`: on a 525 MB matrix, one thread per row reached about half the bandwidth of interleaved threads. Decode went from 74 to 254 tok/s, and logits stayed byte-identical to the CPU build. |
| Tile GEMMs on tensor cores | `reports/g3/cublas_probe.txt`, `reports/g3/tc_probe.txt`. An mma result was bit-identical when rows and lanes were permuted, which suggested tiles could preserve invariance. End-to-end runs then confirmed it. |
| RMSNorm with one block per lane | The per-tile version gave a batch of 64 only four warps. See the A/B in `reports/g4`. |
| Paged attention | A single chain over positions cost about 4.3 microseconds per context position per step and was latency-bound. Page results combine in a fixed order, so more blocks read KV at once. |
| Rows-outer block order | Lanes-outer was tried to keep x tiles in L1 and was slower: 9,848 against 11,208 tok/s at batch 128. Rejected. |
| Register blocking at 2x2 | 4x2 and 4x4 were slower than 2x2 at every batch tested; likely register pressure, not investigated further. 1x2 was slightly ahead at batch 16 (3,511 against 3,288) and batch 64 (8,032 against 7,984), but 2x2 was far ahead at batch 128 (11,183 against 8,988). |
| Eight warps per block for Q, K, V, output and MLP down | These matrices have few rows and can starve for warps. For MLP down at batch 64: K=1 5,965, K=2 7,675, K=4 8,312, K=8 8,551 tok/s. For the output projection, 8,544 to 9,566. For Q, K and V the evidence is mixed: K=8 against K=1 is 4.1% faster at batch 1, 0.9% at 64 and 1.1% slower at 128. MLP gate and up stayed at K=1 because K=2 and K=4 did not help. |
| Fused argmax with an atomic maximum | Logits are never written to memory on the greedy path. The maximum of (value, -index) does not depend on update order, so racing blocks still give one answer. |

## Harness

| Decision | Reason |
| --- | --- |
| Tiny test model has random norm gains and a larger initializer | With default init every norm gain is exactly 1.0 and attention is nearly silent, so several injected bugs changed nothing and could not be caught. |
| Preemption test uses long generations in a small KV arena, and fails if nothing was preempted | The first version's arena was large enough that no preemption ever happened, which hid an injected replay bug. |
| Mutation suite also runs on real weights | It confirms the small-model result holds on real weights: on the exact path the tolerance oracle catches 19 of 25 functional bugs on both. |
| Serving-path mutation run uses first-step logits and no token oracle | The BF16 path is not expected to agree token for token with an FP32 reference. The ±0.6 tolerance is set from how far BF16 was measured to drift, so it is deliberately wide. |
| Rounding-only mutants are counted separately | A reversed sum, or a RoPE shift applied to Q and K alike, changes only rounding. Catching those shows the golden oracle is sensitive, not that it finds functional bugs. |
| NaN logits count as an infinite difference | `nan > tolerance` is false, so NaN would otherwise pass the logit check. |

## Limits

| Item | Status |
| --- | --- |
| Nsight Compute | `ncu` fails with `ERR_NVGPUCTRPERM` on the RunPod hosts tried (`reports/env/ncu_probe.txt`, `reports/g1/host2/ncu_probe.txt`): `RmProfilingAdminOnly` is 1 and containers lack `CAP_SYS_ADMIN`. Nsight Systems works and supplies per-kernel timing. Per-kernel hardware counters need a VM where counters are enabled. |
| Nsight Systems reports | They embed the container's environment, including a RunPod key and SSH client address, even when the profiled process has an empty environment. `scripts/redact_nsys.py` overwrites those values with same-length bytes, and reports still open afterward. |
| Stretch: Llama-3.1-8B | Not attempted. First in the cut order. |
| Prefix caching | Not implemented. Off in vLLM for every comparison. |
| `min_tokens`, logprobs, penalties, tools | Not implemented, except that `min_tokens` is accepted together with `ignore_eos`, where it has no effect. Otherwise the server returns 400. |
