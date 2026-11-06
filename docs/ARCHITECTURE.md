# Architecture

stride serves Llama-3.2-1B from C++ and CUDA, with one property most engines do not have:
a request's output bits do not depend on what else is running.

```
HTTP (server.cpp) ──> per-request queues <── token events
        │                                         ▲
        ▼                                         │
Scheduler (scheduler.cpp)  continuous batching, admission, preemption and replay
        │
        ▼
Engine (engine.cpp)        one forward step over an arbitrary set of rows
        │        │
        │        └── KvCache (kv_cache.cpp)  block allocator over one device arena
        ▼
Kernels    kernels.cu       exact path: FMA, FP32, runs on GPU or CPU
           kernels_tc.cu    tensor-core path: mma, BF16 tiles, GPU only
           kernel_common.h  paged attention, atomic argmax, launcher
```

## Two kernel paths

| | Exact | Tensor-core (serving default) |
| --- | --- | --- |
| Arithmetic | scalar FMA chains | `mma` on 16x16x16 tiles |
| Activations and KV | FP32 | BF16 x tiles and KV, FP32 `h` |
| Same bits at any batch or schedule | yes | yes |
| Same bits on CPU and GPU | yes | no: the hardware's internal summation is not a simple IEEE sequence |
| Agrees with HF at FP32 | to within FP32 rounding | to within BF16 rounding noise |
| Use | correctness oracle, reproducible runs | throughput |

Both run from one scheduler and one server, selected with `--path`.

## Why outputs do not depend on the batch

Results change with batch size in most engines because a BLAS library picks a different
blocking, and so a different summation order, for different shapes. Floating-point addition
is not associative, so a different order gives different low bits, and over many steps those
bits can flip a greedy choice.

stride removes the dependency at its source. For any output element, the operations and
their order depend only on the matrix shape:

- **Dot products** split into fixed partial sums. The exact path uses 32 partials by weight
  line; the tensor-core path chains column tiles in ascending order. Partials combine left to
  right.
- **Warps per block** are a constant of the kernel (`kFewRowsWarps`), never a function of the
  batch.
- **Attention** sums each 64-position page on its own and combines pages in ascending order.
  Pages are addressed by absolute position, so KV placement cannot matter.
- **Argmax** keeps the maximum of (value, -index) with a compare-and-swap, so its result does
  not depend on which block finishes first.
- **`exp`** is hand-written from FMA steps, and `/` and `sqrt` are correctly rounded, so no
  libm enters the path. The compiler may not contract `a*b+c` (`-fmad=false`,
  `-ffp-contract=off`).
- **Sampling** draws from a generator keyed on (seed, position), not on a stream, so the draw
  does not depend on scheduling.

Register blocking, per-lane norms and paired MLP chains change only which warp does what.
That is why optimizations here can be checked by comparing bits.

## Memory layout

| Data | Layout | Why |
| --- | --- | --- |
| Weights, exact | `[group][col][4 rows]` | a 16-byte load is two columns of four rows |
| Weights, tensor-core | `[row tile][col tile][16 rows][16 cols]` | a warp load covers one contiguous tile |
| x tiles | `[lane tile][col tile][16 lanes][16 cols]` | matches the `mma` B operand |
| KV block | `[block][layer][K or V][group][token][dim]` | a page of one group is contiguous |

Weight matrices with RoPE (Q, K) group each `(d, d + D/2)` pair into one tile, so rotation
happens in the warp that finished the sums.

## Fused kernels

A fused kernel does in one launch what would otherwise take several launches and an
intermediate buffer.

| Kernel | Merges | Avoids |
| --- | --- | --- |
| `Qkv`, `TcQkv` | Q, K and V projections, RoPE, KV append | three launches, a rotation pass, K and V scratch |
| `GemmResidual`, `TcGemmResidual` | projection and residual add | an output buffer and an add pass; on tiles `h` is the initial accumulator |
| `Mlp13`, `TcMlp13` | gate and up projections, SiLU, product | two intermediate vectors; on tiles one x load serves both matrices |
| `LmGreedy`, `TcLmGreedy` | output projection and argmax | writing logits to memory |
| `AttentionPage` | scores, softmax and value mix for one KV page | separate score and weight buffers |

## Paged KV cache

The cache is a free list of 64-token blocks over one arena, with a table per sequence. Memory
follows actual length rather than a reserved maximum, and allocation costs a vector pop.

## Scheduler

Each step runs one decode row for every sequence that is ready, then prefill rows up to the
step budget. Sequences join and leave between steps. If the arena fills, the youngest sequence
is preempted, its blocks are freed, and it replays later. Replay is bit-exact, so a client
cannot observe a preemption in its output. The test suite asserts that replay really happened.

## Verifying determinism from outside

The API returns `system_fingerprint: stride-<hash>`, a hash chained over every generated token
and the exact bits of its logit. Two responses with the same fingerprint computed the same
values, so a client can check for itself that load changed nothing.

## Test harness

- `tests/test_correctness.py`: CPU tests against PyTorch, schedule invariance, preemption
  replay, seeded sampling. Runs in CI.
- `tests/g1_real.py`: differential run on real weights.
- `tests/mutations.py`: 29 injected bugs in kernels, KV and scheduler, each checked by a
  tolerance oracle, an invariance oracle and a golden-bits oracle.
- `scripts/cross_device.py`: byte comparison of exact-path logits across devices.
- `scripts/run_all.sh`: reruns everything behind the numbers in the README.
