# Tuning log

Raw console results from kernel experiments, in the order they were run on the RTX 4090.
These are single runs of `stride_cli` used to choose a design, not benchmark results; the
reported numbers come from `scripts/run_all.sh`. Prompts were random token ids, 255 tokens
plus BOS (127 plus BOS in the earliest rows), with 256 generated tokens.

## Batch 1, exact path, first kernels against block-per-row kernels

| Kernels | tok/s |
| --- | --- |
| One thread per row | 74.1 |
| One block per row, interleaved threads | 254.4 (253.9, 254.7, 254.4) |

## Attention cost against context length, first kernels (one chain over all positions)

Sum of Attention kernel time per decode step, from an Nsight Systems trace at batch 1:

| Context position | 1 | 32 | 64 | 128 | 192 |
| --- | --- | --- | --- | --- | --- |
| Microseconds per step | 104 | 267 | 379 | 652 | 922 |

## Scaling of the first batched kernels (FMA, exact path)

| Batch | tok/s | ms/step |
| --- | --- | --- |
| 8 | 648 | 12.25 |
| 64 | 565 | 106.17 |
| 128 | 562 | 199.70 |

## First tensor-core kernels

| Batch | tok/s | ms/step |
| --- | --- | --- |
| 1 | 207 | 4.84 |
| 8 | 1427 | 5.56 |
| 16 | 2462 | 6.40 |
| 64 | 7126 | 8.42 |
| 128 | 9562 | 11.74 |

## Per-lane norm and multi-warp residual (prompts 127 plus BOS)

| Variant | Batch | tok/s | ms/step |
| --- | --- | --- | --- |
| Per-tile norm | 64 | 7551 | 8.35 |
| Per-lane norm | 64 | 8850 | 7.12 |
| Per-lane norm | 128 | 10368 | 11.97 |

## Attention load refactor (float4 shared values; same bits, prompts 127 plus BOS)

| Batch | tok/s | ms/step | AttentionPage us/step (decode) |
| --- | --- | --- | --- |
| 1 | 270 | 3.70 | |
| 64 | 10284 | 6.13 | 570, was 1371 |
| 128 | 11977 | 10.36 | |

## Register blocking, R row tiles by LT lane tiles (prompts 255 plus BOS; all bit-identical)

| Batch | 1x1 | 1x2 | 2x2 | 4x2 | 4x4 |
| --- | --- | --- | --- | --- | --- |
| 16 | 3460 | 3511 | 3288 | 2611 | 2567 |
| 64 | 7456 | 8032 | 7984 | 6364 | 4463 |
| 128 | 7853 | 8988 | 11183 | 9550 | 6513 |

## Block order (tok/s; rows-outer and lanes-outer are bit-identical)

| Blocking | Batch | Rows outer | Lanes outer |
| --- | --- | --- | --- |
| 2x2 | 64 | 8005 | 7564 |
| 2x2 | 128 | 11208 | 9848 |
| 1x1 | 128 | 7868 | 6341 |

## Paired gate and up chains (tok/s; bit-identical to sequential chains)

| Batch | Sequential | Paired |
| --- | --- | --- |
| 64 | 8034 | 8312 |
| 128 | 11204 | 11812 |

## Warps per block for MLP down (tok/s; changes the summation tree)

| Batch | K=1 | K=2 | K=4 | K=8 |
| --- | --- | --- | --- | --- |
| 1 | 226 | 259 | 271 | 271 |
| 16 | 3215 | 3625 | 3741 | 3772 |
| 64 | 5965 | 7675 | 8312 | 8551 |
| 128 | 9404 | 11284 | 11812 | 11783 |

## Warps per block for the output projection (tok/s; changes the summation tree)

| Batch | K=1 | K=2 | K=4 | K=8 |
| --- | --- | --- | --- | --- |
| 1 | 272 | 282 | 284 | 287 |
| 64 | 8544 | 9263 | 9455 | 9566 |
| 128 | 11834 | 12476 | 12591 | 12643 |

## Warps per block for Q, K, V and for the MLP (tok/s)

| Variant | Batch 1 | Batch 64 | Batch 128 |
| --- | --- | --- | --- |
| QKV K=1, MLP K=1 | 294 | 9932 | 12754 |
| QKV K=4, MLP K=1 | 305 | 9892 | 12447 |
| QKV K=8, MLP K=1 (chosen) | 306 | 10022 | 12615 |
| QKV K=8, MLP K=2 | 305 | 10039 | 12358 |
| QKV K=8, MLP K=4 | 305 | 9865 | 12253 |
