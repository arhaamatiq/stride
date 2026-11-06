# Resume lines

Generated from the ledger by `scripts/render_docs.py`, which rounds these numbers down and rejects any line over 104 characters.

Stride | C++, CUDA, Tensor Cores, Python, PyTorch, Nsight Systems, vLLM, AIPerf

## Target lines

1. Built a C++/CUDA LLM inference engine from scratch, serving Llama-3.2-1B at 11,813 tok/s on an RTX 4090
2. Engineered paged KV caching and continuous batching for 64 concurrent requests at 87% of vLLM throughput
3. Wrote 9 fused CUDA kernels, profiled in Nsight Systems; batch-1 decode hits 76% of peak memory bandwidth
4. Built a differential harness vs PyTorch: 100% greedy match (FP32 path); caught 29/29 injected bugs in CI
5. Cut inter-token latency 33% via register-blocked tensor-core GEMMs, verified by NVIDIA AIPerf A/B sweeps

## Standout lines

6. Made outputs load-invariant: 0 of 968 requests changed under load, vs 85% on vLLM at concurrency 128
7. Produced byte-identical Llama logits on RTX 4090, x86 and ARM (32,833,536 values, one SHA-256)
