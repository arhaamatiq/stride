# Resume lines

Generated from the ledger by `scripts/render_docs.py`, which rounds these numbers down and rejects any line over 104 characters.

Stride | C++, CUDA, Tensor Cores, Python, PyTorch, Nsight Systems, vLLM, AIPerf

## Target lines

1. Built a C++/CUDA LLM inference engine from scratch, serving Llama-3.2-1B at {{g3.aiperf.stride.c128.output_tok_per_s|floor,}} tok/s on an RTX 4090
2. Engineered paged KV caching and continuous batching for 64 concurrent requests at {{g3.aiperf.pct_of_vllm.c64|floor}}% of vLLM throughput
3. Wrote {{g3.fused_kernels}} fused CUDA kernels, profiled in Nsight Systems; batch-1 decode hits {{g3.batch1.tc.pct_of_roofline|floor}}% of peak memory bandwidth
4. Built a differential harness vs PyTorch: {{g1.greedy.stride_exact_vs_hf_fp32.pct|floor}}% greedy match (FP32 path); caught {{g5.cpu_tiny.caught}}/{{g5.cpu_tiny.mutants}} injected bugs in CI
5. Cut inter-token latency {{g4.micro.c128.itl_p50_cut_pct|floor}}% via register-blocked tensor-core GEMMs, verified by NVIDIA AIPerf A/B sweeps

## Standout lines

6. Made outputs load-invariant: {{g2.determinism.stride.total_requests_differing}} of {{g2.determinism.stride.total_requests}} requests changed under load, vs {{g2.determinism.vllm.c128.requests_differing_pct|floor}}% on vLLM at concurrency 128
7. Produced byte-identical Llama logits on RTX 4090, x86 and ARM ({{g1.cross_device.logit_values|floor,}} values, one SHA-256)
