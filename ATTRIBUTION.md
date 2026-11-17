# Attribution

## Libraries in the runtime path

| Library | Used for | License | Version |
| --- | --- | --- | --- |
| [nlohmann/json](https://github.com/nlohmann/json) | safetensors header, configuration, server JSON | MIT | 3.11.3 |
| [PCRE2](https://github.com/PCRE2Project/pcre2) | tokenizer split regex | BSD-3-Clause with PCRE2 exception | 10.47 |
| [cpp-httplib](https://github.com/yhirose/cpp-httplib) | HTTP server | MIT | 0.59.0 |
| NVIDIA CUDA Toolkit | `nvcc`, CUDA runtime, `mma.h` tensor-core intrinsics | NVIDIA EULA | 12.8 |

cuBLAS is not called by the runtime path. Every GEMM is an original kernel. The safetensors parser and the BPE tokenizer are original.

## Used only in tests and benchmarks

| Tool | Used for | License |
| --- | --- | --- |
| [PyTorch](https://github.com/pytorch/pytorch), [transformers](https://github.com/huggingface/transformers) | reference path; tiny-model generation | BSD-3-Clause, Apache-2.0 |
| [vLLM](https://github.com/vllm-project/vllm) | baseline | Apache-2.0 |
| [NVIDIA AIPerf](https://github.com/NVIDIA/aiperf) | load generation and latency measurement | Apache-2.0 |
| NVIDIA Nsight Systems | timelines | NVIDIA EULA |

## Model

Llama 3.2 is licensed under the [Llama 3.2 Community License](https://www.llama.com/llama3_2/license/), Copyright © Meta Platforms, Inc. All Rights Reserved. Built with Llama. Model weights are not distributed in this repository. Results use `unsloth/Llama-3.2-1B-Instruct`, pinned in [`weights.lock.json`](weights.lock.json).

## Prior work studied, not copied

- [yalm](https://github.com/andrewkchan/yalm): the idea of expressing batch-1 decode as a percentage of the memory-bandwidth ceiling.
- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm): a small paged-KV and continuous-batching serving design.
- Horace He and Thinking Machines Lab, [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/): batch-dependent numerics as the main cause of nondeterministic LLM output.

No code was adapted from these projects.
