// Host/device shim. The same kernel source compiles with nvcc for the GPU and
// with a plain C++ compiler for the CPU backend, so correctness tests run in CI
// without a GPU and CPU/GPU results can be compared bit for bit.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#define STRIDE_DEVICE __device__ __forceinline__
#define STRIDE_HD __host__ __device__ __forceinline__
namespace stride {
STRIDE_DEVICE float as_f32(uint32_t u) { return __uint_as_float(u); }
STRIDE_DEVICE uint32_t as_u32(float f) { return __float_as_uint(f); }
}  // namespace stride
#else
#define STRIDE_DEVICE inline __attribute__((always_inline))
#define STRIDE_HD inline __attribute__((always_inline))
struct alignas(16) float4 {
  float x, y, z, w;
};
struct alignas(16) uint4 {
  uint32_t x, y, z, w;
};
namespace stride {
STRIDE_DEVICE float as_f32(uint32_t u) {
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}
STRIDE_DEVICE uint32_t as_u32(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}
}  // namespace stride
#endif

#if defined(STRIDE_MUTATION)
#define STRIDE_BUG(p, id) ((p).bug == (id))
#else
#define STRIDE_BUG(p, id) false
#endif
