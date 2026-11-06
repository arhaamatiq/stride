// Deterministic scalar numerics. Every function here is built only from
// IEEE-754 correctly rounded operations (+, *, /, sqrt, fma, floor), so its bits
// do not depend on the platform's libm or on the compiler's contraction choices.
#pragma once

#include "hd.h"

namespace stride {

// BF16 to FP32 is exact: a BF16 value is the top half of the FP32 bit pattern.
STRIDE_DEVICE float bf16_lo(uint32_t packed) { return as_f32(packed << 16); }
STRIDE_DEVICE float bf16_hi(uint32_t packed) { return as_f32(packed & 0xFFFF0000u); }

// exp(x) from a fixed Cody-Waite split and fixed FMA polynomial. Accurate to
// roughly 1-2 ulp; what matters here is that the bits are the same everywhere.
STRIDE_DEVICE float det_exp(float x) {
  if (x < -87.0f) return 0.0f;
  if (x > 88.0f) return as_f32(0x7F800000u);
  const float k = floorf(fmaf(x, 1.44269504088896341f, 0.5f));
  float r = fmaf(-k, 0.693359375f, x);
  r = fmaf(-k, -2.12194440e-4f, r);
  float p = 1.9875691500e-4f;
  p = fmaf(p, r, 1.3981999507e-3f);
  p = fmaf(p, r, 8.3334519073e-3f);
  p = fmaf(p, r, 4.1665795894e-2f);
  p = fmaf(p, r, 1.6666665459e-1f);
  p = fmaf(p, r, 5.0000001201e-1f);
  const float y = fmaf(fmaf(p, r, 1.0f), r, 1.0f);
  return as_f32(as_u32(y) + (static_cast<uint32_t>(static_cast<int32_t>(k)) << 23));
}

// SiLU: x * sigmoid(x) = x / (1 + exp(-x)).
STRIDE_DEVICE float det_silu(float x) { return x / (1.0f + det_exp(-x)); }

}  // namespace stride
