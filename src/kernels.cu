// Fused kernels. Compiled by nvcc for CUDA and as plain C++ for the CPU backend.
//
// Batch invariance is a property of this file, not of a test: for a given output
// element, every kernel performs the same sequence of IEEE operations no matter
// how many sequences share the step, which lane the sequence landed in, or which
// thread ran which part. Rules that keep that true:
//   * a dot product is split across 32 threads by 64-byte weight line: thread t
//     owns lines l == t (mod 32), and chains FMAs in ascending column order;
//   * the 32 partial sums are combined left to right;
//   * that summation tree depends only on the matrix shape, never on the batch;
//   * attention sums run in ascending position order regardless of KV placement.
//
// The same layout is what makes the reads fast. tools/bench_memread.cu measured
// 489 GB/s when each thread sweeps its own row and 913 GB/s when a block owns a row
// and its threads interleave, because every warp load then covers contiguous memory.
#include "kernel_common.h"

namespace stride {

namespace kimpl {

STRIDE_DEVICE void unpack_weights(const uint4& v, float* w) {
  w[0] = bf16_lo(v.x);
  w[1] = bf16_hi(v.x);
  w[2] = bf16_lo(v.y);
  w[3] = bf16_hi(v.y);
  w[4] = bf16_lo(v.z);
  w[5] = bf16_hi(v.z);
  w[6] = bf16_lo(v.w);
  w[7] = bf16_hi(v.w);
}

// One column of a four-row group against one lane.
STRIDE_DEVICE void fma_col1(const float* w, float x, float* acc) {
  acc[0] = fmaf(w[0], x, acc[0]);
  acc[1] = fmaf(w[1], x, acc[1]);
  acc[2] = fmaf(w[2], x, acc[2]);
  acc[3] = fmaf(w[3], x, acc[3]);
}

// One column of a four-row group against four lanes.
STRIDE_DEVICE void fma_col4(const float* w, const float4& x, float* acc) {
  for (int r = 0; r < 4; ++r) {
    acc[r * 4 + 0] = fmaf(w[r], x.x, acc[r * 4 + 0]);
    acc[r * 4 + 1] = fmaf(w[r], x.y, acc[r * 4 + 1]);
    acc[r * 4 + 2] = fmaf(w[r], x.z, acc[r * 4 + 2]);
    acc[r * 4 + 3] = fmaf(w[r], x.w, acc[r * 4 + 3]);
  }
}

STRIDE_DEVICE void gemm_unit4(const uint4& v, const float* x, int stride, int k, float* acc) {
  float w[8];
  unpack_weights(v, w);
  const float4 a = *reinterpret_cast<const float4*>(x + static_cast<int64_t>(k) * stride);
  const float4 b = *reinterpret_cast<const float4*>(x + static_cast<int64_t>(k + 1) * stride);
  fma_col4(w, a, acc);
  fma_col4(w + 4, b, acc);
}

// Partial dot products for thread t: every 32nd 64-byte weight line (8 columns).
// All four 16-byte weight loads are issued before the first is used, so a thread
// has a full cache line in flight instead of stalling on each load.
template <int J>
STRIDE_DEVICE void gemm_partial(const uint16_t* wrow, const float* x, int stride, int cols, int t, float* acc) {
  for (int e = 0; e < 4 * J; ++e) acc[e] = 0.0f;
  const int lines = cols / 8;
  for (int line = t; line < lines; line += kBlockThreads) {
    const int k = line * 8;
    const uint4* wl = reinterpret_cast<const uint4*>(wrow + static_cast<int64_t>(k) * 4);
    const uint4 v0 = wl[0], v1 = wl[1], v2 = wl[2], v3 = wl[3];
    if constexpr (J == 1) {
      const float4 a = *reinterpret_cast<const float4*>(x + k);
      const float4 b = *reinterpret_cast<const float4*>(x + k + 4);
      float w[8];
      unpack_weights(v0, w);
      fma_col1(w, a.x, acc);
      fma_col1(w + 4, a.y, acc);
      unpack_weights(v1, w);
      fma_col1(w, a.z, acc);
      fma_col1(w + 4, a.w, acc);
      unpack_weights(v2, w);
      fma_col1(w, b.x, acc);
      fma_col1(w + 4, b.y, acc);
      unpack_weights(v3, w);
      fma_col1(w, b.z, acc);
      fma_col1(w + 4, b.w, acc);
    } else {
      gemm_unit4(v0, x, stride, k, acc);
      gemm_unit4(v1, x, stride, k + 2, acc);
      gemm_unit4(v2, x, stride, k + 4, acc);
      gemm_unit4(v3, x, stride, k + 6, acc);
    }
  }
}

STRIDE_DEVICE const float* lane_base(const float* x, int lane_group) { return x + lane_group * 4; }

struct Embed {
  using Params = EmbedParams;
  static constexpr int kPhases = 1;
  struct Shared {
    char unused;
  };
  static int blocks(const Params& p) { return p.l.rows; }
  template <int P>
  STRIDE_DEVICE static void phase(const Params& p, int block, int t, Shared&) {
    const int token = p.tokens[block];
    const uint16_t* row = p.embed + static_cast<int64_t>(token / 4) * p.dim * 4;
    const int r = token % 4;
    for (int k = t; k < p.dim; k += kBlockThreads) {
      float v = as_f32(static_cast<uint32_t>(row[static_cast<int64_t>(k) * 4 + r]) << 16);
      if (STRIDE_BUG(p.l, 101) && k == 0) v = -v;
      p.h[static_cast<int64_t>(k) * p.l.stride + block] = v;
    }
  }
};

// RMSNorm over four-dim units; thread t owns units u == t (mod 32).
template <int J>
struct RmsNorm {
  using Params = NormParams;
  static constexpr int kPhases = 3;
  struct Shared {
    float partial[kBlockThreads][J];
    float r[J];
  };
  static int blocks(const Params& p) { return p.l.lane_groups; }
  template <int P>
  STRIDE_DEVICE static void phase(const Params& p, int block, int t, Shared& s) {
    const int units = p.dim / 4;
    const float* in = J == 1 ? p.in : lane_base(p.in, block);
    float* out = J == 1 ? p.out : p.out + block * 4;
    const int stride = p.l.stride;
    if constexpr (P == 0) {
      float acc[J];
      for (int j = 0; j < J; ++j) acc[j] = 0.0f;
      for (int u = t; u < units; u += kBlockThreads) {
        const int k = u * 4;
        if constexpr (J == 1) {
          const float4 v = *reinterpret_cast<const float4*>(in + k);
          acc[0] = fmaf(v.x, v.x, acc[0]);
          acc[0] = fmaf(v.y, v.y, acc[0]);
          acc[0] = fmaf(v.z, v.z, acc[0]);
          acc[0] = fmaf(v.w, v.w, acc[0]);
        } else {
          for (int c = 0; c < 4; ++c) {
            const float4 v = *reinterpret_cast<const float4*>(in + static_cast<int64_t>(k + c) * stride);
            acc[0] = fmaf(v.x, v.x, acc[0]);
            acc[1] = fmaf(v.y, v.y, acc[1]);
            acc[2] = fmaf(v.z, v.z, acc[2]);
            acc[3] = fmaf(v.w, v.w, acc[3]);
          }
        }
      }
      for (int j = 0; j < J; ++j) s.partial[t][j] = acc[j];
    } else if constexpr (P == 1) {
      if (t < J) {
        float sum = s.partial[0][t];
        if (STRIDE_BUG(p.l, 102)) {
          sum = s.partial[kBlockThreads - 1][t];
          for (int i = kBlockThreads - 2; i >= 0; --i) sum = sum + s.partial[i][t];
        } else {
          for (int i = 1; i < kBlockThreads; ++i) sum = sum + s.partial[i][t];
        }
        float denom = static_cast<float>(p.dim);
        if (STRIDE_BUG(p.l, 103)) denom = static_cast<float>(p.dim - 1);
        const float eps = STRIDE_BUG(p.l, 104) ? 0.0f : p.eps;
        s.r[t] = 1.0f / sqrtf(sum / denom + eps);
      }
    } else {
      for (int u = t; u < units; u += kBlockThreads) {
        const int k = u * 4;
        const float4 w4 = *reinterpret_cast<const float4*>(p.weight + k);
        const float w0 = STRIDE_BUG(p.l, 105) ? 1.0f : w4.x;
        if constexpr (J == 1) {
          const float4 v = *reinterpret_cast<const float4*>(in + k);
          const float r = s.r[0];
          float4 o;
          o.x = w0 * (v.x * r);
          o.y = w4.y * (v.y * r);
          o.z = w4.z * (v.z * r);
          o.w = w4.w * (v.w * r);
          *reinterpret_cast<float4*>(out + k) = o;
        } else {
          const float gains[4] = {w0, w4.y, w4.z, w4.w};
          for (int c = 0; c < 4; ++c) {
            const int64_t at = static_cast<int64_t>(k + c) * stride;
            const float4 v = *reinterpret_cast<const float4*>(in + at);
            float4 o;
            o.x = gains[c] * (v.x * s.r[0]);
            o.y = gains[c] * (v.y * s.r[1]);
            o.z = gains[c] * (v.z * s.r[2]);
            o.w = gains[c] * (v.w * s.r[3]);
            *reinterpret_cast<float4*>(out + at) = o;
          }
        }
      }
    }
  }
};

// Q, K and V projections fused with RoPE and the KV append. Each four-row group
// holds two RoPE pairs, so rotation happens right after the sums.
template <int J>
struct Qkv {
  using Params = QkvParams;
  static constexpr int kPhases = 3;
  struct Shared {
    float p[kBlockThreads][4 * J];
    float y[4 * J];
  };
  static int blocks(const Params& p) { return (p.q.groups + 2 * p.k.groups) * p.l.lane_groups; }

  template <int P>
  STRIDE_DEVICE static void phase(const Params& p, int block, int t, Shared& s) {
    const int group_index = block / p.l.lane_groups;
    const int lane_group = block % p.l.lane_groups;
    int kind, group;
    const GemmMat* m;
    if (group_index < p.q.groups) {
      kind = 0;
      group = group_index;
      m = &p.q;
    } else if (group_index < p.q.groups + p.k.groups) {
      kind = 1;
      group = group_index - p.q.groups;
      m = &p.k;
    } else {
      kind = 2;
      group = group_index - p.q.groups - p.k.groups;
      m = &p.v;
    }
    if constexpr (P == 0) {
      float acc[4 * J];
      gemm_partial<J>(m->w + static_cast<int64_t>(group) * m->cols * 4,
                      J == 1 ? p.x : lane_base(p.x, lane_group), p.l.stride, m->cols, t, acc);
      for (int e = 0; e < 4 * J; ++e) s.p[t][e] = acc[e];
    } else if constexpr (P == 1) {
      if (t < 4 * J) {
        float y = s.p[0][t];
        for (int i = 1; i < kBlockThreads; ++i) y = y + s.p[i][t];
        s.y[t] = y;
      }
    } else {
      if (t >= J) return;
      const int j = lane_group * 4 + t;
      if (j >= p.l.rows) return;
      const int D = p.head_dim;
      const int head = group / (D / 4);
      const int d0 = 2 * (group % (D / 4));
      const int half = D / 2;
      const int pos = p.row_pos[j];
      const int slot = p.row_slot[j];
      float a0 = s.y[0 * J + t], a1 = s.y[1 * J + t], b0 = s.y[2 * J + t], b1 = s.y[3 * J + t];
      if (kind != 2) {
        const int rp = STRIDE_BUG(p.l, 111) ? pos + 1 : pos;
        const float* cs = p.rope + (static_cast<int64_t>(rp) * half + d0) * 2;
        const float c0 = cs[0], s0 = cs[1], c1 = cs[2], s1 = cs[3];
        const float ra0 = a0 * c0 - b0 * s0;
        const float rb0 = b0 * c0 + a0 * s0;
        const float ra1 = a1 * c1 - b1 * s1;
        const float rb1 = STRIDE_BUG(p.l, 112) ? b1 * c1 - a1 * s1 : b1 * c1 + a1 * s1;
        a0 = ra0;
        b0 = rb0;
        a1 = ra1;
        b1 = rb1;
      }
      if (kind == 0) {
        float* q = p.q_out + static_cast<int64_t>(j) * p.q_stride + static_cast<int64_t>(head) * D;
        q[d0] = a0;
        q[d0 + 1] = a1;
        q[d0 + half] = b0;
        q[d0 + half + 1] = b1;
      } else {
        const int which = STRIDE_BUG(p.l, 113) ? 2 - kind : kind - 1;
        float* dst = kv_ptr<float>(p.kv, p.layer, which, head, slot, pos);
        dst[d0] = a0;
        dst[d0 + 1] = a1;
        dst[d0 + half] = b0;
        dst[d0 + half + 1] = b1;
      }
    }
  }
};

// FP32 dim-major attention output for the exact path.
struct LaneWriter {
  STRIDE_DEVICE static void write(const AttentionParams& p, int j, int idx, float value) {
    static_cast<float*>(p.out)[static_cast<int64_t>(idx) * p.l.stride + j] = value;
  }
};

// y = W x, then h += y. Used for the output projection and the MLP down projection.
template <int J>
struct GemmResidual {
  using Params = GemmResidualParams;
  static constexpr int kPhases = 2;
  struct Shared {
    float p[kBlockThreads][4 * J];
  };
  static int blocks(const Params& p) { return p.m.groups * p.l.lane_groups; }
  template <int P>
  STRIDE_DEVICE static void phase(const Params& p, int block, int t, Shared& s) {
    const int group = block / p.l.lane_groups;
    const int lane_group = block % p.l.lane_groups;
    if constexpr (P == 0) {
      float acc[4 * J];
      gemm_partial<J>(p.m.w + static_cast<int64_t>(group) * p.m.cols * 4,
                      J == 1 ? p.x : lane_base(p.x, lane_group), p.l.stride, p.m.cols, t, acc);
      for (int e = 0; e < 4 * J; ++e) s.p[t][e] = acc[e];
    } else if (t < 4 * J) {
      float y = s.p[0][t];
      if (STRIDE_BUG(p.l, 131)) {
        y = s.p[kBlockThreads - 1][t];
        for (int i = kBlockThreads - 2; i >= 0; --i) y = y + s.p[i][t];
      } else {
        for (int i = 1; i < kBlockThreads; ++i) y = y + s.p[i][t];
      }
      const int r = t / J;
      const int jj = t % J;
      float* h = p.h + static_cast<int64_t>(group * 4 + r) * p.l.stride + lane_group * 4 + jj;
      h[0] = STRIDE_BUG(p.l, 132) ? y : h[0] + y;
    }
  }
};

// Gate and up projections side by side, combined with SiLU, so neither
// intermediate vector is ever written to global memory.
template <int J>
struct Mlp13 {
  using Params = Mlp13Params;
  static constexpr int kPhases = 2;
  struct Shared {
    float a[kBlockThreads][4 * J];
    float b[kBlockThreads][4 * J];
  };
  static int blocks(const Params& p) { return p.w1.groups * p.l.lane_groups; }
  template <int P>
  STRIDE_DEVICE static void phase(const Params& p, int block, int t, Shared& s) {
    const int group = block / p.l.lane_groups;
    const int lane_group = block % p.l.lane_groups;
    if constexpr (P == 0) {
      const float* x = J == 1 ? p.x : lane_base(p.x, lane_group);
      float acc[4 * J];
      gemm_partial<J>(p.w1.w + static_cast<int64_t>(group) * p.w1.cols * 4, x, p.l.stride, p.w1.cols, t, acc);
      for (int e = 0; e < 4 * J; ++e) s.a[t][e] = acc[e];
      gemm_partial<J>(p.w3.w + static_cast<int64_t>(group) * p.w3.cols * 4, x, p.l.stride, p.w3.cols, t, acc);
      for (int e = 0; e < 4 * J; ++e) s.b[t][e] = acc[e];
    } else if (t < 4 * J) {
      float a = s.a[0][t];
      float b = s.b[0][t];
      for (int i = 1; i < kBlockThreads; ++i) {
        a = a + s.a[i][t];
        b = b + s.b[i][t];
      }
      const int r = t / J;
      const int jj = t % J;
      float* z = p.z + static_cast<int64_t>(group * 4 + r) * p.l.stride + lane_group * 4 + jj;
      const float gate = STRIDE_BUG(p.l, 141) ? a : det_silu(a);
      z[0] = STRIDE_BUG(p.l, 142) ? det_silu(b) * a : gate * b;
    }
  }
};

struct Compact {
  using Params = CompactParams;
  static constexpr int kPhases = 1;
  struct Shared {
    char unused;
  };
  static int blocks(const Params& p) { return (p.dim + kBlockThreads - 1) / kBlockThreads; }
  template <int P>
  STRIDE_DEVICE static void phase(const Params& p, int block, int t, Shared&) {
    const int k = block * kBlockThreads + t;
    if (k >= p.dim) return;
    for (int l = 0; l < p.lm_count; ++l) {
      p.out[static_cast<int64_t>(k) * p.lm_stride + l] = p.in[static_cast<int64_t>(k) * p.l.stride + p.lm_rows[l]];
    }
  }
};

// Output projection fused with argmax: logits are never materialized.
template <int J>
struct LmGreedy {
  using Params = LmGreedyParams;
  static constexpr int kPhases = 3;
  struct Shared {
    float p[kBlockThreads][4 * J];
    float y[4 * J];
  };
  static int blocks(const Params& p) { return p.m.groups * p.l.lane_groups; }
  template <int P>
  STRIDE_DEVICE static void phase(const Params& p, int block, int t, Shared& s) {
    const int group = block / p.l.lane_groups;
    const int lane_group = block % p.l.lane_groups;
    if constexpr (P == 0) {
      float acc[4 * J];
      gemm_partial<J>(p.m.w + static_cast<int64_t>(group) * p.m.cols * 4,
                      J == 1 ? p.x : lane_base(p.x, lane_group), p.l.stride, p.m.cols, t, acc);
      for (int e = 0; e < 4 * J; ++e) s.p[t][e] = acc[e];
    } else if constexpr (P == 1) {
      if (t < 4 * J) {
        float y = s.p[0][t];
        for (int i = 1; i < kBlockThreads; ++i) y = y + s.p[i][t];
        s.y[t] = y;
      }
    } else if (t < J) {
      const int lane = lane_group * 4 + t;
      if (lane >= p.l.rows) return;
      float best = as_f32(0xFF800000u);
      int32_t best_i = -1;
      for (int r = 0; r < 4; ++r) {
        const int index = group * 4 + r;
        if (index >= p.vocab) continue;
        const float v = s.y[r * J + t];
        const bool wins = STRIDE_BUG(p.l, 151) ? (v < best || best_i < 0) : v > best;
        if (wins) {
          best = v;
          best_i = index;
        }
      }
      if (best_i >= 0) best_update(&p.best[lane], best, best_i);
    }
  }
};

template <int J>
struct LmFull {
  using Params = LmFullParams;
  static constexpr int kPhases = 2;
  struct Shared {
    float p[kBlockThreads][4 * J];
  };
  static int blocks(const Params& p) { return p.m.groups * p.l.lane_groups; }
  template <int P>
  STRIDE_DEVICE static void phase(const Params& p, int block, int t, Shared& s) {
    const int group = block / p.l.lane_groups;
    const int lane_group = block % p.l.lane_groups;
    if constexpr (P == 0) {
      float acc[4 * J];
      gemm_partial<J>(p.m.w + static_cast<int64_t>(group) * p.m.cols * 4,
                      J == 1 ? p.x : lane_base(p.x, lane_group), p.l.stride, p.m.cols, t, acc);
      for (int e = 0; e < 4 * J; ++e) s.p[t][e] = acc[e];
    } else if (t < 4 * J) {
      float y = s.p[0][t];
      for (int i = 1; i < kBlockThreads; ++i) y = y + s.p[i][t];
      const int r = t / J;
      const int jj = t % J;
      p.logits[static_cast<int64_t>(group * 4 + r) * p.l.stride + lane_group * 4 + jj] = y;
    }
  }
};

}  // namespace kimpl

#if defined(__CUDACC__)
cudaStream_t& kernel_stream() {
  static cudaStream_t stream = nullptr;
  return stream;
}
#endif

using namespace kimpl;

void k_embed(const EmbedParams& p) { launch<Embed>(p); }
void k_norm(const NormParams& p, int J) {
  if (J == 1) launch<RmsNorm<1>>(p);
  else launch<RmsNorm<4>>(p);
}
void k_qkv(const QkvParams& p, int J) {
  if (J == 1) launch<Qkv<1>>(p);
  else launch<Qkv<4>>(p);
}
void k_attention(const AttentionParams& p) {
  if (p.head_dim <= kBlockThreads) {
    launch<AttentionPage<float, 1>>(p);
    launch<AttentionCombine<LaneWriter, 1>>(p);
  } else if (p.head_dim <= 2 * kBlockThreads) {
    launch<AttentionPage<float, 2>>(p);
    launch<AttentionCombine<LaneWriter, 2>>(p);
  } else {
    launch<AttentionPage<float, 4>>(p);
    launch<AttentionCombine<LaneWriter, 4>>(p);
  }
}
void k_gemm_residual(const GemmResidualParams& p, int J) {
  if (J == 1) launch<GemmResidual<1>>(p);
  else launch<GemmResidual<4>>(p);
}
void k_mlp13(const Mlp13Params& p, int J) {
  if (J == 1) launch<Mlp13<1>>(p);
  else launch<Mlp13<4>>(p);
}
void k_compact(const CompactParams& p) { launch<Compact>(p); }
void k_lm_greedy(const LmGreedyParams& p, int J) {
  if (J == 1) launch<LmGreedy<1>>(p);
  else launch<LmGreedy<4>>(p);
}
void k_lm_full(const LmFullParams& p, int J) {
  if (J == 1) launch<LmFull<1>>(p);
  else launch<LmFull<4>>(p);
}

}  // namespace stride
