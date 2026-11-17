// Device helpers and the kernel launcher shared by the exact and tensor-core paths.
#pragma once

#include "kernels.h"
#include "numerics.h"

#if !defined(__CUDACC__)
#include "cpu_pool.h"
#endif

namespace stride {

#if defined(__CUDACC__)
cudaStream_t& kernel_stream();
#endif

namespace kimpl {

template <class T>
STRIDE_DEVICE T* kv_ptr(const KvRef& kv, int layer, int which, int group, int slot, int pos) {
  const int block = kv.table[slot * kv.max_blocks + (pos >> kKvBlockShift)];
  const int token = pos & (kKvBlockTokens - 1);
  return static_cast<T*>(kv.arena) + block * kv.block_elems + layer * kv.layer_elems + which * kv.which_elems +
         group * kv.group_elems + token * kv.head_dim;
}

// Round to nearest, ties to even, from the FP32 bit pattern. Exact integer logic,
// so its result does not depend on the compiler or the device.
STRIDE_DEVICE uint32_t bf16_round_bits(float f) {
  const uint32_t u = as_u32(f);
  return (u + 0x7FFFu + ((u >> 16) & 1u)) >> 16;
}

STRIDE_DEVICE float bf16_bits_to_f32(uint32_t b) { return as_f32(b << 16); }

// Sixteen consecutive dims of a KV row as FP32. BF16 widening is exact.
STRIDE_DEVICE void kv_load_16(const float* row, int d, float* out) {
  const float4* v = reinterpret_cast<const float4*>(row + d);
  const float4 a = v[0], b = v[1], c = v[2], e = v[3];
  out[0] = a.x; out[1] = a.y; out[2] = a.z; out[3] = a.w;
  out[4] = b.x; out[5] = b.y; out[6] = b.z; out[7] = b.w;
  out[8] = c.x; out[9] = c.y; out[10] = c.z; out[11] = c.w;
  out[12] = e.x; out[13] = e.y; out[14] = e.z; out[15] = e.w;
}

STRIDE_DEVICE void kv_load_16(const uint16_t* row, int d, float* out) {
  const uint4* v = reinterpret_cast<const uint4*>(row + d);
  const uint4 a = v[0], b = v[1];
  out[0] = bf16_lo(a.x); out[1] = bf16_hi(a.x); out[2] = bf16_lo(a.y); out[3] = bf16_hi(a.y);
  out[4] = bf16_lo(a.z); out[5] = bf16_hi(a.z); out[6] = bf16_lo(a.w); out[7] = bf16_hi(a.w);
  out[8] = bf16_lo(b.x); out[9] = bf16_hi(b.x); out[10] = bf16_lo(b.y); out[11] = bf16_hi(b.y);
  out[12] = bf16_lo(b.z); out[13] = bf16_hi(b.z); out[14] = bf16_lo(b.w); out[15] = bf16_hi(b.w);
}

STRIDE_DEVICE float kv_widen(float v) { return v; }
STRIDE_DEVICE float kv_widen(uint16_t v) { return as_f32(static_cast<uint32_t>(v) << 16); }

STRIDE_DEVICE bool better_than(float v, int32_t i, float cv, int32_t ci) {
  return v > cv || (v == cv && i < ci);
}

// Keeps the lexicographic maximum of (value, -index). The result is independent
// of update order, so racing blocks still produce one deterministic answer.
STRIDE_DEVICE void best_update(BestEntry* slot, float value, int32_t index) {
  auto* word = reinterpret_cast<unsigned long long*>(slot);
  const unsigned long long want =
      (static_cast<unsigned long long>(static_cast<uint32_t>(index)) << 32) | as_u32(value);
  unsigned long long seen = *reinterpret_cast<volatile unsigned long long*>(word);
  for (;;) {
    const float cv = as_f32(static_cast<uint32_t>(seen));
    const int32_t ci = static_cast<int32_t>(seen >> 32);
    if (!better_than(value, index, cv, ci)) return;
#if defined(__CUDACC__)
    const unsigned long long prev = atomicCAS(word, seen, want);
#else
    unsigned long long prev = seen;
    __atomic_compare_exchange_n(word, &prev, want, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
#endif
    if (prev == seen) return;
    seen = prev;
  }
}

// Attention over one 64-position KV page for one (row, KV group). A fixed tree
// over pages replaces one long chain: each page is summed on its own and pages are
// combined in order, so more blocks can read KV at once while the result still
// depends only on positions, never on the batch or on where the pages live. The
// query heads sharing a KV group read each K and V row once.
// PT is dims per thread in the value mix (head_dim / 32, at least 1).
template <class KvT, int PT>
struct AttentionPage {
  using Params = AttentionParams;
  static constexpr int kPhases = 6;
  static constexpr int kInFlight = 8;
  // Four query heads share a KV group. Per-dim and per-position values sit in one
  // float4, so a thread makes one shared load per four FMAs instead of four.
  struct Shared {
    float4 q[kMaxHeadDim];
    float4 s[kKvBlockTokens];
    float4 m[kBlockThreads];
    float4 mm;
    float4 denom;
  };
  static int blocks(const Params& p) { return p.item_count * p.n_kv; }

  STRIDE_DEVICE static void load_dims(const KvT* src, float* out) {
    if constexpr (PT == 2 && sizeof(KvT) == 2) {
      const uint32_t v = *reinterpret_cast<const uint32_t*>(src);
      out[0] = bf16_lo(v);
      out[1] = bf16_hi(v);
    } else {
      for (int dd = 0; dd < PT; ++dd) out[dd] = kv_widen(src[dd]);
    }
  }

  template <int P>
  STRIDE_DEVICE static void phase(const Params& p, int block, int t, Shared& s) {
    const int item = block / p.n_kv;
    const int g = block % p.n_kv;
    const int j = p.items[item * 2];
    const int page = p.items[item * 2 + 1];
    const int slot = p.row_slot[j];
    const int visible = (STRIDE_BUG(p.l, 121) ? p.row_pos[j] : p.row_pos[j] + 1);
    const int p0 = page << kKvBlockShift;
    const int n = visible - p0 < kKvBlockTokens ? visible - p0 : kKvBlockTokens;
    const int D = p.head_dim;
    if constexpr (P == 0) {
      const float* qrow = p.q + static_cast<int64_t>(j) * p.q_stride + static_cast<int64_t>(g * 4) * D;
      for (int d = t; d < D; d += kBlockThreads) {
        float4 v;
        v.x = qrow[d];
        v.y = qrow[D + d];
        v.z = qrow[2 * D + d];
        v.w = qrow[3 * D + d];
        s.q[d] = v;
      }
    } else if constexpr (P == 1) {
      const float neg_inf = as_f32(0xFF800000u);
      float l0 = neg_inf, l1 = neg_inf, l2 = neg_inf, l3 = neg_inf;
      if (n > 0) {
        const KvT* keys = kv_ptr<KvT>(p.kv, p.layer, 0, g, slot, p0);
        const float scale = STRIDE_BUG(p.l, 122) ? 1.0f : p.scale;
        for (int i = t; i < n; i += kBlockThreads) {
          const KvT* key = keys + static_cast<int64_t>(i) * D;
          float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
          for (int d = 0; d < D; d += 16) {
            float k[16];
            kv_load_16(key, d, k);
            for (int e = 0; e < 16; ++e) {
              const float4 q = s.q[d + e];
              a0 = fmaf(q.x, k[e], a0);
              a1 = fmaf(q.y, k[e], a1);
              a2 = fmaf(q.z, k[e], a2);
              a3 = fmaf(q.w, k[e], a3);
            }
          }
          float4 score;
          score.x = a0 * scale;
          score.y = a1 * scale;
          score.z = a2 * scale;
          score.w = a3 * scale;
          s.s[i] = score;
          if (score.x > l0) l0 = score.x;
          if (score.y > l1) l1 = score.y;
          if (score.z > l2) l2 = score.z;
          if (score.w > l3) l3 = score.w;
        }
      }
      float4 local;
      local.x = l0;
      local.y = l1;
      local.z = l2;
      local.w = l3;
      s.m[t] = local;
    } else if constexpr (P == 2) {
      if (t < 4) {
        float mm = (&s.m[0].x)[t];
        for (int i = 1; i < kBlockThreads; ++i) {
          const float v = (&s.m[i].x)[t];
          if (v > mm) mm = v;
        }
        (&s.mm.x)[t] = mm;
      }
    } else if constexpr (P == 3) {
      const float4 mm = s.mm;
      for (int i = t; i < n; i += kBlockThreads) {
        float4 v = s.s[i];
        v.x = det_exp(v.x - mm.x);
        v.y = det_exp(v.y - mm.y);
        v.z = det_exp(v.z - mm.z);
        v.w = det_exp(v.w - mm.w);
        s.s[i] = v;
      }
    } else if constexpr (P == 4) {
      if (t < 4) {
        float denom = 0.0f;
        if (STRIDE_BUG(p.l, 123)) {
          for (int i = n - 1; i >= 0; --i) denom = denom + (&s.s[i].x)[t];
        } else {
          for (int i = 0; i < n; ++i) denom = denom + (&s.s[i].x)[t];
        }
        (&s.denom.x)[t] = denom;
      }
    } else {
      float* rec = p.scratch + (static_cast<int64_t>(j * p.n_kv + g) * p.max_pages + page) * p.rec_floats;
      float* v_out = rec + 8;
      if (t < 4) {
        rec[t] = (&s.mm.x)[t];
        rec[4 + t] = (&s.denom.x)[t];
      }
      const int d0 = t * PT;
      if (d0 >= D) return;
      float a0[PT], a1[PT], a2[PT], a3[PT];
      for (int dd = 0; dd < PT; ++dd) a0[dd] = a1[dd] = a2[dd] = a3[dd] = 0.0f;
      if (n > 0) {
        const int which = STRIDE_BUG(p.l, 124) ? 0 : 1;
        const int vp = (STRIDE_BUG(p.l, 125) && page > 0) ? p0 - kKvBlockTokens : p0;
        const KvT* value = kv_ptr<KvT>(p.kv, p.layer, which, g, slot, vp) + d0;
        int i = 0;
        for (; i + kInFlight <= n; i += kInFlight) {
          float v[kInFlight][PT];
          for (int b = 0; b < kInFlight; ++b) load_dims(value + static_cast<int64_t>(i + b) * D, v[b]);
          for (int b = 0; b < kInFlight; ++b) {
            const float4 w = s.s[i + b];
            for (int dd = 0; dd < PT; ++dd) {
              a0[dd] = fmaf(w.x, v[b][dd], a0[dd]);
              a1[dd] = fmaf(w.y, v[b][dd], a1[dd]);
              a2[dd] = fmaf(w.z, v[b][dd], a2[dd]);
              a3[dd] = fmaf(w.w, v[b][dd], a3[dd]);
            }
          }
        }
        for (; i < n; ++i) {
          float v[PT];
          load_dims(value + static_cast<int64_t>(i) * D, v);
          const float4 w = s.s[i];
          for (int dd = 0; dd < PT; ++dd) {
            a0[dd] = fmaf(w.x, v[dd], a0[dd]);
            a1[dd] = fmaf(w.y, v[dd], a1[dd]);
            a2[dd] = fmaf(w.z, v[dd], a2[dd]);
            a3[dd] = fmaf(w.w, v[dd], a3[dd]);
          }
        }
      }
      for (int dd = 0; dd < PT; ++dd) {
        v_out[d0 + dd] = a0[dd];
        v_out[D + d0 + dd] = a1[dd];
        v_out[2 * D + d0 + dd] = a2[dd];
        v_out[3 * D + d0 + dd] = a3[dd];
      }
    }
  }
};

// Combines page results in ascending page order and writes the attention output.
template <class Writer, int PT>
struct AttentionCombine {
  using Params = AttentionParams;
  static constexpr int kPhases = 2;
  struct Shared {
    float f[4][kMaxCtx >> kKvBlockShift];
    float denom[4];
  };
  static int blocks(const Params& p) { return p.l.rows * p.n_kv; }
  template <int P>
  STRIDE_DEVICE static void phase(const Params& p, int block, int t, Shared& s) {
    const int j = block / p.n_kv;
    const int g = block % p.n_kv;
    const int pages = (p.row_pos[j] >> kKvBlockShift) + 1;
    const int D = p.head_dim;
    const float* recs = p.scratch + static_cast<int64_t>(j * p.n_kv + g) * p.max_pages * p.rec_floats;
    if constexpr (P == 0) {
      if (t < 4) {
        float m = as_f32(0xFF800000u);
        for (int c = 0; c < pages; ++c) {
          const float mc = recs[static_cast<int64_t>(c) * p.rec_floats + t];
          if (mc > m) m = mc;
        }
        float denom = 0.0f;
        for (int c = 0; c < pages; ++c) {
          const float* rec = recs + static_cast<int64_t>(c) * p.rec_floats;
          const float f = det_exp(rec[t] - m);
          s.f[t][c] = f;
          denom = fmaf(f, rec[4 + t], denom);
        }
        s.denom[t] = denom;
      }
    } else {
      const int d0 = t * PT;
      if (d0 >= D) return;
      float acc[4][PT];
      for (int q = 0; q < p.n_q; ++q)
        for (int dd = 0; dd < PT; ++dd) acc[q][dd] = 0.0f;
      for (int c = 0; c < pages; ++c) {
        const float* v = recs + static_cast<int64_t>(c) * p.rec_floats + 8;
        for (int q = 0; q < p.n_q; ++q) {
          const float f = s.f[q][c];
          for (int dd = 0; dd < PT; ++dd) acc[q][dd] = fmaf(f, v[q * D + d0 + dd], acc[q][dd]);
        }
      }
      for (int q = 0; q < p.n_q; ++q) {
        const int head = STRIDE_BUG(p.l, 126) ? (g * p.n_q + (p.n_q - 1 - q)) : (g * p.n_q + q);
        for (int dd = 0; dd < PT; ++dd) {
          Writer::write(p, j, head * D + d0 + dd, acc[q][dd] / s.denom[q]);
        }
      }
    }
  }
};

// Threads per block; a kernel overrides the default with kBlockSize.
template <class K, class = void>
struct BlockSize {
  static constexpr int value = kBlockThreads;
};
template <class K>
struct BlockSize<K, decltype(void(K::kBlockSize))> {
  static constexpr int value = K::kBlockSize;
};

#if defined(__CUDACC__)
template <class K>
__global__ void kernel_entry(typename K::Params p) {
  __shared__ typename K::Shared s;
  const int b = blockIdx.x;
  const int t = threadIdx.x;
  K::template phase<0>(p, b, t, s);
  if constexpr (K::kPhases > 1) {
    __syncthreads();
    K::template phase<1>(p, b, t, s);
  }
  if constexpr (K::kPhases > 2) {
    __syncthreads();
    K::template phase<2>(p, b, t, s);
  }
  if constexpr (K::kPhases > 3) {
    __syncthreads();
    K::template phase<3>(p, b, t, s);
    __syncthreads();
    K::template phase<4>(p, b, t, s);
    __syncthreads();
    K::template phase<5>(p, b, t, s);
  }
}

template <class K>
void launch(const typename K::Params& p) {
  const int blocks = K::blocks(p);
  if (blocks <= 0) return;
  kernel_entry<K><<<blocks, BlockSize<K>::value, 0, kernel_stream()>>>(p);
}
#else
template <class K>
void launch(const typename K::Params& p) {
  const int blocks = K::blocks(p);
  if (blocks <= 0) return;
  cpu_parallel_for(blocks, [&p](int b) {
    auto shared = new typename K::Shared;
    for (int t = 0; t < kBlockThreads; ++t) K::template phase<0>(p, b, t, *shared);
    if constexpr (K::kPhases > 1)
      for (int t = 0; t < kBlockThreads; ++t) K::template phase<1>(p, b, t, *shared);
    if constexpr (K::kPhases > 2)
      for (int t = 0; t < kBlockThreads; ++t) K::template phase<2>(p, b, t, *shared);
    if constexpr (K::kPhases > 3) {
      for (int t = 0; t < kBlockThreads; ++t) K::template phase<3>(p, b, t, *shared);
      for (int t = 0; t < kBlockThreads; ++t) K::template phase<4>(p, b, t, *shared);
      for (int t = 0; t < kBlockThreads; ++t) K::template phase<5>(p, b, t, *shared);
    }
    delete shared;
  });
}
#endif

}  // namespace kimpl
}  // namespace stride
