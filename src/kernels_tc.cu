// Tensor-core path: the same model math, computed with 16x16x16 mma instructions
// on BF16 tiles. This is what makes large batches fast. tools/tc_probe.cu showed an
// mma result for a (row, lane) is bit-identical wherever that row and lane sit in
// the tile, so a chain of mma operations in ascending column-tile order stays
// batch-invariant. Unlike the exact path it does not reproduce on a CPU, because the
// hardware's internal summation is not a simple IEEE sequence.
//
// Layouts:
//   weights   W[row tile][col tile][16 rows][16 cols], BF16
//   x tiles   X[lane tile][col tile][16 lanes][16 cols], BF16
//   h tiles   same layout as x, FP32
#include <cuda_bf16.h>
#include <mma.h>

#include "kernel_common.h"
#include "kernels_tc.h"

namespace stride {
namespace kimpl {

using namespace nvcuda;
using Bf16 = __nv_bfloat16;
using FragA = wmma::fragment<wmma::matrix_a, 16, 16, 16, Bf16, wmma::row_major>;
using FragB = wmma::fragment<wmma::matrix_b, 16, 16, 16, Bf16, wmma::col_major>;
using FragC = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

constexpr int kTile = 256;  // elements in one 16x16 tile

STRIDE_DEVICE const Bf16* as_bf16(const uint16_t* p) { return reinterpret_cast<const Bf16*>(p); }

STRIDE_DEVICE int tile_parts(int lane_tiles, int lt_per_block) { return (lane_tiles + lt_per_block - 1) / lt_per_block; }
STRIDE_HD int tile_parts_hd(int lane_tiles, int lt_per_block) { return (lane_tiles + lt_per_block - 1) / lt_per_block; }

// Chains A*B + C across column tiles for LT lane tiles that share weight loads.
// Weights for the next four tiles are in flight before the first mma uses them.
template <int LT>
STRIDE_DEVICE void tile_chain(const uint16_t* wrow, const uint16_t* x, int col_tiles, int lt0, int lane_tiles, FragC* c) {
  const Bf16* w = as_bf16(wrow);
  const Bf16* xb = as_bf16(x);
  int ct = 0;
  for (; ct + 4 <= col_tiles; ct += 4) {
    FragA a0, a1, a2, a3;
    wmma::load_matrix_sync(a0, w + static_cast<int64_t>(ct) * kTile, 16);
    wmma::load_matrix_sync(a1, w + static_cast<int64_t>(ct + 1) * kTile, 16);
    wmma::load_matrix_sync(a2, w + static_cast<int64_t>(ct + 2) * kTile, 16);
    wmma::load_matrix_sync(a3, w + static_cast<int64_t>(ct + 3) * kTile, 16);
    for (int l = 0; l < LT; ++l) {
      const int lt = lt0 + l;
      if (lt >= lane_tiles) break;
      const Bf16* xl = xb + (static_cast<int64_t>(lt) * col_tiles + ct) * kTile;
      FragB b0, b1, b2, b3;
      wmma::load_matrix_sync(b0, xl, 16);
      wmma::load_matrix_sync(b1, xl + kTile, 16);
      wmma::load_matrix_sync(b2, xl + 2 * kTile, 16);
      wmma::load_matrix_sync(b3, xl + 3 * kTile, 16);
      wmma::mma_sync(c[l], a0, b0, c[l]);
      wmma::mma_sync(c[l], a1, b1, c[l]);
      wmma::mma_sync(c[l], a2, b2, c[l]);
      wmma::mma_sync(c[l], a3, b3, c[l]);
    }
  }
  for (; ct < col_tiles; ++ct) {
    FragA a;
    wmma::load_matrix_sync(a, w + static_cast<int64_t>(ct) * kTile, 16);
    for (int l = 0; l < LT; ++l) {
      const int lt = lt0 + l;
      if (lt >= lane_tiles) break;
      FragB b;
      wmma::load_matrix_sync(b, xb + (static_cast<int64_t>(lt) * col_tiles + ct) * kTile, 16);
      wmma::mma_sync(c[l], a, b, c[l]);
    }
  }
}

struct TcEmbed {
  using Params = TcEmbedParams;
  static constexpr int kPhases = 1;
  struct Shared {
    char unused;
  };
  static int blocks(const Params& p) { return p.l.rows; }
  template <int P>
  STRIDE_DEVICE static void phase(const Params& p, int block, int t, Shared&) {
    const int token = p.tokens[block];
    const int rt = token / 16;
    const int slot = token % 16;
    const int lt = block / 16;
    const int jj = block % 16;
    const int col_tiles = p.dim / 16;
    const uint16_t* row = p.embed + static_cast<int64_t>(rt) * col_tiles * kTile;
    for (int ct = t; ct < col_tiles; ct += kBlockThreads) {
      const uint16_t* w = row + static_cast<int64_t>(ct) * kTile + slot * 16;
      float* h = p.h + ((static_cast<int64_t>(lt) * col_tiles + ct) * 16 + jj) * 16;
      for (int c = 0; c < 16; ++c) {
        float v = as_f32(static_cast<uint32_t>(w[c]) << 16);
        if (STRIDE_BUG(p.l, 101) && ct == 0 && c == 0) v = -v;
        h[c] = v;
      }
    }
  }
};

// RMSNorm from FP32 h to BF16 x. Thread t owns column tiles == t (mod 32).
struct TcNorm {
  using Params = TcNormParams;
  static constexpr int kPhases = 3;
  struct Shared {
    float partial[kBlockThreads][16];
    float r[16];
  };
  static int blocks(const Params& p) { return p.l.lane_groups; }
  template <int P>
  STRIDE_DEVICE static void phase(const Params& p, int block, int t, Shared& s) {
    const int col_tiles = p.dim / 16;
    const int lanes = p.l.rows - block * 16 < 16 ? p.l.rows - block * 16 : 16;
    if constexpr (P == 0) {
      float acc[16];
      for (int j = 0; j < 16; ++j) acc[j] = 0.0f;
      for (int ct = t; ct < col_tiles; ct += kBlockThreads) {
        const float* tile = p.h + (static_cast<int64_t>(block) * col_tiles + ct) * kTile;
        for (int j = 0; j < lanes; ++j) {
          float v[16];
          kv_load_16(tile + j * 16, 0, v);
          for (int c = 0; c < 16; ++c) acc[j] = fmaf(v[c], v[c], acc[j]);
        }
      }
      for (int j = 0; j < 16; ++j) s.partial[t][j] = acc[j];
    } else if constexpr (P == 1) {
      if (t < lanes) {
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
      for (int ct = t; ct < col_tiles; ct += kBlockThreads) {
        const float* tile = p.h + (static_cast<int64_t>(block) * col_tiles + ct) * kTile;
        uint4* out = reinterpret_cast<uint4*>(p.x + (static_cast<int64_t>(block) * col_tiles + ct) * kTile);
        float g[16];
        kv_load_16(p.weight, ct * 16, g);
        if (STRIDE_BUG(p.l, 105) && ct == 0) g[0] = 1.0f;
        for (int j = 0; j < 16; ++j) {
          uint4 o0 = {0, 0, 0, 0}, o1 = {0, 0, 0, 0};
          if (j < lanes) {
            float v[16];
            kv_load_16(tile + j * 16, 0, v);
            const float r = s.r[j];
            uint32_t b[16];
            for (int c = 0; c < 16; ++c) b[c] = bf16_round_bits(g[c] * (v[c] * r));
            o0.x = b[0] | (b[1] << 16);
            o0.y = b[2] | (b[3] << 16);
            o0.z = b[4] | (b[5] << 16);
            o0.w = b[6] | (b[7] << 16);
            o1.x = b[8] | (b[9] << 16);
            o1.y = b[10] | (b[11] << 16);
            o1.z = b[12] | (b[13] << 16);
            o1.w = b[14] | (b[15] << 16);
          }
          out[j * 2] = o0;
          out[j * 2 + 1] = o1;
        }
      }
    }
  }
};

// Q, K, V projections fused with RoPE and the KV append. Each row tile holds eight
// RoPE pairs: rows (d0 + i) and (d0 + D/2 + i) for i in 0..7.
template <int LT>
struct TcQkv {
  using Params = TcQkvParams;
  static constexpr int kPhases = 1;
  struct Shared {
    float c[LT][kTile];
  };
  static int blocks(const Params& p) {
    return (p.q.groups + 2 * p.k.groups) * tile_parts_hd(p.l.lane_groups, LT);
  }
  template <int P>
  STRIDE_DEVICE static void phase(const Params& p, int block, int t, Shared& s) {
    const int parts = tile_parts(p.l.lane_groups, LT);
    const int index = block / parts;
    const int lt0 = (block % parts) * LT;
    int kind, rt;
    const GemmMat* m;
    if (index < p.q.groups) {
      kind = 0;
      rt = index;
      m = &p.q;
    } else if (index < p.q.groups + p.k.groups) {
      kind = 1;
      rt = index - p.q.groups;
      m = &p.k;
    } else {
      kind = 2;
      rt = index - p.q.groups - p.k.groups;
      m = &p.v;
    }
    const int col_tiles = m->cols / 16;
    FragC c[LT];
    for (int l = 0; l < LT; ++l) wmma::fill_fragment(c[l], 0.0f);
    tile_chain<LT>(m->w + static_cast<int64_t>(rt) * col_tiles * kTile, p.x, col_tiles, lt0, p.l.lane_groups, c);
    for (int l = 0; l < LT; ++l) wmma::store_matrix_sync(s.c[l], c[l], 16, wmma::mem_row_major);
    __syncwarp();

    const int D = p.head_dim;
    const int tiles_per_head = D / 16;
    const int head = rt / tiles_per_head;
    const int d0 = (rt % tiles_per_head) * 8;
    const int half = D / 2;
    const int pair = t % 8;
    for (int l = 0; l < LT; ++l) {
      const int lt = lt0 + l;
      for (int n = 0; n < 4; ++n) {
        const int jj = (t / 8) * 4 + n;
        const int j = lt * 16 + jj;
        if (j >= p.l.rows) continue;
        const int pos = p.row_pos[j];
        const int slot = p.row_slot[j];
        const float a = s.c[l][pair * 16 + jj];
        const float b = s.c[l][(8 + pair) * 16 + jj];
        const int d = d0 + pair;
        if (kind == 2) {
          const int which = STRIDE_BUG(p.l, 113) ? 0 : 1;
          uint16_t* dst = kv_ptr<uint16_t>(p.kv, p.layer, which, head, slot, pos);
          dst[d] = static_cast<uint16_t>(bf16_round_bits(a));
          dst[d + half] = static_cast<uint16_t>(bf16_round_bits(b));
        } else {
          const int rp = STRIDE_BUG(p.l, 111) ? pos + 1 : pos;
          const float* cs = p.rope + (static_cast<int64_t>(rp) * half + d) * 2;
          const float co = cs[0], si = cs[1];
          const float ra = a * co - b * si;
          const float rb = STRIDE_BUG(p.l, 112) ? b * co - a * si : b * co + a * si;
          if (kind == 0) {
            float* q = p.q_out + static_cast<int64_t>(j) * p.q_stride + static_cast<int64_t>(head) * D;
            q[d] = ra;
            q[d + half] = rb;
          } else {
            const int which = STRIDE_BUG(p.l, 113) ? 1 : 0;
            uint16_t* dst = kv_ptr<uint16_t>(p.kv, p.layer, which, head, slot, pos);
            dst[d] = static_cast<uint16_t>(bf16_round_bits(ra));
            dst[d + half] = static_cast<uint16_t>(bf16_round_bits(rb));
          }
        }
      }
    }
  }
};

// h += W x with h as the initial accumulator: the residual costs no extra pass and
// no intermediate vector.
template <int LT>
struct TcGemmResidual {
  using Params = TcGemmResidualParams;
  static constexpr int kPhases = 1;
  struct Shared {
    char unused;
  };
  static int blocks(const Params& p) { return p.m.groups * tile_parts_hd(p.l.lane_groups, LT); }
  template <int P>
  STRIDE_DEVICE static void phase(const Params& p, int block, int, Shared&) {
    const int parts = tile_parts(p.l.lane_groups, LT);
    const int rt = block / parts;
    const int lt0 = (block % parts) * LT;
    const int col_tiles = p.m.cols / 16;
    const int h_tiles = p.m.groups;
    FragC c[LT];
    for (int l = 0; l < LT; ++l) {
      const int lt = lt0 + l;
      if (lt < p.l.lane_groups && !STRIDE_BUG(p.l, 132)) {
        wmma::load_matrix_sync(c[l], p.h + (static_cast<int64_t>(lt) * h_tiles + rt) * kTile, 16, wmma::mem_col_major);
      } else {
        wmma::fill_fragment(c[l], 0.0f);
      }
    }
    tile_chain<LT>(p.m.w + static_cast<int64_t>(rt) * col_tiles * kTile, p.x, col_tiles, lt0, p.l.lane_groups, c);
    for (int l = 0; l < LT; ++l) {
      const int lt = lt0 + l;
      if (lt >= p.l.lane_groups) break;
      wmma::store_matrix_sync(p.h + (static_cast<int64_t>(lt) * h_tiles + rt) * kTile, c[l], 16, wmma::mem_col_major);
    }
  }
};

// Gate and up projections combined with SiLU in the same warp.
template <int LT>
struct TcMlp13 {
  using Params = TcMlp13Params;
  static constexpr int kPhases = 1;
  struct Shared {
    float a[LT][kTile];
    float b[LT][kTile];
  };
  static int blocks(const Params& p) { return p.w1.groups * tile_parts_hd(p.l.lane_groups, LT); }
  template <int P>
  STRIDE_DEVICE static void phase(const Params& p, int block, int t, Shared& s) {
    const int parts = tile_parts(p.l.lane_groups, LT);
    const int rt = block / parts;
    const int lt0 = (block % parts) * LT;
    const int col_tiles = p.w1.cols / 16;
    const int z_tiles = p.w1.groups;
    FragC c1[LT], c3[LT];
    for (int l = 0; l < LT; ++l) {
      wmma::fill_fragment(c1[l], 0.0f);
      wmma::fill_fragment(c3[l], 0.0f);
    }
    tile_chain<LT>(p.w1.w + static_cast<int64_t>(rt) * col_tiles * kTile, p.x, col_tiles, lt0, p.l.lane_groups, c1);
    tile_chain<LT>(p.w3.w + static_cast<int64_t>(rt) * col_tiles * kTile, p.x, col_tiles, lt0, p.l.lane_groups, c3);
    for (int l = 0; l < LT; ++l) {
      wmma::store_matrix_sync(s.a[l], c1[l], 16, wmma::mem_col_major);
      wmma::store_matrix_sync(s.b[l], c3[l], 16, wmma::mem_col_major);
    }
    __syncwarp();
    for (int l = 0; l < LT; ++l) {
      const int lt = lt0 + l;
      if (lt >= p.l.lane_groups) break;
      uint32_t bits[8];
      for (int e = 0; e < 8; ++e) {
        const int i = t * 8 + e;
        const float a = s.a[l][i];
        const float b = s.b[l][i];
        const float gate = STRIDE_BUG(p.l, 141) ? a : det_silu(a);
        const float v = STRIDE_BUG(p.l, 142) ? det_silu(b) * a : gate * b;
        bits[e] = bf16_round_bits(v);
      }
      uint4 o;
      o.x = bits[0] | (bits[1] << 16);
      o.y = bits[2] | (bits[3] << 16);
      o.z = bits[4] | (bits[5] << 16);
      o.w = bits[6] | (bits[7] << 16);
      uint4* z = reinterpret_cast<uint4*>(p.z + (static_cast<int64_t>(lt) * z_tiles + rt) * kTile);
      z[t] = o;
    }
  }
};

struct TcCompact {
  using Params = TcCompactParams;
  static constexpr int kPhases = 1;
  struct Shared {
    char unused;
  };
  static int blocks(const Params& p) { return p.lm_count; }
  template <int P>
  STRIDE_DEVICE static void phase(const Params& p, int block, int t, Shared&) {
    const int src = p.lm_rows[block];
    const int src_lt = src / 16, src_j = src % 16;
    const int dst_lt = block / 16, dst_j = block % 16;
    for (int ct = t; ct < p.col_tiles; ct += kBlockThreads) {
      const uint4* in = reinterpret_cast<const uint4*>(
          p.in + ((static_cast<int64_t>(src_lt) * p.col_tiles + ct) * 16 + src_j) * 16);
      uint4* out = reinterpret_cast<uint4*>(p.out + ((static_cast<int64_t>(dst_lt) * p.col_tiles + ct) * 16 + dst_j) * 16);
      const uint4 a = in[0], b = in[1];
      out[0] = a;
      out[1] = b;
    }
  }
};

// Output projection fused with argmax; logits are never materialized.
template <int LT>
struct TcLmGreedy {
  using Params = TcLmGreedyParams;
  static constexpr int kPhases = 1;
  struct Shared {
    float c[LT][kTile];
  };
  static int blocks(const Params& p) { return p.m.groups * tile_parts_hd(p.l.lane_groups, LT); }
  template <int P>
  STRIDE_DEVICE static void phase(const Params& p, int block, int t, Shared& s) {
    const int parts = tile_parts(p.l.lane_groups, LT);
    const int rt = block / parts;
    const int lt0 = (block % parts) * LT;
    const int col_tiles = p.m.cols / 16;
    FragC c[LT];
    for (int l = 0; l < LT; ++l) wmma::fill_fragment(c[l], 0.0f);
    tile_chain<LT>(p.m.w + static_cast<int64_t>(rt) * col_tiles * kTile, p.x, col_tiles, lt0, p.l.lane_groups, c);
    for (int l = 0; l < LT; ++l) wmma::store_matrix_sync(s.c[l], c[l], 16, wmma::mem_row_major);
    __syncwarp();
    if (t >= 16) return;
    for (int l = 0; l < LT; ++l) {
      const int lane = (lt0 + l) * 16 + t;
      if (lane >= p.l.rows) continue;
      float best = as_f32(0xFF800000u);
      int32_t best_i = -1;
      for (int r = 0; r < 16; ++r) {
        const int index = rt * 16 + r;
        if (index >= p.vocab) continue;
        const float v = s.c[l][r * 16 + t];
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

template <int LT>
struct TcLmFull {
  using Params = TcLmFullParams;
  static constexpr int kPhases = 1;
  struct Shared {
    char unused;
  };
  static int blocks(const Params& p) { return p.m.groups * tile_parts_hd(p.l.lane_groups, LT); }
  template <int P>
  STRIDE_DEVICE static void phase(const Params& p, int block, int, Shared&) {
    const int parts = tile_parts(p.l.lane_groups, LT);
    const int rt = block / parts;
    const int lt0 = (block % parts) * LT;
    const int col_tiles = p.m.cols / 16;
    FragC c[LT];
    for (int l = 0; l < LT; ++l) wmma::fill_fragment(c[l], 0.0f);
    tile_chain<LT>(p.m.w + static_cast<int64_t>(rt) * col_tiles * kTile, p.x, col_tiles, lt0, p.l.lane_groups, c);
    for (int l = 0; l < LT; ++l) {
      const int lt = lt0 + l;
      if (lt >= p.l.lane_groups) break;
      wmma::store_matrix_sync(p.logits + (static_cast<int64_t>(rt) * p.l.lane_groups + lt) * kTile, c[l], 16, wmma::mem_row_major);
    }
  }
};

struct TileWriter {
  STRIDE_DEVICE static void write(const AttentionParams& p, int j, int idx, float value) {
    const int lt = j / 16;
    const int jj = j % 16;
    const int col_tiles = p.out_cols / 16;
    uint16_t* out = reinterpret_cast<uint16_t*>(p.out);
    out[((static_cast<int64_t>(lt) * col_tiles + idx / 16) * 16 + jj) * 16 + idx % 16] =
        static_cast<uint16_t>(bf16_round_bits(value));
  }
};

}  // namespace kimpl

using namespace kimpl;

void tc_embed(const TcEmbedParams& p) { launch<TcEmbed>(p); }
void tc_norm(const TcNormParams& p) { launch<TcNorm>(p); }
void tc_qkv(const TcQkvParams& p, int lt) {
  if (lt == 1) launch<TcQkv<1>>(p);
  else launch<TcQkv<2>>(p);
}
void tc_attention(const AttentionParams& p) {
  if (p.head_dim <= kBlockThreads) {
    launch<AttentionPage<uint16_t, 1>>(p);
    launch<AttentionCombine<TileWriter, 1>>(p);
  } else if (p.head_dim <= 2 * kBlockThreads) {
    launch<AttentionPage<uint16_t, 2>>(p);
    launch<AttentionCombine<TileWriter, 2>>(p);
  } else {
    launch<AttentionPage<uint16_t, 4>>(p);
    launch<AttentionCombine<TileWriter, 4>>(p);
  }
}
void tc_gemm_residual(const TcGemmResidualParams& p, int lt) {
  if (lt == 1) launch<TcGemmResidual<1>>(p);
  else launch<TcGemmResidual<2>>(p);
}
void tc_mlp13(const TcMlp13Params& p, int lt) {
  if (lt == 1) launch<TcMlp13<1>>(p);
  else launch<TcMlp13<2>>(p);
}
void tc_compact(const TcCompactParams& p) { launch<TcCompact>(p); }
void tc_lm_greedy(const TcLmGreedyParams& p, int lt) {
  if (lt == 1) launch<TcLmGreedy<1>>(p);
  else launch<TcLmGreedy<2>>(p);
}
void tc_lm_full(const TcLmFullParams& p, int lt) {
  if (lt == 1) launch<TcLmFull<1>>(p);
  else launch<TcLmFull<2>>(p);
}

}  // namespace stride
