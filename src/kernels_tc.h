// Parameter blocks and entry points for the tensor-core path (CUDA only).
// In these structs Lanes.lane_groups holds the number of 16-lane tiles, and
// GemmMat.groups holds the number of 16-row tiles.
#pragma once

#include "kernels.h"

namespace stride {

struct TcEmbedParams {
  Lanes l;
  const uint16_t* embed;
  const int32_t* tokens;
  float* h;
  int32_t dim;
};

struct TcNormParams {
  Lanes l;
  const float* h;
  uint16_t* x;
  const float* weight;
  int32_t dim;
  float eps;
};

struct TcQkvParams {
  Lanes l;
  const uint16_t* x;
  GemmMat q, k, v;
  float* q_out;
  int32_t q_stride;
  KvRef kv;
  const float* rope;
  const int32_t* row_slot;
  const int32_t* row_pos;
  int32_t layer;
  int32_t head_dim;
};

struct TcGemmResidualParams {
  Lanes l;
  const uint16_t* x;
  GemmMat m;
  float* h;
};

struct TcMlp13Params {
  Lanes l;
  const uint16_t* x;
  GemmMat w1, w3;
  uint16_t* z;
};

struct TcCompactParams {
  Lanes l;
  const uint16_t* in;
  uint16_t* out;
  const int32_t* lm_rows;
  int32_t lm_count;
  int32_t col_tiles;
};

struct TcLmGreedyParams {
  Lanes l;
  const uint16_t* x;
  GemmMat m;
  int32_t vocab;
  BestEntry* best;
};

struct TcLmFullParams {
  Lanes l;
  const uint16_t* x;
  GemmMat m;
  float* logits;
};

void tc_embed(const TcEmbedParams& p);
void tc_norm(const TcNormParams& p, bool per_lane);
void tc_qkv(const TcQkvParams& p);
void tc_attention(const AttentionParams& p);
void tc_gemm_residual(const TcGemmResidualParams& p, int row_tiles_per_warp, int lane_tiles_per_warp);
void tc_mlp13(const TcMlp13Params& p, int row_tiles_per_warp, int lane_tiles_per_warp);
void tc_compact(const TcCompactParams& p);
void tc_lm_greedy(const TcLmGreedyParams& p, int row_tiles_per_warp, int lane_tiles_per_warp);
void tc_lm_full(const TcLmFullParams& p, int lane_tiles_per_block);

}  // namespace stride
