// Parameter blocks and entry points for the fused kernels.
//
// Layout conventions:
//   Activations are dim-major: A[dim * stride + lane], with lanes grouped in
//   fours so a 16-byte load fetches one dim across four sequences. Single-row
//   steps use stride 1 and the J=1 kernels.
//   Weights are packed in four-row groups: W[group][col][0..3], BF16.
//   One block owns one (row group, lane group). Its 32 threads interleave over
//   64-byte weight lines, so each warp load covers one contiguous 2 KB window.
//   KV blocks are [block][layer][K|V][group][token][dim], FP32.
#pragma once

#include <cstdint>

namespace stride {

constexpr int kBlockThreads = 32;
constexpr int kMaxCtx = 2048;
constexpr int kMaxHeadDim = 128;
constexpr int kKvBlockShift = 6;
constexpr int kKvBlockTokens = 1 << kKvBlockShift;

struct GemmMat {
  const uint16_t* w;
  int32_t groups;
  int32_t cols;
};

// Offsets are in elements; the exact path stores FP32, the tensor-core path BF16.
struct KvRef {
  void* arena;
  const int32_t* table;
  int64_t block_elems;
  int64_t layer_elems;
  int64_t which_elems;
  int64_t group_elems;
  int32_t max_blocks;
  int32_t head_dim;
};

struct Lanes {
  int32_t rows;
  int32_t lane_groups;
  int32_t stride;
  int32_t bug;
};

struct EmbedParams {
  Lanes l;
  const uint16_t* embed;
  const int32_t* tokens;
  float* h;
  int32_t dim;
};

struct NormParams {
  Lanes l;
  const float* in;
  float* out;
  const float* weight;
  int32_t dim;
  float eps;
};

struct QkvParams {
  Lanes l;
  const float* x;
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

struct AttentionParams {
  Lanes l;
  const float* q;  // [row][head][dim], FP32
  int32_t q_stride;
  void* out;
  int32_t out_cols;
  KvRef kv;
  const int32_t* row_slot;
  const int32_t* row_pos;
  const int32_t* items;  // (row, page) pairs to compute this step
  int32_t item_count;
  float* scratch;
  int32_t max_pages;
  int32_t rec_floats;
  int32_t layer;
  int32_t n_kv;
  int32_t n_q;
  int32_t head_dim;
  float scale;
};

struct GemmResidualParams {
  Lanes l;
  const float* x;
  GemmMat m;
  float* h;
};

struct Mlp13Params {
  Lanes l;
  const float* x;
  GemmMat w1, w3;
  float* z;
};

struct CompactParams {
  Lanes l;
  const float* in;
  float* out;
  const int32_t* lm_rows;
  int32_t lm_count;
  int32_t lm_stride;
  int32_t dim;
};

// 8-byte aligned so the argmax can update it with one atomic compare-and-swap.
struct alignas(8) BestEntry {
  float value;
  int32_t index;
};

struct LmGreedyParams {
  Lanes l;  // rows = number of logit lanes
  const float* x;
  GemmMat m;
  int32_t vocab;
  BestEntry* best;
};

struct LmFullParams {
  Lanes l;
  const float* x;
  GemmMat m;
  float* logits;
};

// J is 1 for single-row steps and 4 otherwise. Launches are ordered on one
// stream; the host synchronizes with device_wait().
void k_embed(const EmbedParams& p);
void k_norm(const NormParams& p, int J);
void k_qkv(const QkvParams& p, int J);
void k_attention(const AttentionParams& p);
void k_gemm_residual(const GemmResidualParams& p, int J);
void k_mlp13(const Mlp13Params& p, int J);
void k_compact(const CompactParams& p);
void k_lm_greedy(const LmGreedyParams& p, int J);
void k_lm_full(const LmFullParams& p, int J);


}  // namespace stride
