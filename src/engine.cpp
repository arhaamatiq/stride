#include "engine.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "device.h"
#if defined(STRIDE_BACKEND_CUDA)
#include "kernels_tc.h"
#endif

namespace stride {

namespace {

struct LaneShape {
  int J;
  int lane_groups;
  int stride;
};

// A lone row runs the J=1 kernels with stride 1; anything else uses four-lane groups.
LaneShape lane_shape(int rows) {
  if (rows == 1) return {1, 1, 1};
  const int groups = (rows + 3) / 4;
  return {4, groups, groups * 4};
}

size_t round16(int rows) { return static_cast<size_t>((rows + 15) / 16 * 16); }

}  // namespace

void* Engine::alloc_scratch(size_t bytes) {
  void* ptr = device_alloc(bytes);
  scratch_.push_back(ptr);
  return ptr;
}

Engine::Engine(const std::string& model_dir, const EngineOptions& opts)
    : cfg_(ModelConfig::load(model_dir)), opts_(opts) {
#if !defined(STRIDE_BACKEND_CUDA)
  if (opts_.path == KernelPath::kTensorCore) throw std::runtime_error("the tensor-core path requires a CUDA build");
#endif
  device_init();
  weights_ = std::make_unique<Weights>(model_dir, cfg_, opts_.path);
  const int kv_elem_bytes = opts_.path == KernelPath::kTensorCore ? 2 : 4;
  const int64_t kv_block_bytes = int64_t{kKvBlockTokens} * cfg_.head_dim * cfg_.kv_heads * 2 * cfg_.layers * kv_elem_bytes;
  const int64_t kv_bytes = opts_.kv_blocks > 0 ? opts_.kv_blocks * kv_block_bytes : opts_.kv_bytes;
  kv_ = std::make_unique<KvCache>(cfg_, kv_bytes, opts_.max_slots, kv_elem_bytes, opts_.kv_shuffle_seed);
  kv_->set_bug(opts_.bug);

  // Sized for FP32 in both paths; the BF16 tile buffers simply use half.
  const size_t lanes = round16(opts_.max_rows);
  const size_t full_lanes = round16(opts_.max_full_rows);
  const size_t q_row = static_cast<size_t>(cfg_.heads) * cfg_.head_dim;
  h_ = static_cast<float*>(alloc_scratch(lanes * cfg_.hidden * 4));
  nx_ = static_cast<float*>(alloc_scratch(lanes * cfg_.hidden * 4));
  q_ = static_cast<float*>(alloc_scratch(lanes * q_row * 4));
  attn_ = static_cast<float*>(alloc_scratch(lanes * q_row * 4));
  z_ = static_cast<float*>(alloc_scratch(lanes * cfg_.mlp * 4));
  lm_x_ = static_cast<float*>(alloc_scratch(lanes * cfg_.hidden * 4));
  const int group_rows = opts_.path == KernelPath::kTensorCore ? 16 : 4;
  const size_t vocab_rows = static_cast<size_t>(weights_->lm().groups) * group_rows;
  logits_ = static_cast<float*>(alloc_scratch(vocab_rows * full_lanes * 4));
  best_ = static_cast<BestEntry*>(alloc_scratch(lanes * sizeof(BestEntry)));
  tokens_ = static_cast<int32_t*>(alloc_scratch(lanes * 4));
  row_slot_ = static_cast<int32_t*>(alloc_scratch(lanes * 4));
  row_pos_ = static_cast<int32_t*>(alloc_scratch(lanes * 4));
  lm_rows_ = static_cast<int32_t*>(alloc_scratch(lanes * 4));

  // Unused lanes in a tile must never hold uninitialized memory.
  for (auto [ptr, bytes] : {std::pair<void*, size_t>{h_, lanes * cfg_.hidden * 4},
                            {nx_, lanes * cfg_.hidden * 4},
                            {attn_, lanes * q_row * 4},
                            {z_, lanes * cfg_.mlp * 4},
                            {lm_x_, lanes * cfg_.hidden * 4}}) {
    const std::vector<char> zeros(bytes, 0);
    device_upload(ptr, zeros.data(), bytes);
  }

  max_pages_ = (kMaxCtx + kKvBlockTokens - 1) >> kKvBlockShift;
  rec_floats_ = 4 * cfg_.head_dim + 8;
  attn_items_ = static_cast<int32_t*>(alloc_scratch(lanes * max_pages_ * 2 * 4));
  attn_scratch_ = static_cast<float*>(alloc_scratch(lanes * cfg_.kv_heads * max_pages_ * rec_floats_ * 4));
}

Engine::~Engine() {
  device_wait();
  for (void* p : scratch_) device_free(p);
}

void Engine::build_attention_items(int rows) {
  host_items_.clear();
  for (int j = 0; j < rows; ++j) {
    const int pages = (host_pos_[j] >> kKvBlockShift) + 1;
    for (int page = 0; page < pages; ++page) {
      host_items_.push_back(j);
      host_items_.push_back(page);
    }
  }
  device_upload(attn_items_, host_items_.data(), host_items_.size() * 4);
}

void Engine::step(const std::vector<StepRow>& rows, std::vector<BestEntry>& greedy, std::vector<float>& full) {
  greedy.clear();
  full.clear();
  const int n = static_cast<int>(rows.size());
  if (n == 0) return;
  if (n > opts_.max_rows) throw std::runtime_error("step exceeds max_rows");

  host_tokens_.resize(n);
  host_slots_.resize(n);
  host_pos_.resize(n);
  greedy_rows_.clear();
  full_rows_.clear();
  for (int i = 0; i < n; ++i) {
    host_tokens_[i] = rows[i].token;
    host_slots_[i] = rows[i].slot;
    host_pos_[i] = rows[i].pos;
    if (rows[i].mode == LogitMode::kGreedy) greedy_rows_.push_back(i);
    if (rows[i].mode == LogitMode::kFull) full_rows_.push_back(i);
  }
  if (opts_.bug == 303 && greedy_rows_.size() > 1 && static_cast<int>(greedy_rows_.size()) < n) {
    std::swap(greedy_rows_[0], greedy_rows_[1]);
  }
  if (static_cast<int>(full_rows_.size()) > opts_.max_full_rows) {
    throw std::runtime_error("step exceeds max_full_rows");
  }
  kv_->sync();
  device_upload(tokens_, host_tokens_.data(), n * 4);
  device_upload(row_slot_, host_slots_.data(), n * 4);
  device_upload(row_pos_, host_pos_.data(), n * 4);
  build_attention_items(n);

  if (opts_.path == KernelPath::kTensorCore) {
    step_tensor_core(rows, greedy, full);
  } else {
    step_exact(rows, greedy, full);
  }
  device_wait();
}

void Engine::step_exact(const std::vector<StepRow>& rows, std::vector<BestEntry>& greedy, std::vector<float>& full) {
  const int n = static_cast<int>(rows.size());
  const LaneShape shape = lane_shape(n);
  Lanes lanes{n, shape.lane_groups, shape.stride, opts_.bug};
  const KvRef kv = kv_->ref();
  const int d = cfg_.hidden;
  const int q_stride = cfg_.heads * cfg_.head_dim;
  const float scale = static_cast<float>(std::pow(static_cast<double>(cfg_.head_dim), -0.5));

  EmbedParams embed{};
  embed.l = lanes;
  embed.embed = weights_->embed();
  embed.tokens = tokens_;
  embed.h = h_;
  embed.dim = d;
  k_embed(embed);

  NormParams norm{};
  norm.l = lanes;
  norm.dim = d;
  norm.eps = cfg_.rms_eps;

  for (int li = 0; li < cfg_.layers; ++li) {
    const LayerWeights& w = weights_->layer(li);

    norm.in = h_;
    norm.out = nx_;
    norm.weight = w.norm_attn;
    k_norm(norm, shape.J);

    QkvParams qkv{};
    qkv.l = lanes;
    qkv.x = nx_;
    qkv.q = w.q;
    qkv.k = w.k;
    qkv.v = w.v;
    qkv.q_out = q_;
    qkv.q_stride = q_stride;
    qkv.kv = kv;
    qkv.rope = weights_->rope();
    qkv.row_slot = row_slot_;
    qkv.row_pos = row_pos_;
    qkv.layer = opts_.bug == 301 ? 0 : li;
    qkv.head_dim = cfg_.head_dim;
    k_qkv(qkv, shape.J);

    AttentionParams attn{};
    attn.l = lanes;
    attn.q = q_;
    attn.q_stride = q_stride;
    attn.out = attn_;
    attn.out_cols = q_stride;
    attn.kv = kv;
    attn.row_slot = row_slot_;
    attn.row_pos = row_pos_;
    attn.items = attn_items_;
    attn.item_count = static_cast<int32_t>(host_items_.size() / 2);
    attn.scratch = attn_scratch_;
    attn.max_pages = max_pages_;
    attn.rec_floats = rec_floats_;
    attn.layer = opts_.bug == 301 ? 0 : li;
    attn.n_kv = cfg_.kv_heads;
    attn.n_q = cfg_.repeat();
    attn.head_dim = cfg_.head_dim;
    attn.scale = scale;
    k_attention(attn);

    GemmResidualParams o{};
    o.l = lanes;
    o.x = attn_;
    o.m = w.o;
    o.h = h_;
    k_gemm_residual(o, shape.J);

    norm.in = h_;
    norm.out = nx_;
    norm.weight = w.norm_mlp;
    k_norm(norm, shape.J);

    Mlp13Params mlp{};
    mlp.l = lanes;
    mlp.x = nx_;
    mlp.w1 = w.w1;
    mlp.w3 = w.w3;
    mlp.z = z_;
    k_mlp13(mlp, shape.J);

    GemmResidualParams down{};
    down.l = lanes;
    down.x = z_;
    down.m = w.w2;
    down.h = h_;
    k_gemm_residual(down, shape.J);
  }

  norm.in = h_;
  norm.out = nx_;
  norm.weight = opts_.bug == 302 ? weights_->layer(0).norm_mlp : weights_->norm_final();
  k_norm(norm, shape.J);

  const auto compact = [&](const std::vector<int32_t>& wanted, const LaneShape& ls) {
    device_upload(lm_rows_, wanted.data(), wanted.size() * 4);
    CompactParams c{};
    c.l = lanes;
    c.in = nx_;
    c.out = lm_x_;
    c.lm_rows = lm_rows_;
    c.lm_count = static_cast<int32_t>(wanted.size());
    c.lm_stride = ls.stride;
    c.dim = d;
    k_compact(c);
  };

  if (!greedy_rows_.empty()) {
    const int count = static_cast<int>(greedy_rows_.size());
    const LaneShape ls = lane_shape(count);
    const bool direct = count == n;  // rows already sit in lane order
    if (!direct) compact(greedy_rows_, ls);
    best_init_.assign(count, BestEntry{-std::numeric_limits<float>::infinity(), -1});
    device_upload(best_, best_init_.data(), count * sizeof(BestEntry));
    LmGreedyParams lm{};
    lm.l = Lanes{count, ls.lane_groups, ls.stride, opts_.bug};
    lm.x = direct ? nx_ : lm_x_;
    lm.m = weights_->lm();
    lm.vocab = cfg_.vocab;
    lm.best = best_;
    k_lm_greedy(lm, ls.J);

    greedy.resize(count);
    device_download(greedy.data(), best_, count * sizeof(BestEntry));
  }

  if (!full_rows_.empty()) {
    const int count = static_cast<int>(full_rows_.size());
    const LaneShape ls = lane_shape(count);
    const bool direct = count == n;
    if (!direct) compact(full_rows_, ls);
    LmFullParams lm{};
    lm.l = Lanes{count, ls.lane_groups, ls.stride, opts_.bug};
    lm.x = direct ? nx_ : lm_x_;
    lm.m = weights_->lm();
    lm.logits = logits_;
    k_lm_full(lm, ls.J);

    const size_t vocab_rows = static_cast<size_t>(lm.m.groups) * 4;
    full_download_.resize(vocab_rows * ls.stride);
    device_download(full_download_.data(), logits_, full_download_.size() * 4);
    full.resize(static_cast<size_t>(cfg_.vocab) * count);
    for (int l = 0; l < count; ++l) {
      float* out = &full[static_cast<size_t>(l) * cfg_.vocab];
      for (int i = 0; i < cfg_.vocab; ++i) out[i] = full_download_[static_cast<size_t>(i) * ls.stride + l];
    }
  }
}

#if defined(STRIDE_BACKEND_CUDA)
void Engine::step_tensor_core(const std::vector<StepRow>& rows, std::vector<BestEntry>& greedy, std::vector<float>& full) {
  const int n = static_cast<int>(rows.size());
  const int lane_tiles = (n + 15) / 16;
  Lanes lanes{n, lane_tiles, 0, opts_.bug};
  const KvRef kv = kv_->ref();
  const int d = cfg_.hidden;
  const int q_stride = cfg_.heads * cfg_.head_dim;
  const float scale = static_cast<float>(std::pow(static_cast<double>(cfg_.head_dim), -0.5));
  auto* x = reinterpret_cast<uint16_t*>(nx_);
  auto* attn_tiles = reinterpret_cast<uint16_t*>(attn_);
  auto* z = reinterpret_cast<uint16_t*>(z_);
  auto* lm_x = reinterpret_cast<uint16_t*>(lm_x_);
  // Sharing one weight load across two lane tiles pays off once the batch is large.
  const int lt_mlp = opts_.tc_lt_mlp > 0 ? opts_.tc_lt_mlp : (lane_tiles >= 4 ? 2 : 1);

  TcEmbedParams embed{};
  embed.l = lanes;
  embed.embed = weights_->embed();
  embed.tokens = tokens_;
  embed.h = h_;
  embed.dim = d;
  tc_embed(embed);

  TcNormParams norm{};
  norm.l = lanes;
  norm.h = h_;
  norm.x = x;
  norm.dim = d;
  norm.eps = cfg_.rms_eps;

  for (int li = 0; li < cfg_.layers; ++li) {
    const LayerWeights& w = weights_->layer(li);

    norm.weight = w.norm_attn;
    tc_norm(norm, opts_.tc_norm_per_lane);

    TcQkvParams qkv{};
    qkv.l = lanes;
    qkv.x = x;
    qkv.q = w.q;
    qkv.k = w.k;
    qkv.v = w.v;
    qkv.q_out = q_;
    qkv.q_stride = q_stride;
    qkv.kv = kv;
    qkv.rope = weights_->rope();
    qkv.row_slot = row_slot_;
    qkv.row_pos = row_pos_;
    qkv.layer = opts_.bug == 301 ? 0 : li;
    qkv.head_dim = cfg_.head_dim;
    tc_qkv(qkv, 1);

    AttentionParams attn{};
    attn.l = lanes;
    attn.q = q_;
    attn.q_stride = q_stride;
    attn.out = attn_tiles;
    attn.out_cols = q_stride;
    attn.kv = kv;
    attn.row_slot = row_slot_;
    attn.row_pos = row_pos_;
    attn.items = attn_items_;
    attn.item_count = static_cast<int32_t>(host_items_.size() / 2);
    attn.scratch = attn_scratch_;
    attn.max_pages = max_pages_;
    attn.rec_floats = rec_floats_;
    attn.layer = opts_.bug == 301 ? 0 : li;
    attn.n_kv = cfg_.kv_heads;
    attn.n_q = cfg_.repeat();
    attn.head_dim = cfg_.head_dim;
    attn.scale = scale;
    tc_attention(attn);

    TcGemmResidualParams o{};
    o.l = lanes;
    o.x = attn_tiles;
    o.m = w.o;
    o.h = h_;
    tc_gemm_residual(o);

    norm.weight = w.norm_mlp;
    tc_norm(norm, opts_.tc_norm_per_lane);

    TcMlp13Params mlp{};
    mlp.l = lanes;
    mlp.x = x;
    mlp.w1 = w.w1;
    mlp.w3 = w.w3;
    mlp.z = z;
    tc_mlp13(mlp, lt_mlp);

    TcGemmResidualParams down{};
    down.l = lanes;
    down.x = z;
    down.m = w.w2;
    down.h = h_;
    tc_gemm_residual(down);
  }

  norm.weight = opts_.bug == 302 ? weights_->layer(0).norm_mlp : weights_->norm_final();
  tc_norm(norm, opts_.tc_norm_per_lane);

  const auto compact = [&](const std::vector<int32_t>& wanted) {
    device_upload(lm_rows_, wanted.data(), wanted.size() * 4);
    TcCompactParams c{};
    c.l = lanes;
    c.in = x;
    c.out = lm_x;
    c.lm_rows = lm_rows_;
    c.lm_count = static_cast<int32_t>(wanted.size());
    c.col_tiles = d / 16;
    tc_compact(c);
  };

  if (!greedy_rows_.empty()) {
    const int count = static_cast<int>(greedy_rows_.size());
    const int tiles = (count + 15) / 16;
    const bool direct = count == n;
    if (!direct) compact(greedy_rows_);
    best_init_.assign(count, BestEntry{-std::numeric_limits<float>::infinity(), -1});
    device_upload(best_, best_init_.data(), count * sizeof(BestEntry));
    TcLmGreedyParams lm{};
    lm.l = Lanes{count, tiles, 0, opts_.bug};
    lm.x = direct ? x : lm_x;
    lm.m = weights_->lm();
    lm.vocab = cfg_.vocab;
    lm.best = best_;
    tc_lm_greedy(lm, opts_.tc_lt_lm > 0 ? opts_.tc_lt_lm : (tiles >= 4 ? 2 : 1));

    greedy.resize(count);
    device_download(greedy.data(), best_, count * sizeof(BestEntry));
  }

  if (!full_rows_.empty()) {
    const int count = static_cast<int>(full_rows_.size());
    const int tiles = (count + 15) / 16;
    const bool direct = count == n;
    if (!direct) compact(full_rows_);
    TcLmFullParams lm{};
    lm.l = Lanes{count, tiles, 0, opts_.bug};
    lm.x = direct ? x : lm_x;
    lm.m = weights_->lm();
    lm.logits = logits_;
    tc_lm_full(lm, 1);

    const int row_tiles = lm.m.groups;
    full_download_.resize(static_cast<size_t>(row_tiles) * tiles * 256);
    device_download(full_download_.data(), logits_, full_download_.size() * 4);
    full.resize(static_cast<size_t>(cfg_.vocab) * count);
    for (int l = 0; l < count; ++l) {
      float* out = &full[static_cast<size_t>(l) * cfg_.vocab];
      const int lt = l / 16;
      const int jj = l % 16;
      for (int i = 0; i < cfg_.vocab; ++i) {
        out[i] = full_download_[(static_cast<size_t>(i / 16) * tiles + lt) * 256 + (i % 16) * 16 + jj];
      }
    }
  }
}
#else
void Engine::step_tensor_core(const std::vector<StepRow>&, std::vector<BestEntry>&, std::vector<float>&) {
  throw std::runtime_error("the tensor-core path requires a CUDA build");
}
#endif

}  // namespace stride
