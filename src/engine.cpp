#include "engine.h"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "device.h"

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

int padded(int rows) { return rows == 1 ? 1 : (rows + 3) / 4 * 4; }

}  // namespace

void* Engine::alloc_scratch(size_t bytes) {
  void* ptr = device_alloc(bytes);
  scratch_.push_back(ptr);
  return ptr;
}

Engine::Engine(const std::string& model_dir, const EngineOptions& opts)
    : cfg_(ModelConfig::load(model_dir)), opts_(opts) {
  device_init();
  weights_ = std::make_unique<Weights>(model_dir, cfg_);
  kv_ = std::make_unique<KvCache>(cfg_, opts_.kv_bytes, opts_.max_slots, opts_.kv_shuffle_seed);

  const size_t lanes = static_cast<size_t>(padded(opts_.max_rows));
  const size_t full_lanes = static_cast<size_t>(padded(opts_.max_full_rows));
  h_ = static_cast<float*>(alloc_scratch(lanes * cfg_.hidden * 4));
  nx_ = static_cast<float*>(alloc_scratch(lanes * cfg_.hidden * 4));
  q_ = static_cast<float*>(alloc_scratch(lanes * cfg_.heads * cfg_.head_dim * 4));
  attn_ = static_cast<float*>(alloc_scratch(lanes * cfg_.heads * cfg_.head_dim * 4));
  z_ = static_cast<float*>(alloc_scratch(lanes * cfg_.mlp * 4));
  lm_x_ = static_cast<float*>(alloc_scratch(lanes * cfg_.hidden * 4));
  const size_t vocab_rows = static_cast<size_t>(weights_->lm().groups) * 4;
  logits_ = static_cast<float*>(alloc_scratch(vocab_rows * full_lanes * 4));
  best_ = static_cast<BestEntry*>(alloc_scratch(lanes * sizeof(BestEntry)));
  tokens_ = static_cast<int32_t*>(alloc_scratch(lanes * 4));
  row_slot_ = static_cast<int32_t*>(alloc_scratch(lanes * 4));
  row_pos_ = static_cast<int32_t*>(alloc_scratch(lanes * 4));
  lm_rows_ = static_cast<int32_t*>(alloc_scratch(lanes * 4));
}

Engine::~Engine() {
  device_wait();
  for (void* p : scratch_) device_free(p);
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
  if (static_cast<int>(full_rows_.size()) > opts_.max_full_rows) {
    throw std::runtime_error("step exceeds max_full_rows");
  }
  kv_->sync();
  device_upload(tokens_, host_tokens_.data(), n * 4);
  device_upload(row_slot_, host_slots_.data(), n * 4);
  device_upload(row_pos_, host_pos_.data(), n * 4);

  const LaneShape shape = lane_shape(n);
  Lanes lanes{n, shape.lane_groups, shape.stride, opts_.bug};
  const KvRef kv = kv_->ref();
  const int d = cfg_.hidden;
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
    qkv.kv = kv;
    qkv.rope = weights_->rope();
    qkv.row_slot = row_slot_;
    qkv.row_pos = row_pos_;
    qkv.layer = li;
    qkv.head_dim = cfg_.head_dim;
    k_qkv(qkv, shape.J);

    AttentionParams attn{};
    attn.l = lanes;
    attn.q = q_;
    attn.out = attn_;
    attn.kv = kv;
    attn.row_slot = row_slot_;
    attn.row_pos = row_pos_;
    attn.layer = li;
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
  norm.weight = weights_->norm_final();
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
  device_wait();
}

}  // namespace stride
