// One forward step over an arbitrary set of rows.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kernels.h"
#include "kv_cache.h"
#include "model.h"

namespace stride {

enum class LogitMode : int32_t { kNone = 0, kGreedy = 1, kFull = 2 };

struct StepRow {
  int32_t slot;
  int32_t pos;
  int32_t token;
  LogitMode mode;
};

struct EngineOptions {
  int max_rows = 512;
  int max_full_rows = 32;
  int max_slots = 256;
  int64_t kv_bytes = int64_t{16} << 30;
  uint32_t kv_shuffle_seed = 0;
  int bug = 0;
  KernelPath path = KernelPath::kExact;
};

class Engine {
 public:
  Engine(const std::string& model_dir, const EngineOptions& opts);
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  const ModelConfig& config() const { return cfg_; }
  KvCache& kv() { return *kv_; }
  const Weights& weights() const { return *weights_; }
  const EngineOptions& options() const { return opts_; }

  // Runs one step. greedy receives one entry per kGreedy row in row order.
  // full receives vocab floats per kFull row in row order.
  void step(const std::vector<StepRow>& rows, std::vector<BestEntry>& greedy, std::vector<float>& full);

 private:
  void* alloc_scratch(size_t bytes);
  void build_attention_items(int rows);
  void step_exact(const std::vector<StepRow>& rows, std::vector<BestEntry>& greedy, std::vector<float>& full);
  void step_tensor_core(const std::vector<StepRow>& rows, std::vector<BestEntry>& greedy, std::vector<float>& full);

  ModelConfig cfg_;
  EngineOptions opts_;
  std::unique_ptr<Weights> weights_;
  std::unique_ptr<KvCache> kv_;

  float* h_ = nullptr;
  float* nx_ = nullptr;
  float* q_ = nullptr;
  float* attn_ = nullptr;
  float* z_ = nullptr;
  float* lm_x_ = nullptr;
  float* logits_ = nullptr;
  BestEntry* best_ = nullptr;
  int32_t* tokens_ = nullptr;
  int32_t* row_slot_ = nullptr;
  int32_t* row_pos_ = nullptr;
  int32_t* lm_rows_ = nullptr;
  int32_t* attn_items_ = nullptr;
  float* attn_scratch_ = nullptr;
  int rec_floats_ = 0;
  int max_pages_ = 0;
  std::vector<void*> scratch_;

  std::vector<int32_t> host_tokens_;
  std::vector<int32_t> host_slots_;
  std::vector<int32_t> host_pos_;
  std::vector<int32_t> host_items_;
  std::vector<int32_t> greedy_rows_;
  std::vector<int32_t> full_rows_;
  std::vector<BestEntry> best_init_;
  std::vector<float> full_download_;
};

}  // namespace stride
