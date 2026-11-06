// Model configuration, weight loading and packing.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "kernels.h"

namespace stride {

struct ModelConfig {
  int hidden = 0;
  int mlp = 0;
  int layers = 0;
  int heads = 0;
  int kv_heads = 0;
  int head_dim = 0;
  int vocab = 0;
  float rms_eps = 1e-5f;
  double rope_theta = 10000.0;
  bool rope_llama3 = false;
  double rope_factor = 1.0;
  double rope_low = 1.0;
  double rope_high = 4.0;
  double rope_orig_ctx = 8192.0;
  std::vector<int> eos_tokens;

  static ModelConfig load(const std::string& model_dir);
  int repeat() const { return heads / kv_heads; }
};

struct LayerWeights {
  GemmMat q, k, v, o, w1, w3, w2;
  const float* norm_attn;
  const float* norm_mlp;
};

// Which kernel family a Weights instance is packed for.
enum class KernelPath { kExact, kTensorCore };

class Weights {
 public:
  Weights(const std::string& model_dir, const ModelConfig& cfg, KernelPath path);
  ~Weights();
  Weights(const Weights&) = delete;
  Weights& operator=(const Weights&) = delete;

  const LayerWeights& layer(int i) const { return layers_[i]; }
  const uint16_t* embed() const { return embed_; }
  GemmMat lm() const { return lm_; }
  const float* norm_final() const { return norm_final_; }
  const float* rope() const { return rope_; }
  // Bytes of weights a single forward pass reads, for the bandwidth roofline.
  uint64_t bytes_per_pass() const { return bytes_per_pass_; }
  uint64_t bytes_on_device() const { return bytes_on_device_; }

 private:
  std::vector<void*> allocations_;
  std::vector<LayerWeights> layers_;
  const uint16_t* embed_ = nullptr;
  GemmMat lm_{};
  const float* norm_final_ = nullptr;
  const float* rope_ = nullptr;
  uint64_t bytes_per_pass_ = 0;
  uint64_t bytes_on_device_ = 0;
};

// cos/sin table, [pos][half][2], matching the FP32 construction in the reference.
std::vector<float> build_rope_table(const ModelConfig& cfg, int max_positions);

}  // namespace stride
