#include "model.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <nlohmann/json.hpp>
#include <stdexcept>

#include "device.h"
#include "safetensors.h"

namespace stride {

ModelConfig ModelConfig::load(const std::string& model_dir) {
  std::ifstream in(model_dir + "/config.json");
  if (!in) throw std::runtime_error("cannot open config.json in " + model_dir);
  const auto j = nlohmann::json::parse(in);
  ModelConfig c;
  c.hidden = j.at("hidden_size");
  c.mlp = j.at("intermediate_size");
  c.layers = j.at("num_hidden_layers");
  c.heads = j.at("num_attention_heads");
  c.kv_heads = j.value("num_key_value_heads", c.heads);
  c.head_dim = j.value("head_dim", c.hidden / c.heads);
  c.vocab = j.at("vocab_size");
  c.rms_eps = j.value("rms_norm_eps", 1e-5f);
  c.rope_theta = j.value("rope_theta", 10000.0);
  if (j.contains("rope_scaling") && !j["rope_scaling"].is_null()) {
    const auto& r = j["rope_scaling"];
    const std::string type = r.value("rope_type", r.value("type", std::string()));
    if (type == "llama3") {
      c.rope_llama3 = true;
      c.rope_factor = r.value("factor", 32.0);
      c.rope_low = r.value("low_freq_factor", 1.0);
      c.rope_high = r.value("high_freq_factor", 4.0);
      c.rope_orig_ctx = r.value("original_max_position_embeddings", 8192.0);
    } else if (!type.empty() && type != "default") {
      throw std::runtime_error("unsupported rope type " + type);
    }
  }
  if (j.contains("eos_token_id") && !j["eos_token_id"].is_null()) {
    if (j["eos_token_id"].is_array()) {
      c.eos_tokens = j["eos_token_id"].get<std::vector<int>>();
    } else {
      c.eos_tokens.push_back(j["eos_token_id"].get<int>());
    }
  }
  if (c.heads % c.kv_heads != 0 || c.repeat() > 4) throw std::runtime_error("unsupported head layout");
  if (c.head_dim % 4 != 0 || c.head_dim > kMaxHeadDim) throw std::runtime_error("unsupported head_dim");
  return c;
}

std::vector<float> build_rope_table(const ModelConfig& cfg, int max_positions) {
  const int dim = cfg.head_dim;
  const int half = dim / 2;
  std::vector<float> inv(half);
  const float pi2 = static_cast<float>(2.0 * M_PI);
  for (int i = 0; i < half; ++i) {
    const float e = static_cast<float>(2 * i) / static_cast<float>(dim);
    const float base = static_cast<float>(std::pow(cfg.rope_theta, static_cast<double>(e)));
    float f = 1.0f / base;
    if (cfg.rope_llama3) {
      const float factor = static_cast<float>(cfg.rope_factor);
      const float low_wave = static_cast<float>(cfg.rope_orig_ctx / cfg.rope_low);
      const float high_wave = static_cast<float>(cfg.rope_orig_ctx / cfg.rope_high);
      const float wave = pi2 / f;
      const float smooth = (static_cast<float>(cfg.rope_orig_ctx) / wave - static_cast<float>(cfg.rope_low)) /
                           static_cast<float>(cfg.rope_high - cfg.rope_low);
      if (wave > low_wave) {
        f = f / factor;
      } else if (!(wave < high_wave)) {
        f = (1.0f - smooth) * (f / factor) + smooth * f;
      }
    }
    inv[i] = f;
  }
  std::vector<float> table(static_cast<size_t>(max_positions) * half * 2);
  for (int pos = 0; pos < max_positions; ++pos) {
    for (int i = 0; i < half; ++i) {
      const float angle = inv[i] * static_cast<float>(pos);
      float* entry = &table[(static_cast<size_t>(pos) * half + i) * 2];
      entry[0] = static_cast<float>(std::cos(static_cast<double>(angle)));
      entry[1] = static_cast<float>(std::sin(static_cast<double>(angle)));
    }
  }
  return table;
}

namespace {

// Packs rows into four-row groups: dst[group][col][slot]. row_of maps a group and
// slot to a source row, or -1 for a zero padding row.
std::vector<uint16_t> pack(const TensorView& t, int groups, const std::function<int(int, int)>& row_of) {
  if (t.dtype != "BF16") throw std::runtime_error("expected BF16 weights, got " + t.dtype);
  const int rows = static_cast<int>(t.shape[0]);
  const int cols = static_cast<int>(t.shape[1]);
  const auto* src = reinterpret_cast<const uint16_t*>(t.data);
  std::vector<uint16_t> out(static_cast<size_t>(groups) * cols * 4, 0);
  for (int g = 0; g < groups; ++g) {
    for (int slot = 0; slot < 4; ++slot) {
      const int row = row_of(g, slot);
      if (row < 0) continue;
      if (row >= rows) throw std::runtime_error("packing row out of range");
      const uint16_t* r = src + static_cast<int64_t>(row) * cols;
      uint16_t* d = &out[static_cast<size_t>(g) * cols * 4 + slot];
      for (int c = 0; c < cols; ++c) d[static_cast<size_t>(c) * 4] = r[c];
    }
  }
  return out;
}

// Packs rows into 16-row tiles: dst[row tile][col tile][slot][col % 16].
std::vector<uint16_t> pack_tiles(const TensorView& t, int row_tiles, const std::function<int(int, int)>& row_of) {
  if (t.dtype != "BF16") throw std::runtime_error("expected BF16 weights, got " + t.dtype);
  const int rows = static_cast<int>(t.shape[0]);
  const int cols = static_cast<int>(t.shape[1]);
  const int col_tiles = cols / 16;
  const auto* src = reinterpret_cast<const uint16_t*>(t.data);
  std::vector<uint16_t> out(static_cast<size_t>(row_tiles) * col_tiles * 256, 0);
  for (int rt = 0; rt < row_tiles; ++rt) {
    for (int slot = 0; slot < 16; ++slot) {
      const int row = row_of(rt, slot);
      if (row < 0) continue;
      if (row >= rows) throw std::runtime_error("tile packing row out of range");
      const uint16_t* r = src + static_cast<int64_t>(row) * cols;
      for (int ct = 0; ct < col_tiles; ++ct) {
        std::memcpy(&out[(static_cast<size_t>(rt) * col_tiles + ct) * 256 + slot * 16], r + ct * 16, 32);
      }
    }
  }
  return out;
}

std::vector<float> widen(const TensorView& t) {
  if (t.dtype != "BF16") throw std::runtime_error("expected BF16 weights, got " + t.dtype);
  const auto* src = reinterpret_cast<const uint16_t*>(t.data);
  const size_t n = t.bytes / 2;
  std::vector<float> out(n);
  for (size_t i = 0; i < n; ++i) {
    const uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
    std::memcpy(&out[i], &bits, 4);
  }
  return out;
}

}  // namespace

Weights::Weights(const std::string& model_dir, const ModelConfig& cfg, KernelPath path) {
  SafeTensors st(model_dir + "/model.safetensors");

  const auto place = [this](const void* data, size_t bytes) {
    void* ptr = device_alloc(bytes);
    device_upload(ptr, data, bytes);
    allocations_.push_back(ptr);
    bytes_on_device_ += bytes;
    return ptr;
  };
  const auto place_u16 = [&](const std::vector<uint16_t>& v) {
    return static_cast<const uint16_t*>(place(v.data(), v.size() * 2));
  };
  const auto place_f32 = [&](const std::vector<float>& v) {
    return static_cast<const float*>(place(v.data(), v.size() * 4));
  };

  const int D = cfg.head_dim;
  // Four-row groups hold two RoPE pairs: (d, d+1, d+D/2, d+D/2+1).
  const auto rope_rows = [D](int g, int slot) {
    const int head = g / (D / 4);
    const int d0 = 2 * (g % (D / 4));
    const int offsets[4] = {d0, d0 + 1, d0 + D / 2, d0 + D / 2 + 1};
    return head * D + offsets[slot];
  };
  const auto plain_rows = [](int g, int slot) { return g * 4 + slot; };

  const bool tiles = path == KernelPath::kTensorCore;
  // A row tile holds eight RoPE pairs: (d0 + i) and (d0 + D/2 + i).
  const auto rope_tile_rows = [D](int rt, int slot) {
    const int per_head = D / 16;
    const int head = rt / per_head;
    const int d0 = (rt % per_head) * 8;
    return head * D + (slot < 8 ? d0 + slot : d0 + D / 2 + slot - 8);
  };
  const auto plain_tile_rows = [](int rt, int slot) { return rt * 16 + slot; };

  const auto mat_of = [&](const TensorView& t, bool rope) {
    const int rows = static_cast<int>(t.shape[0]);
    const int cols = static_cast<int>(t.shape[1]);
    if (tiles ? (rows % 16 != 0 || cols % 16 != 0) : (rows % 4 != 0 || cols % 8 != 0)) {
      throw std::runtime_error("matrix shape incompatible with the packed kernels");
    }
    const int groups = rows / (tiles ? 16 : 4);
    GemmMat m{};
    m.w = tiles ? place_u16(pack_tiles(t, groups, rope ? std::function<int(int, int)>(rope_tile_rows) : std::function<int(int, int)>(plain_tile_rows)))
                : place_u16(pack(t, groups, rope ? std::function<int(int, int)>(rope_rows) : std::function<int(int, int)>(plain_rows)));
    m.groups = groups;
    m.cols = cols;
    bytes_per_pass_ += static_cast<uint64_t>(rows) * cols * 2;
    return m;
  };

  layers_.resize(cfg.layers);
  for (int i = 0; i < cfg.layers; ++i) {
    const std::string pre = "model.layers." + std::to_string(i) + ".";
    LayerWeights& l = layers_[i];
    l.q = mat_of(st.get(pre + "self_attn.q_proj.weight"), true);
    l.k = mat_of(st.get(pre + "self_attn.k_proj.weight"), true);
    l.v = mat_of(st.get(pre + "self_attn.v_proj.weight"), true);
    l.o = mat_of(st.get(pre + "self_attn.o_proj.weight"), false);
    l.w1 = mat_of(st.get(pre + "mlp.gate_proj.weight"), false);
    l.w3 = mat_of(st.get(pre + "mlp.up_proj.weight"), false);
    l.w2 = mat_of(st.get(pre + "mlp.down_proj.weight"), false);
    l.norm_attn = place_f32(widen(st.get(pre + "input_layernorm.weight")));
    l.norm_mlp = place_f32(widen(st.get(pre + "post_attention_layernorm.weight")));
  }
  norm_final_ = place_f32(widen(st.get("model.norm.weight")));

  const int group_rows = tiles ? 16 : 4;
  const int vocab_groups = (cfg.vocab + group_rows - 1) / group_rows;
  const std::function<int(int, int)> vocab_row = [&cfg, group_rows](int g, int slot) {
    const int row = g * group_rows + slot;
    return row < cfg.vocab ? row : -1;
  };
  const auto pack_vocab = [&](const TensorView& t) {
    return tiles ? pack_tiles(t, vocab_groups, vocab_row) : pack(t, vocab_groups, vocab_row);
  };
  embed_ = place_u16(pack_vocab(st.get("model.embed_tokens.weight")));
  if (st.has("lm_head.weight")) {
    lm_.w = place_u16(pack_vocab(st.get("lm_head.weight")));
  } else {
    lm_.w = embed_;
  }
  lm_.groups = vocab_groups;
  lm_.cols = cfg.hidden;
  bytes_per_pass_ += static_cast<uint64_t>(cfg.vocab) * cfg.hidden * 2;

  rope_ = place_f32(build_rope_table(cfg, kMaxCtx + 1));
}

Weights::~Weights() {
  for (void* p : allocations_) device_free(p);
}

}  // namespace stride
