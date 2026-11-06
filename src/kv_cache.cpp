#include "kv_cache.h"

#include <algorithm>
#include <stdexcept>

#include "device.h"

namespace stride {

KvCache::KvCache(const ModelConfig& cfg, int64_t arena_bytes, int max_slots, int elem_bytes, uint32_t shuffle_seed)
    : elem_bytes_(elem_bytes), max_slots_(max_slots), rng_(shuffle_seed), shuffle_(shuffle_seed != 0) {
  const int64_t group_elems = static_cast<int64_t>(kKvBlockTokens) * cfg.head_dim;
  const int64_t which_elems = group_elems * cfg.kv_heads;
  const int64_t layer_elems = which_elems * 2;
  block_elems_ = layer_elems * cfg.layers;
  total_blocks_ = static_cast<int>(arena_bytes / (block_elems_ * elem_bytes_));
  if (total_blocks_ <= 0) throw std::runtime_error("KV arena smaller than one block");
  max_blocks_ = (kMaxCtx + kKvBlockTokens - 1) >> kKvBlockShift;

  arena_ = device_alloc(static_cast<size_t>(total_blocks_) * block_elems_ * elem_bytes_);
  table_.assign(static_cast<size_t>(max_slots_) * max_blocks_, 0);
  device_table_ = static_cast<int32_t*>(device_alloc(table_.size() * sizeof(int32_t)));
  device_upload(device_table_, table_.data(), table_.size() * sizeof(int32_t));

  free_.resize(total_blocks_);
  for (int i = 0; i < total_blocks_; ++i) free_[i] = total_blocks_ - 1 - i;
  if (shuffle_) std::shuffle(free_.begin(), free_.end(), rng_);
  for (int i = max_slots_ - 1; i >= 0; --i) free_slots_.push_back(i);
  owned_.resize(max_slots_);
  slot_dirty_.assign(max_slots_, false);

  ref_.arena = arena_;
  ref_.table = device_table_;
  ref_.block_elems = block_elems_;
  ref_.layer_elems = layer_elems;
  ref_.which_elems = which_elems;
  ref_.group_elems = group_elems;
  ref_.max_blocks = max_blocks_;
  ref_.head_dim = cfg.head_dim;
}

KvCache::~KvCache() {
  device_free(arena_);
  device_free(device_table_);
}

int KvCache::acquire_slot() {
  if (free_slots_.empty()) return -1;
  const int slot = free_slots_.back();
  free_slots_.pop_back();
  return slot;
}

void KvCache::release_slot(int slot) {
  auto& blocks = owned_[slot];
  free_.insert(free_.end(), blocks.begin(), blocks.end());
  if (shuffle_) std::shuffle(free_.begin(), free_.end(), rng_);
  blocks.clear();
  free_slots_.push_back(slot);
}

int KvCache::blocks_needed(int slot, int end_pos) const {
  const int want = (end_pos + kKvBlockTokens - 1) >> kKvBlockShift;
  return std::max(0, want - static_cast<int>(owned_[slot].size()));
}

bool KvCache::ensure(int slot, int end_pos) {
  if (end_pos > kMaxCtx) throw std::runtime_error("sequence exceeds kMaxCtx");
  const int need = blocks_needed(slot, end_pos);
  if (need > static_cast<int>(free_.size())) return false;
  for (int i = 0; i < need; ++i) {
    const int32_t block = free_.back();
    free_.pop_back();
    table_[static_cast<size_t>(slot) * max_blocks_ + owned_[slot].size()] = block;
    owned_[slot].push_back(block);
  }
  if (need > 0 && !slot_dirty_[slot]) {
    slot_dirty_[slot] = true;
    dirty_slots_.push_back(slot);
  }
  return true;
}

void KvCache::sync() {
  for (const int slot : dirty_slots_) {
    const size_t offset = static_cast<size_t>(slot) * max_blocks_;
    device_upload(device_table_ + offset, table_.data() + offset, max_blocks_ * sizeof(int32_t));
    slot_dirty_[slot] = false;
  }
  dirty_slots_.clear();
}

}  // namespace stride
