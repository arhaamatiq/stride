// Paged KV cache: a host-side block allocator over one device arena. A sequence
// owns a list of fixed-size blocks, so memory follows actual length rather than
// max_context, and block placement has no effect on results.
#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "kernels.h"
#include "model.h"

namespace stride {

class KvCache {
 public:
  // shuffle_seed != 0 randomizes block placement; the harness uses it to prove
  // that results do not depend on where a sequence's pages live.
  KvCache(const ModelConfig& cfg, int64_t arena_bytes, int max_slots, int elem_bytes, uint32_t shuffle_seed = 0);
  ~KvCache();
  KvCache(const KvCache&) = delete;
  KvCache& operator=(const KvCache&) = delete;

  int acquire_slot();
  void release_slot(int slot);

  // Makes sure positions [0, end_pos) are backed. False if the arena is exhausted.
  bool ensure(int slot, int end_pos);
  int blocks_needed(int slot, int end_pos) const;

  int free_blocks() const { return static_cast<int>(free_.size()); }
  int total_blocks() const { return total_blocks_; }
  int free_slots() const { return static_cast<int>(free_slots_.size()); }
  int64_t block_bytes() const { return block_elems_ * elem_bytes_; }

  // Pushes changed table entries to the device.
  void sync();
  KvRef ref() const { return ref_; }

 private:
  KvRef ref_{};
  int64_t block_elems_ = 0;
  int elem_bytes_ = 4;
  int total_blocks_ = 0;
  int max_slots_ = 0;
  int max_blocks_ = 0;
  std::vector<int32_t> table_;
  std::vector<int32_t> free_;
  std::vector<int> free_slots_;
  std::vector<std::vector<int32_t>> owned_;
  std::vector<int> dirty_slots_;
  std::vector<bool> slot_dirty_;
  void* arena_ = nullptr;
  int32_t* device_table_ = nullptr;
  std::mt19937 rng_;
  bool shuffle_ = false;
};

}  // namespace stride
