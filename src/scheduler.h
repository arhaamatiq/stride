// Continuous batching. Every step runs the union of: one decode row per sequence
// that is ready to decode, plus prefill rows up to the step budget. Sequences join
// and leave between steps. When the KV arena is full the youngest sequence is
// preempted and later replayed; replay is bit-exact, so preemption cannot change
// any output a client has already seen.
#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "engine.h"

namespace stride {

struct SamplingParams {
  float temperature = 0.0f;
  float top_p = 1.0f;
  int top_k = 0;
  uint64_t seed = 0;
};

struct TokenEvent {
  int token = -1;
  float value = 0.0f;
  int index = 0;  // 0-based generated-token index
  bool finished = false;
  const char* finish_reason = nullptr;
  uint64_t fingerprint = 0;
};

struct GenRequest {
  std::vector<int> prompt;
  int max_new_tokens = 16;
  SamplingParams sampling;
  std::vector<int> stop_tokens;
  // Request kFull logits for every step and pass them to on_token.
  bool full_logits = false;
  std::function<void(const TokenEvent&, const float* logits)> on_token;
  std::shared_ptr<std::atomic<bool>> cancelled;
};

struct SchedulerOptions {
  int max_seqs = 256;
  int max_rows = 256;
  int prefill_chunk = 64;
  uint32_t row_shuffle_seed = 0;  // harness only: randomizes lane placement
  int bug = 0;
};

struct SchedulerStats {
  uint64_t steps = 0;
  uint64_t rows = 0;
  uint64_t generated = 0;
  uint64_t preemptions = 0;
  uint64_t replayed_rows = 0;
  int peak_seqs = 0;
  int peak_blocks = 0;
};

class Scheduler {
 public:
  Scheduler(Engine& engine, const SchedulerOptions& opts);
  ~Scheduler();

  // Thread-safe.
  void submit(GenRequest request);
  // Runs one engine step. Returns false when nothing is queued or running.
  bool step();
  bool idle();
  const SchedulerStats& stats() const { return stats_; }

 private:
  struct Seq;
  void admit();
  bool build_rows();
  void preempt_youngest();
  void finish(Seq& seq, const char* reason);
  int choose_token(Seq& seq, const float* logits, float* value);

  Engine& engine_;
  SchedulerOptions opts_;
  SchedulerStats stats_;

  std::mutex mu_;
  std::deque<GenRequest> incoming_;

  std::deque<std::unique_ptr<Seq>> waiting_;
  std::vector<std::unique_ptr<Seq>> running_;

  std::vector<StepRow> rows_;
  std::vector<Seq*> row_owner_;
  std::vector<BestEntry> greedy_;
  std::vector<float> full_;
  std::vector<int> order_;
  uint64_t rng_state_ = 0;
  uint64_t arrival_ = 0;
};

}  // namespace stride
