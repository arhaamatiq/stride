#include "scheduler.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "numerics.h"

namespace stride {

namespace {

constexpr uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;

uint64_t fnv_mix(uint64_t h, uint32_t v) {
  for (int i = 0; i < 4; ++i) {
    h ^= (v >> (i * 8)) & 0xFF;
    h *= kFnvPrime;
  }
  return h;
}

uint64_t splitmix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

}  // namespace

struct Scheduler::Seq {
  GenRequest req;
  std::vector<int> tokens;
  int prompt_len = 0;
  int next_pos = 0;
  int generated = 0;
  int slot = -1;
  uint64_t arrival = 0;
  uint64_t fingerprint = kFnvOffset;
  bool done = false;
  bool wants_full() const { return req.full_logits || req.sampling.temperature > 0.0f; }
};

Scheduler::Scheduler(Engine& engine, const SchedulerOptions& opts)
    : engine_(engine), opts_(opts), rng_state_(opts.row_shuffle_seed) {}

Scheduler::~Scheduler() = default;

void Scheduler::submit(GenRequest request) {
  std::lock_guard<std::mutex> lock(mu_);
  incoming_.push_back(std::move(request));
}

bool Scheduler::idle() {
  std::lock_guard<std::mutex> lock(mu_);
  return incoming_.empty() && waiting_.empty() && running_.empty();
}

void Scheduler::finish(Seq& seq, const char* reason) {
  seq.done = true;
  if (seq.slot >= 0) {
    engine_.kv().release_slot(seq.slot);
    seq.slot = -1;
  }
  (void)reason;
}

void Scheduler::admit() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    while (!incoming_.empty()) {
      auto seq = std::make_unique<Seq>();
      seq->req = std::move(incoming_.front());
      incoming_.pop_front();
      seq->tokens = seq->req.prompt;
      seq->prompt_len = static_cast<int>(seq->tokens.size());
      seq->arrival = arrival_++;
      const int room = kMaxCtx - seq->prompt_len;
      if (room <= 0 || seq->prompt_len == 0) {
        TokenEvent event;
        event.finished = true;
        event.finish_reason = seq->prompt_len == 0 ? "empty_prompt" : "context_length";
        if (seq->req.on_token) seq->req.on_token(event, nullptr);
        continue;
      }
      seq->req.max_new_tokens = std::min(seq->req.max_new_tokens, room);
      waiting_.push_back(std::move(seq));
    }
  }
  while (!waiting_.empty() && static_cast<int>(running_.size()) < opts_.max_seqs) {
    Seq& next = *waiting_.front();
    if (next.req.cancelled && next.req.cancelled->load()) {
      waiting_.pop_front();
      continue;
    }
    // Keep room for one block of growth per running sequence before admitting.
    const int reserve = static_cast<int>(running_.size()) + 1;
    const int need = (static_cast<int>(next.tokens.size()) + kKvBlockTokens - 1) >> kKvBlockShift;
    if (engine_.kv().free_blocks() < need + reserve && !running_.empty()) break;
    const int slot = engine_.kv().acquire_slot();
    if (slot < 0) break;
    next.slot = slot;
    // Replay from the start: the blocks were released on preemption.
    if (opts_.bug != 202) next.next_pos = 0;
    running_.push_back(std::move(waiting_.front()));
    waiting_.pop_front();
  }
  stats_.peak_seqs = std::max(stats_.peak_seqs, static_cast<int>(running_.size()));
}

void Scheduler::preempt_youngest() {
  auto it = std::max_element(running_.begin(), running_.end(),
                             [](const auto& a, const auto& b) { return a->arrival < b->arrival; });
  Seq& victim = **it;
  engine_.kv().release_slot(victim.slot);
  victim.slot = -1;
  ++stats_.preemptions;
  waiting_.push_front(std::move(*it));
  running_.erase(it);
}

bool Scheduler::build_rows() {
  for (;;) {
    rows_.clear();
    row_owner_.clear();
    std::vector<std::pair<Seq*, int>> plan;
    int budget = opts_.max_rows;
    // Decode rows first: a sequence with one unprocessed token is ready to produce.
    for (auto& s : running_) {
      if (static_cast<int>(s->tokens.size()) - s->next_pos == 1 && budget > 0) {
        plan.emplace_back(s.get(), 1);
        --budget;
      }
    }
    for (auto& s : running_) {
      const int remaining = static_cast<int>(s->tokens.size()) - s->next_pos;
      if (remaining > 1 && budget > 0) {
        const int chunk = std::min({remaining, opts_.prefill_chunk, budget});
        plan.emplace_back(s.get(), chunk);
        budget -= chunk;
      }
    }
    if (plan.empty()) return false;

    int need = 0;
    for (const auto& [seq, count] : plan) need += engine_.kv().blocks_needed(seq->slot, seq->next_pos + count);
    if (need > engine_.kv().free_blocks()) {
      if (running_.size() <= 1) throw std::runtime_error("KV arena too small for one sequence");
      preempt_youngest();
      continue;
    }

    for (const auto& [seq, count] : plan) {
      engine_.kv().ensure(seq->slot, seq->next_pos + count);
      for (int i = 0; i < count; ++i) {
        const int pos = seq->next_pos + i;
        StepRow row{};
        row.slot = seq->slot;
        row.pos = (opts_.bug == 201 && i == count - 1 && count > 1) ? pos - 1 : pos;
        row.token = (opts_.bug == 203 && pos > 0 && count == 1) ? seq->tokens[pos - 1] : seq->tokens[pos];
        const bool last = pos == static_cast<int>(seq->tokens.size()) - 1;
        row.mode = !last ? LogitMode::kNone : seq->wants_full() ? LogitMode::kFull : LogitMode::kGreedy;
        rows_.push_back(row);
        row_owner_.push_back(seq);
      }
    }
    if (opts_.row_shuffle_seed != 0) {
      order_.resize(rows_.size());
      std::iota(order_.begin(), order_.end(), 0);
      for (int i = static_cast<int>(order_.size()) - 1; i > 0; --i) {
        rng_state_ = splitmix(rng_state_);
        std::swap(order_[i], order_[rng_state_ % (i + 1)]);
      }
      std::vector<StepRow> shuffled(rows_.size());
      std::vector<Seq*> owners(rows_.size());
      for (size_t i = 0; i < order_.size(); ++i) {
        shuffled[i] = rows_[order_[i]];
        owners[i] = row_owner_[order_[i]];
      }
      rows_.swap(shuffled);
      if (opts_.bug != 204) row_owner_.swap(owners);
    }
    for (const auto& [seq, count] : plan) seq->next_pos += count;
    return true;
  }
}

int Scheduler::choose_token(Seq& seq, const float* logits, float* value) {
  const int vocab = engine_.config().vocab;
  const SamplingParams& sp = seq.req.sampling;
  int best = 0;
  for (int i = 1; i < vocab; ++i)
    if (logits[i] > logits[best]) best = i;
  *value = logits[best];
  if (sp.temperature <= 0.0f) return best;

  // Sampling is keyed on (seed, position), not on a stream, so the draw does not
  // depend on what else the scheduler ran.
  std::vector<int> idx(vocab);
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&](int a, int b) {
    if (logits[a] != logits[b]) return logits[a] > logits[b];
    return a < b;
  });
  int count = vocab;
  if (sp.top_k > 0) count = std::min(count, sp.top_k);
  std::vector<double> probs(count);
  double total = 0.0;
  const double top = static_cast<double>(logits[idx[0]]);
  for (int i = 0; i < count; ++i) {
    probs[i] = std::exp((static_cast<double>(logits[idx[i]]) - top) / sp.temperature);
    total += probs[i];
  }
  if (sp.top_p < 1.0f) {
    double cum = 0.0;
    int kept = 0;
    for (int i = 0; i < count; ++i) {
      cum += probs[i] / total;
      ++kept;
      if (cum >= sp.top_p) break;
    }
    count = kept;
    total = 0.0;
    for (int i = 0; i < count; ++i) total += probs[i];
  }
  const uint64_t r = splitmix(sp.seed ^ (static_cast<uint64_t>(seq.tokens.size()) * 0x9E3779B97F4A7C15ULL));
  const double u = static_cast<double>(r >> 11) * (1.0 / 9007199254740992.0) * total;
  double cum = 0.0;
  for (int i = 0; i < count; ++i) {
    cum += probs[i];
    if (u < cum) {
      *value = logits[idx[i]];
      return idx[i];
    }
  }
  *value = logits[idx[count - 1]];
  return idx[count - 1];
}

bool Scheduler::step() {
  // Drop cancelled sequences before they consume another step.
  for (auto it = running_.begin(); it != running_.end();) {
    if ((*it)->req.cancelled && (*it)->req.cancelled->load()) {
      engine_.kv().release_slot((*it)->slot);
      it = running_.erase(it);
    } else {
      ++it;
    }
  }
  admit();
  if (!build_rows()) return false;

  engine_.step(rows_, greedy_, full_);
  ++stats_.steps;
  stats_.rows += rows_.size();
  stats_.peak_blocks = std::max(stats_.peak_blocks, engine_.kv().total_blocks() - engine_.kv().free_blocks());

  const int vocab = engine_.config().vocab;
  size_t next_greedy = 0;
  size_t next_full = 0;
  for (size_t i = 0; i < rows_.size(); ++i) {
    if (rows_[i].mode == LogitMode::kNone) continue;
    Seq& seq = *row_owner_[i];
    int token;
    float value;
    const float* logits = nullptr;
    if (rows_[i].mode == LogitMode::kGreedy) {
      const BestEntry& e = greedy_[next_greedy++];
      token = e.index;
      value = e.value;
    } else {
      logits = &full_[next_full++ * vocab];
      token = choose_token(seq, logits, &value);
    }
    seq.fingerprint = fnv_mix(fnv_mix(seq.fingerprint, static_cast<uint32_t>(token)), as_u32(value));
    seq.tokens.push_back(token);
    ++seq.generated;
    ++stats_.generated;

    TokenEvent event;
    event.token = token;
    event.value = value;
    event.index = seq.generated - 1;
    event.fingerprint = seq.fingerprint;
    const auto& stops = seq.req.stop_tokens;
    if (std::find(stops.begin(), stops.end(), token) != stops.end()) {
      event.finished = true;
      event.finish_reason = "stop";
    } else if (seq.generated >= seq.req.max_new_tokens - (opts_.bug == 207 ? 1 : 0)) {
      event.finished = true;
      event.finish_reason = "length";
    }
    if (seq.req.on_token) seq.req.on_token(event, logits);
    if (event.finished) finish(seq, event.finish_reason);
  }
  running_.erase(std::remove_if(running_.begin(), running_.end(), [](const auto& s) { return s->done; }),
                 running_.end());
  return true;
}

}  // namespace stride
