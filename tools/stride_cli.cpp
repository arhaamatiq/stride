// Batch driver used by the correctness harness and by the bandwidth benchmark.
//   stride_cli --model DIR --input prompts.jsonl --output out.jsonl [options]
// Input lines:  {"id": 0, "tokens": [1, 2, 3]}
// Output lines: {"id": 0, "tokens": [...], "values": ["3f800000", ...], "fingerprint": "..."}
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <random>
#include <string>
#include <vector>

#include "device.h"
#include "engine.h"
#include "numerics.h"
#include "scheduler.h"

using namespace stride;
using json = nlohmann::json;

namespace {

struct Args {
  std::string model, input, output, dump_logits;
  int max_new = 16;
  int concurrency = 1;
  int max_rows = 256;
  int prefill_chunk = 64;
  int max_slots = 256;
  double kv_gb = 1.0;
  uint32_t kv_shuffle = 0;
  uint32_t row_shuffle = 0;
  uint32_t arrival_seed = 0;
  int bug = 0;
  std::string path = "exact";
  bool norm_tile = false;
  int lt_mlp = 0;
  int lt_lm = 0;
  bool full = false;
  int dump_steps = 1;
  bool use_eos = false;
  float temperature = 0.0f;
  float top_p = 1.0f;
  int top_k = 0;
  uint64_t seed = 0;
};

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    const auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << k << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (k == "--model") a.model = next();
    else if (k == "--input") a.input = next();
    else if (k == "--output") a.output = next();
    else if (k == "--dump-logits") a.dump_logits = next();
    else if (k == "--dump-steps") a.dump_steps = std::stoi(next());
    else if (k == "--max-new") a.max_new = std::stoi(next());
    else if (k == "--concurrency") a.concurrency = std::stoi(next());
    else if (k == "--max-rows") a.max_rows = std::stoi(next());
    else if (k == "--prefill-chunk") a.prefill_chunk = std::stoi(next());
    else if (k == "--max-slots") a.max_slots = std::stoi(next());
    else if (k == "--kv-gb") a.kv_gb = std::stod(next());
    else if (k == "--kv-shuffle") a.kv_shuffle = static_cast<uint32_t>(std::stoul(next()));
    else if (k == "--row-shuffle") a.row_shuffle = static_cast<uint32_t>(std::stoul(next()));
    else if (k == "--arrival-seed") a.arrival_seed = static_cast<uint32_t>(std::stoul(next()));
    else if (k == "--bug") a.bug = std::stoi(next());
    else if (k == "--path") a.path = next();
    else if (k == "--norm-tile") a.norm_tile = true;
    else if (k == "--lt-mlp") a.lt_mlp = std::stoi(next());
    else if (k == "--lt-lm") a.lt_lm = std::stoi(next());
    else if (k == "--full") a.full = true;
    else if (k == "--use-eos") a.use_eos = true;
    else if (k == "--temperature") a.temperature = std::stof(next());
    else if (k == "--top-p") a.top_p = std::stof(next());
    else if (k == "--top-k") a.top_k = std::stoi(next());
    else if (k == "--seed") a.seed = std::stoull(next());
    else {
      std::cerr << "unknown flag " << k << "\n";
      std::exit(2);
    }
  }
  if (a.model.empty() || a.input.empty() || a.output.empty()) {
    std::cerr << "usage: stride_cli --model DIR --input FILE --output FILE [options]\n";
    std::exit(2);
  }
  return a;
}

std::string hex32(float v) {
  char buf[16];
  std::snprintf(buf, sizeof(buf), "%08x", as_u32(v));
  return buf;
}

std::string hex64(uint64_t v) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(v));
  return buf;
}

struct Result {
  std::vector<int> tokens;
  std::vector<std::string> values;
  uint64_t fingerprint = 0;
  std::string finish;
  bool finished = false;
};

}  // namespace

int main(int argc, char** argv) {
  const Args args = parse(argc, argv);

  std::vector<std::pair<int64_t, std::vector<int>>> prompts;
  {
    std::ifstream in(args.input);
    if (!in) {
      std::cerr << "cannot open " << args.input << "\n";
      return 2;
    }
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      const auto j = json::parse(line);
      prompts.emplace_back(j.at("id").get<int64_t>(), j.at("tokens").get<std::vector<int>>());
    }
  }

  EngineOptions eo;
  eo.max_rows = args.max_rows;
  eo.max_slots = args.max_slots;
  eo.max_full_rows = std::min(args.max_rows, std::max(args.concurrency, 1));
  eo.kv_bytes = static_cast<int64_t>(args.kv_gb * 1024.0 * 1024.0 * 1024.0);
  eo.kv_shuffle_seed = args.kv_shuffle;
  eo.bug = args.bug;
  eo.path = args.path == "tc" ? KernelPath::kTensorCore : KernelPath::kExact;
  eo.tc_norm_per_lane = !args.norm_tile;
  eo.tc_lt_mlp = args.lt_mlp;
  eo.tc_lt_lm = args.lt_lm;

  const auto t_load0 = std::chrono::steady_clock::now();
  Engine engine(args.model, eo);
  const auto t_load1 = std::chrono::steady_clock::now();

  SchedulerOptions so;
  so.max_seqs = args.concurrency;
  so.max_rows = args.max_rows;
  so.prefill_chunk = args.prefill_chunk;
  so.row_shuffle_seed = args.row_shuffle;
  so.bug = args.bug;
  Scheduler sched(engine, so);

  std::ofstream logits_out;
  if (!args.dump_logits.empty()) logits_out.open(args.dump_logits, std::ios::binary);
  const int vocab = engine.config().vocab;

  // Logits are buffered per request so the dump order never depends on scheduling.
  std::map<int64_t, std::vector<float>> logits_by_id;
  std::map<int64_t, Result> results;

  std::vector<size_t> order(prompts.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = i;
  if (args.arrival_seed != 0) {
    std::mt19937 rng(args.arrival_seed);
    std::shuffle(order.begin(), order.end(), rng);
  }

  for (const size_t index : order) {
    const int64_t id = prompts[index].first;
    GenRequest req;
    req.prompt = prompts[index].second;
    req.max_new_tokens = args.max_new;
    req.full_logits = args.full;
    req.sampling.temperature = args.temperature;
    req.sampling.top_p = args.top_p;
    req.sampling.top_k = args.top_k;
    req.sampling.seed = args.seed + static_cast<uint64_t>(id);
    if (args.use_eos) req.stop_tokens = engine.config().eos_tokens;
    req.on_token = [&, id](const TokenEvent& e, const float* logits) {
      Result& r = results[id];
      if (e.token >= 0) {
        r.tokens.push_back(e.token);
        r.values.push_back(hex32(e.value));
        r.fingerprint = e.fingerprint;
        if (logits != nullptr && logits_out.is_open() && e.index < args.dump_steps) {
          auto& buf = logits_by_id[id];
          buf.insert(buf.end(), logits, logits + vocab);
        }
      }
      if (e.finished) {
        r.finished = true;
        r.finish = e.finish_reason ? e.finish_reason : "";
      }
    };
    sched.submit(std::move(req));
  }

  const auto t0 = std::chrono::steady_clock::now();
  while (sched.step()) {
  }
  const auto t1 = std::chrono::steady_clock::now();

  std::ofstream out(args.output);
  for (const auto& [id, tokens] : prompts) {
    (void)tokens;
    const Result& r = results[id];
    json j;
    j["id"] = id;
    j["tokens"] = r.tokens;
    j["values"] = r.values;
    j["fingerprint"] = hex64(r.fingerprint);
    j["finish"] = r.finish;
    out << j.dump() << "\n";
    if (logits_out.is_open()) {
      const auto& buf = logits_by_id[id];
      logits_out.write(reinterpret_cast<const char*>(buf.data()), buf.size() * sizeof(float));
    }
  }

  const SchedulerStats& s = sched.stats();
  json summary;
  summary["device"] = device_name();
  summary["load_seconds"] = std::chrono::duration<double>(t_load1 - t_load0).count();
  summary["run_seconds"] = std::chrono::duration<double>(t1 - t0).count();
  summary["steps"] = s.steps;
  summary["rows"] = s.rows;
  summary["generated"] = s.generated;
  summary["preemptions"] = s.preemptions;
  summary["peak_seqs"] = s.peak_seqs;
  summary["peak_blocks"] = s.peak_blocks;
  summary["total_blocks"] = engine.kv().total_blocks();
  summary["block_bytes"] = engine.kv().block_bytes();
  summary["weight_bytes_per_pass"] = engine.weights().bytes_per_pass();
  summary["weight_bytes_on_device"] = engine.weights().bytes_on_device();
  std::cerr << summary.dump() << "\n";
  return 0;
}
