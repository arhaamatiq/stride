// OpenAI-compatible HTTP server over the scheduler.
//   stride_server --model DIR [--host H] [--port P] [--kv-gb G] [--max-seqs N] [--max-rows N]
//                 [--prefill-chunk N] [--max-new-default N] [--path exact|tc] [--verbose]
// One engine thread owns the engine and steps the scheduler. Request threads
// tokenize, submit, and consume a per-request mailbox; they never touch the
// engine. system_fingerprint is "stride-" plus the scheduler's 64-bit chain over
// the generated (token, logit) pairs, so equal fingerprints mean equal outputs.
#include <httplib.h>

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <random>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "engine.h"
#include "scheduler.h"
#include "tokenizer.h"

using namespace stride;
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

namespace {

struct Args {
  std::string model, host = "0.0.0.0";
  int port = 8000;
  double kv_gb = 1.0;
#if defined(STRIDE_BACKEND_CUDA)
  std::string path = "tc";
#else
  std::string path = "exact";
#endif
  bool norm_tile = false;
  int micro_r = 0;
  int micro_lt = 0;
  int max_seqs = 256;
  int max_rows = 256;
  int prefill_chunk = 64;
  int max_new_default = 256;
  bool verbose = false;
};

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    const std::string k = argv[i];
    const auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", k.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (k == "--model") a.model = next();
    else if (k == "--host") a.host = next();
    else if (k == "--port") a.port = std::stoi(next());
    else if (k == "--kv-gb") a.kv_gb = std::stod(next());
    else if (k == "--path") a.path = next();
    else if (k == "--norm-tile") a.norm_tile = true;
    else if (k == "--micro-r") a.micro_r = std::stoi(next());
    else if (k == "--micro-lt") a.micro_lt = std::stoi(next());
    else if (k == "--max-seqs") a.max_seqs = std::stoi(next());
    else if (k == "--max-rows") a.max_rows = std::stoi(next());
    else if (k == "--prefill-chunk") a.prefill_chunk = std::stoi(next());
    else if (k == "--max-new-default") a.max_new_default = std::stoi(next());
    else if (k == "--verbose") a.verbose = true;
    else {
      std::fprintf(stderr, "unknown flag %s\n", k.c_str());
      std::exit(2);
    }
  }
  if (a.model.empty() || a.max_seqs < 1 || a.max_rows < 1 || a.prefill_chunk < 1 || a.max_new_default < 1 ||
      a.kv_gb <= 0 || a.port < 0 || a.port > 65535) {
    std::fprintf(stderr,
                 "usage: stride_server --model DIR [--host H] [--port P] [--kv-gb G] [--max-seqs N] [--max-rows N]\n"
                 "                     [--prefill-chunk N] [--max-new-default N] [--path exact|tc] [--verbose]\n");
    std::exit(2);
  }
  return a;
}

enum Reason : uint8_t { kNone = 0, kStop = 1, kLength = 2 };

struct Event {
  int token;
  bool finished;
  uint8_t reason;
  uint64_t fp;
};

// Engine thread pushes, the request thread takes. waiting lets the engine side
// skip the wake-up when the consumer is busy.
struct Mailbox {
  std::mutex mu;
  std::condition_variable cv;
  std::atomic<bool> waiting{false};
  std::vector<Event> q;
  std::shared_ptr<std::atomic<bool>> cancelled = std::make_shared<std::atomic<bool>>(false);

  // Moves everything queued into out; waits up to ms for the first event.
  bool take(std::vector<Event>& out, int ms) {
    std::unique_lock<std::mutex> lk(mu);
    if (q.empty()) {
      waiting.store(true);
      cv.wait_for(lk, std::chrono::milliseconds(ms), [this] { return !q.empty(); });
      waiting.store(false);
    }
    out.clear();
    out.swap(q);
    return !out.empty();
  }
  void wake() {
    if (waiting.load()) cv.notify_one();
  }
};

void put_uint(std::string& s, uint64_t v) {
  char b[24];
  s.append(b, std::to_chars(b, b + sizeof(b), v).ptr);
}

void json_escape(std::string& out, const char* p, size_t n) {
  size_t run = 0;
  for (size_t i = 0; i < n; ++i) {
    const uint8_t c = static_cast<uint8_t>(p[i]);
    if (c >= 0x20 && c != '"' && c != '\\') continue;
    out.append(p + run, i - run);
    run = i + 1;
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default: {
        char b[8];
        std::snprintf(b, sizeof(b), "\\u%04x", c);
        out += b;
      }
    }
  }
  out.append(p + run, n - run);
}

void json_escape(std::string& out, const std::string& s) { json_escape(out, s.data(), s.size()); }

std::string fingerprint(uint64_t fp) {
  char b[40];
  std::snprintf(b, sizeof(b), "stride-%016llx", static_cast<unsigned long long>(fp));
  return b;
}

// "Today Date" the way the HF template's strftime_now("%d %b %Y") renders it.
std::string today() {
  static const char* kMonths[] = {"Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"};
  const std::time_t t = std::time(nullptr);
  std::tm lt;
  localtime_r(&t, &lt);
  char b[32];
  std::snprintf(b, sizeof(b), "%02d %s %04d", lt.tm_mday, kMonths[lt.tm_mon], lt.tm_year + 1900);
  return b;
}

// Turns events into releasable text: skips the stop token, decodes with UTF-8
// buffering, and holds back any tail that could still become a stop string.
struct Gen {
  Gen(const Tokenizer& tok, std::vector<std::string> stop_strings) : dec(tok), stops(std::move(stop_strings)) {
    for (const std::string& s : stops) max_stop = std::max(max_stop, s.size());
  }
  StreamDecoder dec;
  std::vector<std::string> stops;
  size_t max_stop = 0;
  std::string held;
  int tokens = 0;
  uint64_t fp = 0;
  const char* finish = nullptr;

  // Appends releasable text to out. True once the request is complete.
  bool feed(const Event& e, std::string& out) {
    if (e.token >= 0) {
      ++tokens;
      fp = e.fp;
    }
    const bool stop_token = e.finished && e.reason == kStop;
    const bool decode = e.token >= 0 && !stop_token;
    if (stops.empty()) {
      if (decode) dec.push(e.token, out);
      if (!e.finished) return false;
      dec.flush(out);
      finish = stop_token ? "stop" : "length";
      return true;
    }
    const size_t scan = held.size() - std::min(held.size(), max_stop - 1);
    if (decode) dec.push(e.token, held);
    if (e.finished) dec.flush(held);
    size_t hit = std::string::npos;
    for (const std::string& s : stops) hit = std::min(hit, held.find(s, scan));
    if (hit != std::string::npos) {
      out.append(held, 0, hit);
      held.clear();
      finish = "stop";
      return true;
    }
    size_t keep = 0;
    if (!e.finished) {
      for (const std::string& s : stops) {
        for (size_t m = std::min(s.size() - 1, held.size()); m > keep; --m) {
          if (held.compare(held.size() - m, m, s, 0, m) == 0) {
            keep = m;
            break;
          }
        }
      }
    }
    out.append(held, 0, held.size() - keep);
    held.erase(0, held.size() - keep);
    if (!e.finished) return false;
    finish = stop_token ? "stop" : "length";
    return true;
  }
};

class App;

// One request in flight. Streaming responses keep it alive from the handler
// into the content provider.
struct Call {
  explicit Call(App* a);
  ~Call();
  App* app;
  bool chat = true, stream = false, usage = false, ignore_eos = false;
  std::string model, id, pre;  // pre: the start of every chunk, up to the choices
  int64_t created = 0;
  std::vector<int> prompt;
  int prompt_tokens = 0;
  int max_new = 0;
  SamplingParams sampling;
  std::vector<std::string> stops;
  std::shared_ptr<Mailbox> box = std::make_shared<Mailbox>();
  std::unique_ptr<Gen> gen;
  std::vector<Event> local;
  std::string buf, text;
  bool role_sent = false, done = false;
  std::atomic<bool> ended{false};
  Clock::time_point t0 = Clock::now(), t_first{};
};

class App {
 public:
  App(const Args& a)
      : args_(a),
        engine_(a.model, engine_options(a)),
        sched_(engine_, sched_options(a)),
        tok_(a.model + "/tokenizer.json") {
    if (tok_.vocab_size() > engine_.config().vocab)
      throw std::runtime_error("tokenizer has more ids than the model vocab");
    load_stop_tokens();
    kv_blocks_ = engine_.kv().total_blocks();  // cached: nothing but the engine thread touches the engine after this
    max_ctx_ = std::min(kMaxCtx, kv_blocks_ * kKvBlockTokens);
    std::string name = a.model;
    while (name.size() > 1 && name.back() == '/') name.pop_back();
    name_ = name.substr(name.find_last_of('/') == std::string::npos ? 0 : name.find_last_of('/') + 1);
    std::random_device rd;
    salt_ = (static_cast<uint64_t>(rd()) << 32) | rd();
    engine_thread_ = std::thread([this] { engine_loop(); });
    notifier_thread_ = std::thread([this] { notifier_loop(); });
  }

  ~App() { stop(); }

  void stop() {
    std::lock_guard<std::mutex> once(stop_mu_);
    if (!running_.exchange(false)) return;
    { std::lock_guard<std::mutex> lk(wake_mu_); }
    wake_cv_.notify_all();
    { std::lock_guard<std::mutex> lk(nmu_); }
    ncv_.notify_all();
    engine_thread_.join();
    notifier_thread_.join();
  }

  bool running() const { return running_.load(); }
  int max_ctx() const { return max_ctx_; }
  int total_blocks() const { return kv_blocks_; }

  void routes(httplib::Server& svr) {
    svr.Post("/v1/chat/completions", [this](const httplib::Request& q, httplib::Response& r) { complete(q, r, true); });
    svr.Post("/v1/completions", [this](const httplib::Request& q, httplib::Response& r) { complete(q, r, false); });
    svr.Get("/v1/models", [this](const httplib::Request&, httplib::Response& r) {
      json m = {{"id", name_},
                {"object", "model"},
                {"created", started_},
                {"owned_by", "stride"},
                {"max_model_len", max_ctx_}};
      r.set_content(json{{"object", "list"}, {"data", json::array({m})}}.dump(), "application/json");
    });
    svr.Get("/health", [](const httplib::Request&, httplib::Response& r) {
      r.set_content("{\"status\":\"ok\"}", "application/json");
    });
    svr.Get("/stats", [this](const httplib::Request&, httplib::Response& r) {
      r.set_content(stats().dump(), "application/json");
    });
  }

  void finish_call(Call& c, bool ok);
  std::atomic<int> active{0};

 private:
  static EngineOptions engine_options(const Args& a) {
    EngineOptions eo;
    eo.max_rows = a.max_rows;
    eo.max_slots = a.max_seqs;
    eo.max_full_rows = std::min(a.max_rows, a.max_seqs);
    eo.kv_bytes = static_cast<int64_t>(a.kv_gb * 1024.0 * 1024.0 * 1024.0);
    eo.path = a.path == "tc" ? KernelPath::kTensorCore : KernelPath::kExact;
    eo.tc_norm_per_lane = !a.norm_tile;
    eo.tc_micro_r = a.micro_r;
    eo.tc_micro_lt = a.micro_lt;
    return eo;
  }
  static SchedulerOptions sched_options(const Args& a) {
    SchedulerOptions so;
    so.max_seqs = a.max_seqs;
    so.max_rows = a.max_rows;
    so.prefill_chunk = a.prefill_chunk;
    return so;
  }

  void load_stop_tokens() {
    stop_tokens_ = engine_.config().eos_tokens;
    std::ifstream in(args_.model + "/generation_config.json");
    if (in) {
      const json j = json::parse(in, nullptr, false);
      if (j.is_object() && j.contains("eos_token_id")) {
        const json& e = j["eos_token_id"];
        if (e.is_number_integer()) stop_tokens_.push_back(e.get<int>());
        else if (e.is_array())
          for (const json& v : e)
            if (v.is_number_integer()) stop_tokens_.push_back(v.get<int>());
      }
    }
    std::sort(stop_tokens_.begin(), stop_tokens_.end());
    stop_tokens_.erase(std::unique(stop_tokens_.begin(), stop_tokens_.end()), stop_tokens_.end());
  }

  // ---- engine side ----

  void engine_loop() {
    while (running_.load()) {
      const uint64_t ticket = submitted_.load();
      bool busy;
      try {
        busy = sched_.step();
      } catch (const std::exception& e) {
        std::fprintf(stderr, "stride_server: engine failure: %s\n", e.what());
        std::_Exit(1);
      }
      flush_touched();
      {
        std::lock_guard<std::mutex> lk(stats_mu_);
        stats_ = sched_.stats();
        free_blocks_ = engine_.kv().free_blocks();
      }
      if (busy) continue;
      std::unique_lock<std::mutex> lk(wake_mu_);
      wake_cv_.wait(lk, [&] { return submitted_.load() != ticket || !running_.load(); });
    }
  }

  // Waking N request threads costs N syscalls; past a few, a helper thread pays
  // for them so the next engine step is not delayed.
  void flush_touched() {
    if (touched_.empty()) return;
    if (touched_.size() <= kDirectWake) {
      for (auto& b : touched_) b->wake();
    } else {
      {
        std::lock_guard<std::mutex> lk(nmu_);
        for (auto& b : touched_) pending_.push_back(std::move(b));
      }
      ncv_.notify_one();
    }
    touched_.clear();
  }

  void notifier_loop() {
    std::vector<std::shared_ptr<Mailbox>> local;
    for (;;) {
      {
        std::unique_lock<std::mutex> lk(nmu_);
        ncv_.wait(lk, [this] { return !pending_.empty() || !running_.load(); });
        if (pending_.empty()) return;
        local.swap(pending_);
      }
      for (auto& b : local) b->wake();
      local.clear();
    }
  }

  void submit(GenRequest&& r) {
    sched_.submit(std::move(r));
    {
      std::lock_guard<std::mutex> lk(wake_mu_);
      submitted_.fetch_add(1);
    }
    wake_cv_.notify_one();
  }

  json stats() {
    SchedulerStats s;
    int free_blocks;
    {
      std::lock_guard<std::mutex> lk(stats_mu_);
      s = stats_;
      free_blocks = free_blocks_;
    }
    return {{"steps", s.steps},
            {"rows", s.rows},
            {"generated", s.generated},
            {"preemptions", s.preemptions},
            {"replayed_rows", s.replayed_rows},
            {"peak_seqs", s.peak_seqs},
            {"peak_blocks", s.peak_blocks},
            {"kv_total_blocks", kv_blocks_},
            {"kv_free_blocks", free_blocks},
            {"max_context", max_ctx_},
            {"requests", requests_.load()},
            {"active", active.load()},
            {"cancelled", cancelled_.load()},
            {"prompt_tokens", prompt_tokens_.load()},
            {"completion_tokens", completion_tokens_.load()}};
  }

  // ---- request side ----

  static void error(httplib::Response& res, int status, const std::string& msg) {
    const json j = {
        {"error", {{"message", msg}, {"type", "invalid_request_error"}, {"param", nullptr}, {"code", nullptr}}}};
    res.status = status;
    res.set_content(j.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
  }

  static bool unset(const json& v) {
    return v.is_null() || (v.is_boolean() && !v.get<bool>()) || (v.is_number() && v.get<double>() == 0) ||
           ((v.is_array() || v.is_object()) && v.empty());
  }

  // Reads the request into c. Returns an error message, empty if fine.
  std::string parse(const json& j, Call& c) {
    if (!j.is_object()) return "body must be a JSON object";
    for (const char* k : {"logprobs", "top_logprobs", "presence_penalty", "frequency_penalty", "logit_bias", "tools",
                          "functions", "best_of", "echo", "suffix"})
      if (j.contains(k) && !unset(j[k])) return std::string("unsupported parameter: ") + k;
    if (j.contains("n") && !(j["n"].is_number() && j["n"].get<double>() == 1)) return "only n=1 is supported";

    const auto number = [&](const char* k, double def, double* out) -> bool {
      *out = def;
      if (!j.contains(k) || j[k].is_null()) return true;
      if (!j[k].is_number()) return false;
      *out = j[k].get<double>();
      return true;
    };
    double temperature, top_p, top_k, max_tokens, max_completion, min_tokens;
    if (!number("temperature", 0.0, &temperature) || temperature < 0) return "temperature must be a number >= 0";
    if (!number("top_p", 1.0, &top_p) || top_p <= 0 || top_p > 1) return "top_p must be in (0, 1]";
    if (!number("top_k", 0, &top_k)) return "top_k must be an integer";
    const bool has_seed = j.contains("seed") && !j["seed"].is_null();
    if (has_seed && !j["seed"].is_number_integer()) return "seed must be an integer";
    if (!number("max_tokens", -1, &max_tokens)) return "max_tokens must be an integer";
    if (!number("max_completion_tokens", -1, &max_completion)) return "max_completion_tokens must be an integer";
    if (!number("min_tokens", 0, &min_tokens) || min_tokens < 0) return "min_tokens must be an integer >= 0";
    if (j.contains("ignore_eos") && !j["ignore_eos"].is_null()) {
      if (!j["ignore_eos"].is_boolean()) return "ignore_eos must be a boolean";
      c.ignore_eos = j["ignore_eos"].get<bool>();
    }
    if (j.contains("stream") && j["stream"].is_boolean()) c.stream = j["stream"].get<bool>();
    if (j.contains("stream_options") && j["stream_options"].is_object())
      c.usage = j["stream_options"].value("include_usage", json()) == true;
    if (j.contains("stop") && !j["stop"].is_null()) {
      const json& s = j["stop"];
      if (s.is_string()) c.stops.push_back(s.get<std::string>());
      else if (s.is_array())
        for (const json& v : s) {
          if (!v.is_string()) return "stop must be a string or an array of strings";
          c.stops.push_back(v.get<std::string>());
        }
      else return "stop must be a string or an array of strings";
      for (const std::string& s2 : c.stops)
        if (s2.empty()) return "stop strings must not be empty";
    }
    c.model = j.contains("model") && j["model"].is_string() ? j["model"].get<std::string>() : name_;

    if (c.chat) {
      if (!j.contains("messages") || !j["messages"].is_array() || j["messages"].empty())
        return "messages must be a non-empty array";
      std::vector<ChatMessage> msgs;
      for (const json& m : j["messages"]) {
        if (!m.is_object() || !m.contains("role") || !m["role"].is_string()) return "each message needs a string role";
        ChatMessage cm{m["role"].get<std::string>(), ""};
        const json content = m.value("content", json());
        if (content.is_string()) {
          cm.content = content.get<std::string>();
        } else if (content.is_array()) {  // [{"type":"text","text":...}]: text parts joined by newlines
          for (const json& part : content) {
            if (!part.is_object() || part.value("type", "") != "text" || !part.contains("text") ||
                !part["text"].is_string())
              return "only text content parts are supported";
            if (!cm.content.empty()) cm.content += "\n";
            cm.content += part["text"].get<std::string>();
          }
        } else if (!content.is_null()) {
          return "message content must be a string or an array of text parts";
        }
        msgs.push_back(std::move(cm));
      }
      try {
        tok_.encode(tok_.apply_chat_template(msgs, true, today()), true, c.prompt);
      } catch (const std::invalid_argument& e) {
        return e.what();
      }
    } else {
      if (!j.contains("prompt")) return "prompt is required";
      const json* p = &j["prompt"];
      if (p->is_array() && p->size() == 1) p = &(*p)[0];
      if (!p->is_string()) return "prompt must be a string";
      if (tok_.bos_id() >= 0) c.prompt.push_back(tok_.bos_id());
      tok_.encode(p->get<std::string>(), true, c.prompt);
    }

    c.prompt_tokens = static_cast<int>(c.prompt.size());
    const int room = max_ctx_ - c.prompt_tokens;
    if (room < 1)
      return "prompt has " + std::to_string(c.prompt_tokens) + " tokens; the context holds " + std::to_string(max_ctx_);
    const double explicit_max = max_completion >= 0 ? max_completion : max_tokens;
    if (explicit_max >= 0) {
      if (explicit_max < 1 || explicit_max != std::floor(explicit_max)) return "max_tokens must be a positive integer";
      if (explicit_max > room)
        return "prompt has " + std::to_string(c.prompt_tokens) + " tokens and max_tokens is " +
               std::to_string(static_cast<int>(explicit_max)) + "; the context holds " + std::to_string(max_ctx_);
      c.max_new = static_cast<int>(explicit_max);
    } else {
      c.max_new = std::min(args_.max_new_default, room);
    }
    if (min_tokens > 0 && !c.ignore_eos) return "min_tokens is only supported together with ignore_eos";
    if (min_tokens > c.max_new) return "min_tokens exceeds max_tokens";

    c.sampling.temperature = static_cast<float>(temperature);
    c.sampling.top_p = static_cast<float>(top_p);
    c.sampling.top_k = top_k > 0 ? static_cast<int>(std::min(top_k, 1e9)) : 0;
    if (has_seed)
      c.sampling.seed =
          j["seed"].is_number_unsigned() ? j["seed"].get<uint64_t>() : static_cast<uint64_t>(j["seed"].get<int64_t>());
    else if (temperature > 0) c.sampling.seed = salt_ ^ (requests_.load() * 0x9E3779B97F4A7C15ULL);
    return "";
  }

  void complete(const httplib::Request& req, httplib::Response& res, bool chat) {
    const json j = json::parse(req.body, nullptr, false);
    if (j.is_discarded()) return error(res, 400, "request body is not valid JSON");
    auto c = std::make_shared<Call>(this);
    c->chat = chat;
    std::string err;
    try {
      err = parse(j, *c);
    } catch (const json::exception& e) {
      err = e.what();
    }
    if (!err.empty()) return error(res, 400, err);

    c->created = static_cast<int64_t>(std::time(nullptr));
    char idb[40];
    std::snprintf(idb, sizeof(idb), "%s-%016llx", chat ? "chatcmpl" : "cmpl",
                  static_cast<unsigned long long>(salt_ ^ (requests_.fetch_add(1) * 0x9E3779B97F4A7C15ULL)));
    c->id = idb;
    c->pre = "data: {\"id\":\"" + c->id + "\",\"object\":\"" + (chat ? "chat.completion.chunk" : "text_completion") +
             "\",\"created\":" + std::to_string(c->created) + ",\"model\":\"";
    json_escape(c->pre, c->model);
    c->pre += "\",";
    c->gen = std::make_unique<Gen>(tok_, c->stops);

    GenRequest g;
    g.prompt = std::move(c->prompt);
    g.max_new_tokens = c->max_new;
    g.sampling = c->sampling;
    if (!c->ignore_eos) g.stop_tokens = stop_tokens_;
    g.cancelled = c->box->cancelled;
    g.on_token = [this, box = c->box](const TokenEvent& e, const float*) {
      const bool stop = e.finish_reason && !std::strcmp(e.finish_reason, "stop");
      const uint8_t reason = !e.finished ? kNone : stop ? kStop : kLength;
      {
        std::lock_guard<std::mutex> lk(box->mu);
        box->q.push_back(Event{e.token, e.finished, reason, e.fingerprint});
      }
      touched_.push_back(box);
    };
    submit(std::move(g));

    if (c->stream) {
      res.set_header("Cache-Control", "no-cache");
      res.set_header("X-Accel-Buffering", "no");
      res.set_chunked_content_provider(
          "text/event-stream", [this, c](size_t, httplib::DataSink& sink) { return pump(*c, sink); },
          [this, c](bool ok) { finish_call(*c, ok); });
    } else {
      collect(req, res, *c);
    }
  }

  // Non-streaming: wait for the whole generation, then answer once.
  void collect(const httplib::Request& req, httplib::Response& res, Call& c) {
    auto last_check = Clock::now();
    bool finished = false;
    while (!finished) {
      if (c.box->take(c.local, 100)) {
        if (c.t_first == Clock::time_point{}) c.t_first = Clock::now();
        for (const Event& ev : c.local)
          if ((finished = c.gen->feed(ev, c.text))) break;
        if (finished) break;
      }
      const auto now = Clock::now();
      const bool check = now - last_check > std::chrono::milliseconds(250);
      if (check) last_check = now;
      if (!running() || (check && req.is_connection_closed())) {
        finish_call(c, false);
        return;
      }
    }
    c.box->cancelled->store(true);  // a stop string can end the request before the engine does
    const Gen& g = *c.gen;
    json usage = {{"prompt_tokens", c.prompt_tokens},
                  {"completion_tokens", g.tokens},
                  {"total_tokens", c.prompt_tokens + g.tokens}};
    json choice = {{"index", 0}, {"finish_reason", g.finish}};
    if (c.chat) {
      choice["message"] = {{"role", "assistant"}, {"content", c.text}};
    } else {
      choice["text"] = c.text;
      choice["logprobs"] = nullptr;
    }
    const json out = {{"id", c.id},
                      {"object", c.chat ? "chat.completion" : "text_completion"},
                      {"created", c.created},
                      {"model", c.model},
                      {"system_fingerprint", fingerprint(g.fp)},
                      {"choices", json::array({choice})},
                      {"usage", usage}};
    res.set_content(out.dump(-1, ' ', false, json::error_handler_t::replace), "application/json");
    finish_call(c, true);
  }

  // Streaming: one call drains everything queued, formats it by hand into the
  // reused buffer and sends it in a single write.
  bool pump(Call& c, httplib::DataSink& sink) {
    if (!c.box->take(c.local, 100)) return running();
    if (c.t_first == Clock::time_point{}) c.t_first = Clock::now();
    c.buf.clear();
    for (const Event& ev : c.local) {
      c.text.clear();
      if (c.gen->feed(ev, c.text)) {
        final_chunks(c);
        c.done = true;
        break;
      }
      if (!c.text.empty()) content_chunk(c);
    }
    if (!c.buf.empty() && !sink.write(c.buf.data(), c.buf.size())) return false;
    if (c.done) {
      c.box->cancelled->store(true);
      sink.done();
    }
    return true;
  }

  static void role_chunk(Call& c) {
    if (!c.chat || c.role_sent) return;
    c.role_sent = true;
    c.buf += c.pre;
    c.buf += "\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},";
    c.buf += "\"finish_reason\":null}]}\n\n";
  }

  static void content_chunk(Call& c) {
    role_chunk(c);
    c.buf += c.pre;
    c.buf += c.chat ? "\"choices\":[{\"index\":0,\"delta\":{\"content\":\"" : "\"choices\":[{\"index\":0,\"text\":\"";
    json_escape(c.buf, c.text);
    c.buf += c.chat ? "\"},\"finish_reason\":null}]}\n\n" : "\",\"logprobs\":null,\"finish_reason\":null}]}\n\n";
  }

  // Last text, finish_reason and fingerprint in one chunk, then usage and [DONE].
  static void final_chunks(Call& c) {
    role_chunk(c);
    const Gen& g = *c.gen;
    const std::string fp = fingerprint(g.fp);
    c.buf += c.pre;
    c.buf += "\"system_fingerprint\":\"" + fp + "\",\"choices\":[{\"index\":0,";
    if (c.chat) {
      if (c.text.empty()) {
        c.buf += "\"delta\":{}";
      } else {
        c.buf += "\"delta\":{\"content\":\"";
        json_escape(c.buf, c.text);
        c.buf += "\"}";
      }
    } else {
      c.buf += "\"text\":\"";
      json_escape(c.buf, c.text);
      c.buf += "\",\"logprobs\":null";
    }
    c.buf += ",\"finish_reason\":\"";
    c.buf += g.finish;
    c.buf += "\"}]}\n\n";
    if (c.usage) {
      c.buf += c.pre;
      c.buf += "\"system_fingerprint\":\"" + fp + "\",\"choices\":[],\"usage\":{\"prompt_tokens\":";
      put_uint(c.buf, c.prompt_tokens);
      c.buf += ",\"completion_tokens\":";
      put_uint(c.buf, g.tokens);
      c.buf += ",\"total_tokens\":";
      put_uint(c.buf, static_cast<uint64_t>(c.prompt_tokens) + g.tokens);
      c.buf += "}}\n\n";
    }
    c.buf += "data: [DONE]\n\n";
  }

  static constexpr size_t kDirectWake = 4;

  Args args_;
  Engine engine_;
  Scheduler sched_;
  Tokenizer tok_;
  std::vector<int> stop_tokens_;
  int max_ctx_ = 0;
  int kv_blocks_ = 0;
  std::string name_;
  uint64_t salt_ = 0;
  int64_t started_ = static_cast<int64_t>(std::time(nullptr));

  std::atomic<bool> running_{true};
  std::thread engine_thread_, notifier_thread_;
  std::mutex wake_mu_;
  std::condition_variable wake_cv_;
  std::atomic<uint64_t> submitted_{0};
  std::vector<std::shared_ptr<Mailbox>> touched_;  // engine thread only
  std::mutex nmu_;
  std::condition_variable ncv_;
  std::vector<std::shared_ptr<Mailbox>> pending_;

  std::mutex stats_mu_;
  SchedulerStats stats_;
  int free_blocks_ = 0;
  std::atomic<uint64_t> requests_{0}, cancelled_{0}, prompt_tokens_{0}, completion_tokens_{0};
  std::mutex stop_mu_;
};

Call::Call(App* a) : app(a) { app->active.fetch_add(1); }
Call::~Call() { app->active.fetch_sub(1); }

// Runs once per request: from the response destructor, or the blocking path.
void App::finish_call(Call& c, bool ok) {
  if (c.ended.exchange(true)) return;
  c.box->cancelled->store(true);
  if (!ok) cancelled_.fetch_add(1);
  prompt_tokens_.fetch_add(c.prompt_tokens);
  completion_tokens_.fetch_add(c.gen ? c.gen->tokens : 0);
  if (!args_.verbose) return;
  const auto now = Clock::now();
  const double total = std::chrono::duration<double, std::milli>(now - c.t0).count();
  const double ttft =
      c.t_first == Clock::time_point{} ? -1.0 : std::chrono::duration<double, std::milli>(c.t_first - c.t0).count();
  std::fprintf(stderr,
               "stride_server: %s id=%s stream=%d prompt=%d completion=%d finish=%s ttft_ms=%.2f total_ms=%.2f %s\n",
               c.chat ? "chat" : "completion", c.id.c_str(), c.stream ? 1 : 0, c.prompt_tokens,
               c.gen ? c.gen->tokens : 0,
               c.gen && c.gen->finish ? c.gen->finish : "-", ttft, total, ok ? "ok" : "disconnected");
}

}  // namespace

int main(int argc, char** argv) {
  const Args args = parse(argc, argv);
  // One descriptor per connection: lift the soft limit so 512+ keep-alive clients fit.
  rlimit rl;
  if (getrlimit(RLIMIT_NOFILE, &rl) == 0 && rl.rlim_cur < rl.rlim_max) {
    rl.rlim_cur = std::min<rlim_t>(rl.rlim_max, 65536);
    setrlimit(RLIMIT_NOFILE, &rl);
  }
  // Block the signals in every thread; one thread waits for them.
  sigset_t sigs;
  sigemptyset(&sigs);
  sigaddset(&sigs, SIGINT);
  sigaddset(&sigs, SIGTERM);
  pthread_sigmask(SIG_BLOCK, &sigs, nullptr);

  std::unique_ptr<App> app;
  try {
    app = std::make_unique<App>(args);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "stride_server: %s\n", e.what());
    return 1;
  }

  httplib::Server svr;
  svr.new_task_queue = [] { return new httplib::ThreadPool(512, 1024); };
  svr.set_tcp_nodelay(true);
  svr.set_keep_alive_max_count(1 << 20);
  svr.set_keep_alive_timeout(60);
  svr.set_write_timeout(30);
  app->routes(svr);

  const int port =
      args.port == 0 ? svr.bind_to_any_port(args.host) : (svr.bind_to_port(args.host, args.port) ? args.port : -1);
  if (port < 0) {
    std::fprintf(stderr, "stride_server: cannot bind %s:%d\n", args.host.c_str(), args.port);
    return 1;
  }
  std::thread signals([&svr, &app, sigs] {
    int sig;
    sigwait(&sigs, &sig);
    app->stop();
    svr.stop();
  });

  std::printf("stride_server: listening on %s:%d (max_context %d, kv_blocks %d)\n", args.host.c_str(), port,
              app->max_ctx(), app->total_blocks());
  std::fflush(stdout);
  svr.listen_after_bind();
  kill(getpid(), SIGTERM);  // wakes the signal thread if the server stopped on its own
  signals.join();
  app->stop();
  return 0;
}
