#include "tokenizer.h"

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace stride {

using json = nlohmann::json;

namespace {

constexpr char kRepl[] = "\xEF\xBF\xBD";
constexpr uint64_t kNoKey = ~0ULL;
constexpr uint8_t kSpecial = 1;  // added token with special=true: hidden in decode
constexpr char kGpt2Pattern[] = "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

// Classifies the sequence at p[0..n) and returns its length. kind: 0 valid
// character, 1 invalid (a maximal subpart, to be replaced), 2 truncated prefix
// of a valid character.
size_t utf8_unit(const uint8_t* p, size_t n, int& kind) {
  const uint8_t c = p[0];
  kind = 0;
  if (c < 0x80) return 1;
  size_t need;
  uint8_t lo = 0x80, hi = 0xBF;
  if (c >= 0xC2 && c <= 0xDF) {
    need = 1;
  } else if (c >= 0xE0 && c <= 0xEF) {
    need = 2;
    if (c == 0xE0) lo = 0xA0;
    else if (c == 0xED) hi = 0x9F;
  } else if (c >= 0xF0 && c <= 0xF4) {
    need = 3;
    if (c == 0xF0) lo = 0x90;
    else if (c == 0xF4) hi = 0x8F;
  } else {
    kind = 1;
    return 1;
  }
  for (size_t j = 1; j <= need; ++j) {
    if (j >= n) {
      kind = 2;
      return j;
    }
    if (p[j] < lo || p[j] > hi) {
      kind = 1;
      return j;
    }
    lo = 0x80;
    hi = 0xBF;
  }
  return need + 1;
}

// Appends s[0..n) as well-formed UTF-8, replacing invalid bytes with U+FFFD. A
// truncated character at the end is left unconsumed unless final; the return
// value is the number of bytes left.
size_t utf8_append(const char* s, size_t n, std::string& out, bool final) {
  const uint8_t* p = reinterpret_cast<const uint8_t*>(s);
  size_t i = 0;
  while (i < n) {
    if (p[i] < 0x80) {
      size_t j = i + 1;
      while (j < n && p[j] < 0x80) ++j;
      out.append(s + i, j - i);
      i = j;
      continue;
    }
    int kind;
    const size_t len = utf8_unit(p + i, n - i, kind);
    if (kind == 0) out.append(s + i, len);
    else if (kind == 1 || final) out.append(kRepl, 3);
    else return n - i;
    i += len;
  }
  return 0;
}

bool utf8_valid(const char* s, size_t n) {
  const uint8_t* p = reinterpret_cast<const uint8_t*>(s);
  for (size_t i = 0; i < n;) {
    if (p[i] < 0x80) {
      ++i;
      continue;
    }
    int kind;
    const size_t len = utf8_unit(p + i, n - i, kind);
    if (kind != 0) return false;
    i += len;
  }
  return true;
}

// Code point at s[i], or length 0 if the bytes there are not a valid character.
size_t cp_at(const std::string& s, size_t i, uint32_t& cp) {
  const uint8_t* p = reinterpret_cast<const uint8_t*>(s.data()) + i;
  int kind;
  const size_t len = utf8_unit(p, s.size() - i, kind);
  if (kind != 0) return 0;
  cp = len == 1 ? p[0] : len == 2 ? p[0] & 0x1F : len == 3 ? p[0] & 0x0F : p[0] & 0x07;
  for (size_t k = 1; k < len; ++k) cp = (cp << 6) | (p[k] & 0x3F);
  return len;
}

// Python str.isspace(), which is what Jinja's |trim strips.
bool py_space(uint32_t c) {
  return (c >= 9 && c <= 13) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680 ||
         (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

std::string py_strip(const std::string& s) {
  size_t b = 0, e = s.size();
  while (b < e) {
    uint32_t cp;
    const size_t len = cp_at(s, b, cp);
    if (len == 0 || !py_space(cp)) break;
    b += len;
  }
  while (e > b) {
    size_t j = e - 1;
    while (j > b && (static_cast<uint8_t>(s[j]) & 0xC0) == 0x80) --j;
    uint32_t cp;
    const size_t len = cp_at(s, j, cp);
    if (len != e - j || !py_space(cp)) break;
    e = j;
  }
  return s.substr(b, e - b);
}

// GPT-2 byte <-> code point map used by byte-level BPE vocab strings.
struct ByteMap {
  int from_cp[324];
  int to_cp[256];
  ByteMap() {
    std::fill(from_cp, from_cp + 324, -1);
    int extra = 0;
    for (int b = 0; b < 256; ++b) {
      const bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
      to_cp[b] = printable ? b : 256 + extra++;
      from_cp[to_cp[b]] = b;
    }
  }
};

// Vocab string -> raw bytes. False if it holds a character outside the map.
bool unmap(const ByteMap& map, const std::string& s, std::string& out) {
  out.clear();
  for (size_t i = 0; i < s.size(); ++i) {
    const uint8_t c = s[i];
    int cp = c;
    if (c >= 0x80) {
      if ((c >> 5) != 6 || i + 1 >= s.size()) return false;
      cp = ((c & 0x1F) << 6) | (s[++i] & 0x3F);
    }
    if (cp >= 324 || map.from_cp[cp] < 0) return false;
    out.push_back(static_cast<char>(map.from_cp[cp]));
  }
  return true;
}

// The pattern is Oniguruma's. PCRE2 without UCP has ASCII-only \s, and with UCP
// it also counts U+180E, so \s is spelled out as the Unicode White_Space set.
std::string translate_pattern(const std::string& pat) {
  static const std::string kSpace =
      "\\x{9}-\\x{d}\\x{20}\\x{85}\\x{a0}\\x{1680}\\x{2000}-\\x{200a}\\x{2028}\\x{2029}\\x{202f}\\x{205f}\\x{3000}";
  std::string out;
  bool in_class = false;
  for (size_t i = 0; i < pat.size(); ++i) {
    const char c = pat[i];
    if (c == '\\' && i + 1 < pat.size()) {
      const char d = pat[++i];
      if (d == 's') out += in_class ? kSpace : "[" + kSpace + "]";
      else if (d == 'S' && !in_class) out += "[^" + kSpace + "]";
      else if (d == 'S') throw std::runtime_error("\\S inside a class is not supported");
      else {
        out += c;
        out += d;
      }
      continue;
    }
    if (c == '[') in_class = true;
    else if (c == ']') in_class = false;
    out += c;
  }
  return out;
}

std::string find_split_pattern(const json& pre) {
  if (pre.is_null()) throw std::runtime_error("tokenizer.json has no pre_tokenizer");
  const std::string type = pre.value("type", "");
  if (type == "Sequence") {
    std::string found;
    for (const json& p : pre.at("pretokenizers")) {
      const std::string sub = find_split_pattern(p);
      if (found.empty()) found = sub;
    }
    if (found.empty()) throw std::runtime_error("no split pattern in pre_tokenizer");
    return found;
  }
  if (type == "Split") {
    if (!pre.at("pattern").contains("Regex")) throw std::runtime_error("Split pattern must be a Regex");
    if (pre.value("behavior", "") != "Isolated" || pre.value("invert", false))
      throw std::runtime_error("unsupported Split behavior");
    return pre.at("pattern").at("Regex").get<std::string>();
  }
  if (type == "ByteLevel") {
    if (pre.value("add_prefix_space", false)) throw std::runtime_error("ByteLevel add_prefix_space is not supported");
    return pre.value("use_regex", true) ? kGpt2Pattern : "";
  }
  throw std::runtime_error("unsupported pre_tokenizer " + type);
}

struct Matcher {
  uint64_t owner = 0;
  pcre2_match_data* md = nullptr;
  pcre2_match_context* mc = nullptr;
  pcre2_jit_stack* js = nullptr;
  void reset() {
    if (md) pcre2_match_data_free(md);
    if (mc) pcre2_match_context_free(mc);
    if (js) pcre2_jit_stack_free(js);
    md = nullptr;
    mc = nullptr;
    js = nullptr;
  }
  ~Matcher() { reset(); }
};

struct Merge {
  uint64_t key;
  uint32_t rank;
  int32_t id;
};

struct Cand {
  uint32_t rank;
  int32_t pos, left, right, merged;
};

struct Scratch {
  std::vector<int32_t> id, next, prev;
  std::vector<Cand> heap;
};

}  // namespace

struct Tokenizer::Impl {
  struct Node {
    std::vector<std::pair<uint8_t, int>> kids;
    int token = -1;
  };

  uint64_t serial = 0;
  std::string pool;              // token bytes, in id order
  std::vector<uint32_t> off;     // id -> offset into pool; size n + 1
  std::vector<uint8_t> flags;
  std::vector<uint32_t> vtab;    // bytes -> id + 1 (0 empty), open addressing
  size_t vmask = 0;
  std::vector<Merge> mtab;
  size_t mmask = 0;
  int mshift = 0;
  int byte_id[256];
  bool ignore_merges = false;
  std::vector<Node> trie;
  bool first[256] = {};
  std::unordered_map<std::string, int> added;
  pcre2_code* code = nullptr;
  bool jit = false;

  ~Impl() {
    if (code) pcre2_code_free(code);
  }

  void load(const std::string& path);
  int find(const char* p, size_t n) const;
  const Merge* find_merge(int32_t a, int32_t b) const;
  int child(int node, uint8_t c) const;
  Matcher& matcher() const;
  int match(Matcher& m, const char* s, size_t n, size_t pos) const;
  void ordinary(const char* p, size_t n, std::vector<int>& out) const;
  void word(const char* p, size_t n, std::vector<int>& out) const;
  void bpe(const uint8_t* w, size_t n, std::vector<int>& out) const;
};

void Tokenizer::Impl::load(const std::string& path) {
  static std::atomic<uint64_t> next_serial{1};
  serial = next_serial++;
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::stringstream ss;
  ss << in.rdbuf();
  const json j = json::parse(ss.str());
  ss.str(std::string());

  if (!j.value("normalizer", json()).is_null()) throw std::runtime_error("tokenizer normalizers are not supported");
  const json& model = j.at("model");
  if (model.value("type", "") != "BPE") throw std::runtime_error("tokenizer model must be BPE");
  ignore_merges = model.value("ignore_merges", false);

  const ByteMap map;
  std::vector<std::string> toks;
  std::vector<uint8_t> in_vocab;
  auto put = [&](size_t id, std::string bytes, bool mapped) {
    if (id >= toks.size()) {
      toks.resize(id + 1);
      in_vocab.resize(id + 1, 0);
      flags.resize(id + 1, 0);
    }
    toks[id] = std::move(bytes);
    in_vocab[id] = mapped;
  };
  std::string raw;
  for (auto it = model.at("vocab").begin(); it != model.at("vocab").end(); ++it) {
    const bool mapped = unmap(map, it.key(), raw);
    put(it.value().get<size_t>(), mapped ? raw : it.key(), mapped);
  }
  trie.emplace_back();
  for (const json& a : j.value("added_tokens", json::array())) {
    const std::string content = a.at("content").get<std::string>();
    const size_t id = a.at("id").get<size_t>();
    if (content.empty()) continue;
    put(id, content, false);
    if (a.value("special", false)) flags[id] |= kSpecial;
    added[content] = static_cast<int>(id);
    int node = 0;
    for (const char ch : content) {
      const uint8_t c = ch;
      int next = child(node, c);
      if (next < 0) {
        next = static_cast<int>(trie.size());
        trie.emplace_back();
        trie[node].kids.emplace_back(c, next);
      }
      node = next;
    }
    trie[node].token = static_cast<int>(id);
    first[static_cast<uint8_t>(content[0])] = true;
  }

  const size_t n = toks.size();
  off.assign(n + 1, 0);
  for (size_t i = 0; i < n; ++i) off[i + 1] = off[i] + static_cast<uint32_t>(toks[i].size());
  pool.reserve(off[n]);
  for (const std::string& t : toks) pool += t;

  size_t cap = 16;
  while (cap < n * 2) cap <<= 1;
  vtab.assign(cap, 0);
  vmask = cap - 1;
  for (size_t id = 0; id < n; ++id) {
    if (!in_vocab[id]) continue;
    size_t h = std::hash<std::string_view>{}(std::string_view(toks[id])) & vmask;
    while (vtab[h] != 0) h = (h + 1) & vmask;
    vtab[h] = static_cast<uint32_t>(id + 1);
  }
  for (int b = 0; b < 256; ++b) {
    const char c = static_cast<char>(b);
    byte_id[b] = find(&c, 1);
    if (byte_id[b] < 0) throw std::runtime_error("vocab lacks the byte-level alphabet");
  }

  const json& merges = model.at("merges");
  cap = 16;
  while (cap < merges.size() * 3 / 2) cap <<= 1;
  mtab.assign(cap, Merge{kNoKey, 0, 0});
  mmask = cap - 1;
  mshift = 64;
  for (size_t c = cap; c > 1; c >>= 1) --mshift;
  std::string a, b;
  uint32_t rank = 0;
  for (const json& m : merges) {
    std::string sa, sb;
    if (m.is_array()) {
      sa = m.at(0).get<std::string>();
      sb = m.at(1).get<std::string>();
    } else {
      const std::string s = m.get<std::string>();
      const size_t sp = s.find(' ');
      if (sp == std::string::npos) throw std::runtime_error("malformed merge " + s);
      sa = s.substr(0, sp);
      sb = s.substr(sp + 1);
    }
    if (!unmap(map, sa, a) || !unmap(map, sb, b)) throw std::runtime_error("merge outside the byte alphabet");
    const int ia = find(a.data(), a.size()), ib = find(b.data(), b.size());
    const std::string ab = a + b;
    const int ic = find(ab.data(), ab.size());
    if (ia < 0 || ib < 0 || ic < 0) throw std::runtime_error("merge refers to a token missing from the vocab");
    const uint64_t key = (static_cast<uint64_t>(ia) << 32) | static_cast<uint32_t>(ib);
    size_t h = (key * 0x9E3779B97F4A7C15ULL) >> mshift;
    while (mtab[h].key != kNoKey && mtab[h].key != key) h = (h + 1) & mmask;
    mtab[h] = Merge{key, rank++, ic};
  }

  std::string pattern = translate_pattern(find_split_pattern(j.at("pre_tokenizer")));
  int err;
  PCRE2_SIZE err_off;
  code = pcre2_compile(reinterpret_cast<PCRE2_SPTR>(pattern.data()), pattern.size(), PCRE2_UTF, &err, &err_off,
                       nullptr);
  if (!code) {
    PCRE2_UCHAR msg[256];
    pcre2_get_error_message(err, msg, sizeof(msg));
    throw std::runtime_error(std::string("split pattern: ") + reinterpret_cast<char*>(msg));
  }
  jit = pcre2_jit_compile(code, PCRE2_JIT_COMPLETE) == 0;
}

int Tokenizer::Impl::find(const char* p, size_t n) const {
  size_t h = std::hash<std::string_view>{}(std::string_view(p, n)) & vmask;
  for (;; h = (h + 1) & vmask) {
    const uint32_t v = vtab[h];
    if (v == 0) return -1;
    const size_t id = v - 1;
    if (off[id + 1] - off[id] == n && std::memcmp(pool.data() + off[id], p, n) == 0) return static_cast<int>(id);
  }
}

const Merge* Tokenizer::Impl::find_merge(int32_t a, int32_t b) const {
  const uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) | static_cast<uint32_t>(b);
  for (size_t h = (key * 0x9E3779B97F4A7C15ULL) >> mshift;; h = (h + 1) & mmask) {
    if (mtab[h].key == key) return &mtab[h];
    if (mtab[h].key == kNoKey) return nullptr;
  }
}

int Tokenizer::Impl::child(int node, uint8_t c) const {
  for (const auto& kid : trie[node].kids)
    if (kid.first == c) return kid.second;
  return -1;
}

Matcher& Tokenizer::Impl::matcher() const {
  thread_local Matcher m;
  if (m.owner != serial) {
    m.reset();
    m.owner = serial;
    m.md = pcre2_match_data_create_from_pattern(code, nullptr);
    m.mc = pcre2_match_context_create(nullptr);
    pcre2_set_match_limit(m.mc, UINT32_MAX);
    pcre2_set_heap_limit(m.mc, UINT32_MAX);
    if (jit) {
      m.js = pcre2_jit_stack_create(32 * 1024, 1 << 20, nullptr);
      pcre2_jit_stack_assign(m.mc, nullptr, m.js);
    }
  }
  return m;
}

int Tokenizer::Impl::match(Matcher& m, const char* s, size_t n, size_t pos) const {
  const PCRE2_SPTR subject = reinterpret_cast<PCRE2_SPTR>(s);
  int rc = jit ? pcre2_jit_match(code, subject, n, pos, PCRE2_NO_UTF_CHECK, m.md, m.mc)
               : pcre2_match(code, subject, n, pos, PCRE2_NO_UTF_CHECK, m.md, m.mc);
  if (rc == PCRE2_ERROR_JIT_STACKLIMIT)
    rc = pcre2_match(code, subject, n, pos, PCRE2_NO_UTF_CHECK | PCRE2_NO_JIT, m.md, m.mc);
  return rc;
}

void Tokenizer::Impl::ordinary(const char* p, size_t n, std::vector<int>& out) const {
  std::string clean;
  if (!utf8_valid(p, n)) {
    utf8_append(p, n, clean, true);
    p = clean.data();
    n = clean.size();
  }
  Matcher& m = matcher();
  size_t pos = 0;
  while (pos < n) {
    const int rc = match(m, p, n, pos);
    if (rc == PCRE2_ERROR_NOMATCH) {
      word(p + pos, n - pos, out);
      break;
    }
    if (rc < 0) throw std::runtime_error("pcre2 match error " + std::to_string(rc));
    const PCRE2_SIZE* ov = pcre2_get_ovector_pointer(m.md);
    const size_t s = ov[0];
    size_t e = ov[1];
    if (s > pos) word(p + pos, s - pos, out);
    if (e == s) {  // cannot happen with this pattern; never loop
      e = s + 1;
      while (e < n && (static_cast<uint8_t>(p[e]) & 0xC0) == 0x80) ++e;
    }
    word(p + s, e - s, out);
    pos = e;
  }
}

void Tokenizer::Impl::word(const char* p, size_t n, std::vector<int>& out) const {
  if (ignore_merges) {
    const int id = find(p, n);
    if (id >= 0) {
      out.push_back(id);
      return;
    }
  }
  const uint8_t* w = reinterpret_cast<const uint8_t*>(p);
  if (n == 1) out.push_back(byte_id[w[0]]);
  else bpe(w, n, out);
}

// Lowest-rank pair first, leftmost on ties, as in HF. A linked list plus a heap
// with lazy deletion keeps long words O(n log n).
void Tokenizer::Impl::bpe(const uint8_t* w, size_t n, std::vector<int>& out) const {
  thread_local Scratch s;
  s.id.resize(n);
  s.next.resize(n);
  s.prev.resize(n);
  s.heap.clear();
  for (size_t i = 0; i < n; ++i) {
    s.id[i] = byte_id[w[i]];
    s.next[i] = i + 1 < n ? static_cast<int32_t>(i + 1) : -1;
    s.prev[i] = static_cast<int32_t>(i) - 1;
  }
  const auto later = [](const Cand& x, const Cand& y) { return x.rank != y.rank ? x.rank > y.rank : x.pos > y.pos; };
  const auto cand = [&](int32_t pos) {
    const int32_t j = s.next[pos];
    if (j < 0) return false;
    const Merge* m = find_merge(s.id[pos], s.id[j]);
    if (!m) return false;
    s.heap.push_back(Cand{m->rank, pos, s.id[pos], s.id[j], m->id});
    return true;
  };
  for (size_t i = 0; i + 1 < n; ++i) cand(static_cast<int32_t>(i));
  std::make_heap(s.heap.begin(), s.heap.end(), later);
  while (!s.heap.empty()) {
    std::pop_heap(s.heap.begin(), s.heap.end(), later);
    const Cand c = s.heap.back();
    s.heap.pop_back();
    const int32_t i = c.pos;
    const int32_t j = s.next[i];
    if (s.id[i] != c.left || j < 0 || s.id[j] != c.right) continue;
    s.id[i] = c.merged;
    s.id[j] = -1;
    s.next[i] = s.next[j];
    if (s.next[i] >= 0) s.prev[s.next[i]] = i;
    if (s.prev[i] >= 0 && cand(s.prev[i])) std::push_heap(s.heap.begin(), s.heap.end(), later);
    if (cand(i)) std::push_heap(s.heap.begin(), s.heap.end(), later);
  }
  for (int32_t i = 0; i >= 0; i = s.next[i]) out.push_back(s.id[i]);
  if (n > (1u << 16)) s = Scratch();  // do not keep megabytes per thread after one huge word
}

Tokenizer::Tokenizer(const std::string& tokenizer_json) : p_(new Impl) { p_->load(tokenizer_json); }

Tokenizer::~Tokenizer() = default;

void Tokenizer::encode(std::string_view text, bool allow_special, std::vector<int>& out) const {
  const Impl& m = *p_;
  const char* s = text.data();
  const size_t n = text.size();
  size_t i = 0, last = 0;
  while (i < n) {
    if (!m.first[static_cast<uint8_t>(s[i])]) {
      ++i;
      continue;
    }
    int node = 0, best = -1;
    size_t best_len = 0;
    for (size_t j = i; j < n; ++j) {
      node = m.child(node, static_cast<uint8_t>(s[j]));
      if (node < 0) break;
      const int t = m.trie[node].token;
      if (t >= 0 && (allow_special || !(m.flags[t] & kSpecial))) {
        best = t;
        best_len = j - i + 1;
      }
    }
    if (best < 0) {
      ++i;
      continue;
    }
    if (i > last) m.ordinary(s + last, i - last, out);
    out.push_back(best);
    i += best_len;
    last = i;
  }
  if (last < n) m.ordinary(s + last, n - last, out);
}

std::vector<int> Tokenizer::encode(std::string_view text, bool allow_special) const {
  std::vector<int> out;
  encode(text, allow_special, out);
  return out;
}

std::string_view Tokenizer::token_bytes(int id) const {
  if (id < 0 || static_cast<size_t>(id) + 1 >= p_->off.size()) return {};
  return std::string_view(p_->pool.data() + p_->off[id], p_->off[id + 1] - p_->off[id]);
}

bool Tokenizer::is_special(int id) const {
  return id >= 0 && static_cast<size_t>(id) < p_->flags.size() && (p_->flags[id] & kSpecial);
}

int Tokenizer::vocab_size() const { return static_cast<int>(p_->off.size()) - 1; }

int Tokenizer::token_id(std::string_view content) const {
  const auto it = p_->added.find(std::string(content));
  return it == p_->added.end() ? -1 : it->second;
}

int Tokenizer::bos_id() const { return token_id("<|begin_of_text|>"); }

std::string Tokenizer::decode(const std::vector<int>& ids, bool skip_special) const {
  std::string bytes, out;
  for (const int id : ids) {
    if (skip_special && is_special(id)) continue;
    bytes += token_bytes(id);
  }
  utf8_append(bytes.data(), bytes.size(), out, true);
  return out;
}

std::string Tokenizer::apply_chat_template(const std::vector<ChatMessage>& messages, bool add_generation_prompt,
                                           const std::string& date) const {
  std::string s = "<|begin_of_text|>";
  size_t i = 0;
  std::string system;
  if (!messages.empty() && messages[0].role == "system") {
    system = py_strip(messages[0].content);
    i = 1;
  }
  s += "<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: " + date +
       "\n\n" + system + "<|eot_id|>";
  for (; i < messages.size(); ++i) {
    const ChatMessage& m = messages[i];
    if (m.role == "tool" || m.role == "ipython") throw std::invalid_argument("unsupported role '" + m.role + "'");
    s += "<|start_header_id|>" + m.role + "<|end_header_id|>\n\n" + py_strip(m.content) + "<|eot_id|>";
  }
  if (add_generation_prompt) s += "<|start_header_id|>assistant<|end_header_id|>\n\n";
  return s;
}

void StreamDecoder::push(int id, std::string& out) {
  if (tok_->is_special(id)) return;
  const std::string_view b = tok_->token_bytes(id);
  if (held_.empty()) {
    const size_t keep = utf8_append(b.data(), b.size(), out, false);
    held_.assign(b.data() + b.size() - keep, keep);
    return;
  }
  tmp_.assign(held_);
  tmp_.append(b.data(), b.size());
  const size_t keep = utf8_append(tmp_.data(), tmp_.size(), out, false);
  held_.assign(tmp_.data() + tmp_.size() - keep, keep);
}

void StreamDecoder::flush(std::string& out) {
  utf8_append(held_.data(), held_.size(), out, true);
  held_.clear();
}

}  // namespace stride
