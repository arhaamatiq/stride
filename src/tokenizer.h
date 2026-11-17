// Byte-level BPE tokenizer for Llama-3.x, loaded from a HF tokenizer.json.
// Pre-tokenization uses the file's own split regex (PCRE2); encode() output is
// identical to the HF fast tokenizer with add_special_tokens=False.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace stride {

struct ChatMessage {
  std::string role;
  std::string content;
};

class Tokenizer {
 public:
  explicit Tokenizer(const std::string& tokenizer_json);
  ~Tokenizer();
  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;

  // Thread-safe. Appends to out. Added special tokens ("<|eot_id|>") are matched
  // first (longest match) and never split when allow_special; otherwise their
  // text is tokenized like any other text.
  void encode(std::string_view text, bool allow_special, std::vector<int>& out) const;
  std::vector<int> encode(std::string_view text, bool allow_special) const;

  // Concatenated token bytes with invalid UTF-8 replaced by U+FFFD.
  std::string decode(const std::vector<int>& ids, bool skip_special = true) const;

  std::string_view token_bytes(int id) const;  // empty if id is out of range
  bool is_special(int id) const;
  int vocab_size() const;                      // largest id + 1
  int token_id(std::string_view content) const;  // added token by exact text, else -1
  int bos_id() const;                          // "<|begin_of_text|>" or -1

  // Llama 3.x template for system/user/assistant messages. date is the
  // "Today Date" string (HF: strftime "%d %b %Y"). Throws on tool/ipython roles.
  std::string apply_chat_template(const std::vector<ChatMessage>& messages, bool add_generation_prompt,
                                  const std::string& date) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};

// Incremental detokenizer. A token that ends inside a multi-byte character is
// held back, so every string handed out is well-formed UTF-8. Special tokens
// produce no text.
class StreamDecoder {
 public:
  explicit StreamDecoder(const Tokenizer& tok) : tok_(&tok) {}
  void push(int id, std::string& out);   // appends the text now complete
  void flush(std::string& out);          // held partial character becomes U+FFFD

 private:
  const Tokenizer* tok_;
  std::string held_;
  std::string tmp_;
};

}  // namespace stride
