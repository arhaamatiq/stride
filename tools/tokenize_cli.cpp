// Tokenizer test helper: one JSON object per stdin line, one JSON line out.
//   {"op":"encode","text":"...","special":true}              -> {"ids":[...]}  ("hex":"..." for raw bytes)
//   {"op":"chat","messages":[{"role","content"}],"gen":true,
//    "date":"06 Nov 2026"}                                    -> {"text":"...","ids":[...]}
//   {"op":"decode","ids":[...],"skip":true}                  -> {"text":"...","chunks":[...]}
//   {"op":"time","text":"...","reps":3}                      -> {"seconds":..,"tokens":N}
// decode also streams the ids through StreamDecoder; chunks are what it emitted.
#include <chrono>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>

#include "tokenizer.h"

using namespace stride;
using json = nlohmann::json;

int main(int argc, char** argv) {
  if (argc != 3 || std::string(argv[1]) != "--tokenizer") {
    std::cerr << "usage: tokenize_cli --tokenizer tokenizer.json < requests.jsonl\n";
    return 2;
  }
  try {
    const Tokenizer tok(argv[2]);
    std::string line;
    while (std::getline(std::cin, line)) {
      if (line.empty()) continue;
      const json q = json::parse(line);
      const std::string op = q.at("op");
      json r;
      if (op == "encode") {
        std::string text;
        if (q.contains("hex")) {  // raw bytes, for input that is not valid UTF-8
          const std::string hex = q.at("hex");
          for (size_t i = 0; i + 1 < hex.size(); i += 2)
            text.push_back(static_cast<char>(std::stoi(hex.substr(i, 2), nullptr, 16)));
        } else {
          text = q.at("text").get<std::string>();
        }
        r["ids"] = tok.encode(text, q.value("special", true));
      } else if (op == "chat") {
        std::vector<ChatMessage> msgs;
        for (const json& m : q.at("messages")) msgs.push_back({m.at("role"), m.at("content")});
        const std::string text = tok.apply_chat_template(msgs, q.value("gen", true), q.at("date"));
        r["text"] = text;
        r["ids"] = tok.encode(text, true);
      } else if (op == "decode") {
        const std::vector<int> ids = q.at("ids").get<std::vector<int>>();
        r["text"] = tok.decode(ids, q.value("skip", true));
        StreamDecoder dec(tok);
        json chunks = json::array();
        for (const int id : ids) {
          std::string piece;
          dec.push(id, piece);
          chunks.push_back(piece);
        }
        std::string tail;
        dec.flush(tail);
        chunks.push_back(tail);
        r["chunks"] = chunks;
      } else if (op == "time") {
        const std::string text = q.at("text");
        size_t tokens = 0;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < q.value("reps", 1); ++i) tokens = tok.encode(text, true).size();
        r["seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        r["tokens"] = tokens;
      } else {
        r["error"] = "unknown op " + op;
      }
      std::cout << r.dump(-1, ' ', false, json::error_handler_t::strict) << "\n";
    }
  } catch (const std::exception& e) {
    std::cerr << "tokenize_cli: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
