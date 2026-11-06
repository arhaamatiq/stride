// Memory-mapped safetensors reader.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace stride {

struct TensorView {
  std::string dtype;
  std::vector<int64_t> shape;
  const uint8_t* data = nullptr;
  size_t bytes = 0;
};

class SafeTensors {
 public:
  explicit SafeTensors(const std::string& path);
  ~SafeTensors();
  SafeTensors(const SafeTensors&) = delete;
  SafeTensors& operator=(const SafeTensors&) = delete;

  bool has(const std::string& name) const { return tensors_.count(name) != 0; }
  const TensorView& get(const std::string& name) const;
  size_t file_bytes() const { return size_; }

 private:
  std::map<std::string, TensorView> tensors_;
  void* map_ = nullptr;
  size_t size_ = 0;
};

}  // namespace stride
