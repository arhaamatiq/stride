#include "safetensors.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace stride {

SafeTensors::SafeTensors(const std::string& path) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open " + path);
  struct stat st {};
  if (::fstat(fd, &st) != 0) {
    ::close(fd);
    throw std::runtime_error("cannot stat " + path);
  }
  size_ = static_cast<size_t>(st.st_size);
  map_ = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (map_ == MAP_FAILED) {
    map_ = nullptr;
    throw std::runtime_error("cannot mmap " + path);
  }
  const auto* base = static_cast<const uint8_t*>(map_);
  if (size_ < 8) throw std::runtime_error("safetensors file too small: " + path);
  uint64_t header_len = 0;
  std::memcpy(&header_len, base, 8);
  if (header_len + 8 > size_) throw std::runtime_error("safetensors header overruns file: " + path);
  const auto header = nlohmann::json::parse(base + 8, base + 8 + header_len);
  const uint8_t* payload = base + 8 + header_len;
  for (auto it = header.begin(); it != header.end(); ++it) {
    if (it.key() == "__metadata__") continue;
    TensorView view;
    view.dtype = it.value().at("dtype").get<std::string>();
    view.shape = it.value().at("shape").get<std::vector<int64_t>>();
    const auto offsets = it.value().at("data_offsets").get<std::vector<uint64_t>>();
    if (offsets.size() != 2 || offsets[1] < offsets[0] || 8 + header_len + offsets[1] > size_) {
      throw std::runtime_error("bad offsets for tensor " + it.key());
    }
    view.data = payload + offsets[0];
    view.bytes = offsets[1] - offsets[0];
    tensors_.emplace(it.key(), std::move(view));
  }
}

SafeTensors::~SafeTensors() {
  if (map_ != nullptr) ::munmap(map_, size_);
}

const TensorView& SafeTensors::get(const std::string& name) const {
  const auto it = tensors_.find(name);
  if (it == tensors_.end()) throw std::runtime_error("missing tensor " + name);
  return it->second;
}

}  // namespace stride
