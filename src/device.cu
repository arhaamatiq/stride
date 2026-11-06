#include "device.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#if defined(__CUDACC__)
#include <cuda_runtime.h>

namespace stride {

cudaStream_t& kernel_stream();

namespace {
void check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
  }
}
}  // namespace

void device_init() {
  if (kernel_stream() == nullptr) check(cudaStreamCreate(&kernel_stream()), "cudaStreamCreate");
}

void* device_alloc(size_t bytes) {
  void* ptr = nullptr;
  check(cudaMalloc(&ptr, bytes), "cudaMalloc");
  return ptr;
}

void device_free(void* ptr) { cudaFree(ptr); }

void device_upload(void* dst, const void* src, size_t bytes) {
  check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, kernel_stream()), "upload");
  check(cudaStreamSynchronize(kernel_stream()), "upload sync");
}

void device_download(void* dst, const void* src, size_t bytes) {
  check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, kernel_stream()), "download");
  check(cudaStreamSynchronize(kernel_stream()), "download sync");
}

void device_wait() { check(cudaStreamSynchronize(kernel_stream()), "wait"); }

std::string device_name() {
  cudaDeviceProp prop{};
  check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
  return prop.name;
}

}  // namespace stride

#else

namespace stride {

void device_init() {}

void* device_alloc(size_t bytes) {
  void* ptr = nullptr;
  if (posix_memalign(&ptr, 64, bytes) != 0) throw std::runtime_error("posix_memalign failed");
  return ptr;
}

void device_free(void* ptr) { std::free(ptr); }

void device_upload(void* dst, const void* src, size_t bytes) { std::memcpy(dst, src, bytes); }

void device_download(void* dst, const void* src, size_t bytes) { std::memcpy(dst, src, bytes); }

void device_wait() {}

std::string device_name() { return "cpu"; }

}  // namespace stride

#endif
