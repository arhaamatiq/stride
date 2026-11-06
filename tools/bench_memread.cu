// Read-bandwidth microbenchmark: how do thread and block layouts over a packed
// weight matrix affect achieved DRAM throughput? The answer sets the GEMM layout.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

struct P {
  const uint4* w;
  int groups;    // row groups
  int row16;     // 16-byte units per row group
  int threads;   // threads per block
  uint32_t* out;
};

// A: one thread owns a whole row group and reads it sequentially.
__global__ void read_thread_per_row(P p) {
  const int item = blockIdx.x * blockDim.x + threadIdx.x;
  if (item >= p.groups) return;
  const uint4* r = p.w + static_cast<int64_t>(item) * p.row16;
  uint32_t x = 0;
  for (int i = 0; i < p.row16; ++i) {
    const uint4 v = r[i];
    x ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  p.out[item] = x;
}

// B: one block owns one row group; threads interleave at 16 bytes, so each warp
// load instruction covers a contiguous 512-byte window of the row.
__global__ void read_block_interleaved(P p) {
  const int group = blockIdx.x;
  const uint4* r = p.w + static_cast<int64_t>(group) * p.row16;
  uint32_t x = 0;
  for (int i = threadIdx.x; i < p.row16; i += blockDim.x) {
    const uint4 v = r[i];
    x ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  p.out[group * blockDim.x + threadIdx.x] = x;
}

// C: one block owns one row group; each thread owns a contiguous chunk.
__global__ void read_block_chunks(P p) {
  const int group = blockIdx.x;
  const uint4* r = p.w + static_cast<int64_t>(group) * p.row16;
  const int chunk = p.row16 / blockDim.x;
  uint32_t x = 0;
  for (int i = threadIdx.x * chunk; i < (threadIdx.x + 1) * chunk; ++i) {
    const uint4 v = r[i];
    x ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  p.out[group * blockDim.x + threadIdx.x] = x;
}

// D: one block owns K row groups; threads interleave over the contiguous span.
__global__ void read_span_interleaved(P p) {
  const int span = p.threads / 32;  // row groups per block
  const int group = blockIdx.x * span;
  const uint4* r = p.w + static_cast<int64_t>(group) * p.row16;
  const int total = p.row16 * span;
  uint32_t x = 0;
  for (int i = threadIdx.x; i < total; i += blockDim.x) {
    const uint4 v = r[i];
    x ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  p.out[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

// E: as B, with four loads issued per iteration so more reads are in flight.
__global__ void read_block_multiload(P p) {
  const int group = blockIdx.x;
  const uint4* r = p.w + static_cast<int64_t>(group) * p.row16;
  uint32_t x = 0;
  for (int i = threadIdx.x * 4; i < p.row16; i += blockDim.x * 4) {
    const uint4 a = r[i], b = r[i + 1], c = r[i + 2], d = r[i + 3];
    x ^= a.x ^ b.y ^ c.z ^ d.w ^ a.w ^ b.z ^ c.y ^ d.x;
  }
  p.out[group * blockDim.x + threadIdx.x] = x;
}

template <class F>
static double time_it(cudaStream_t s, size_t bytes, F launch) {
  std::vector<double> gbps;
  for (int rep = 0; rep < 12; ++rep) {
    auto t0 = std::chrono::steady_clock::now();
    launch();
    cudaStreamSynchronize(s);
    auto t1 = std::chrono::steady_clock::now();
    gbps.push_back(bytes / std::chrono::duration<double>(t1 - t0).count() / 1e9);
  }
  std::sort(gbps.begin(), gbps.end());
  return gbps[gbps.size() / 2];
}

int main() {
  cudaStream_t s;
  cudaStreamCreate(&s);
  struct Shape {
    const char* name;
    int rows, cols;
  };
  const Shape shapes[] = {{"lm_head 128256x2048", 128256, 2048}, {"gate_proj 8192x2048", 8192, 2048}, {"down_proj 2048x8192", 2048, 8192}};
  uint32_t* out;
  cudaMalloc(&out, 1 << 26);
  printf("shape,pattern,threads_per_block,median_GBps\n");
  for (const Shape& sh : shapes) {
    const int groups = sh.rows / 4;
    const int row16 = sh.cols / 2;
    const size_t bytes = static_cast<size_t>(groups) * row16 * 16;
    std::vector<uint32_t> host(bytes / 4);
    uint32_t state = 0x12345;
    for (auto& v : host) {
      state ^= state << 13;
      state ^= state >> 17;
      state ^= state << 5;
      v = state;
    }
    uint4* w;
    cudaMalloc(&w, bytes);
    cudaMemcpy(w, host.data(), bytes, cudaMemcpyHostToDevice);
    P p{w, groups, row16, 32, out};
    printf("%s,A thread-per-row,32,%.1f\n", sh.name, time_it(s, bytes, [&] { read_thread_per_row<<<(groups + 31) / 32, 32, 0, s>>>(p); }));
    for (int t : {32, 64, 128, 256}) {
      if (t > row16) continue;
      printf("%s,B block-per-row interleaved,%d,%.1f\n", sh.name, t, time_it(s, bytes, [&] { read_block_interleaved<<<groups, t, 0, s>>>(p); }));
      if (row16 % t == 0)
        printf("%s,C block-per-row chunks,%d,%.1f\n", sh.name, t, time_it(s, bytes, [&] { read_block_chunks<<<groups, t, 0, s>>>(p); }));
    }
    for (int t : {64, 128, 256, 512, 1024}) {
      p.threads = t;
      const int span = t / 32;
      printf("%s,D span-of-%d-rows interleaved,%d,%.1f\n", sh.name, span, t, time_it(s, bytes, [&] { read_span_interleaved<<<(groups + span - 1) / span, t, 0, s>>>(p); }));
    }
    for (int t : {32, 64}) {
      if (t * 4 > row16) continue;
      printf("%s,E block-per-row multiload,%d,%.1f\n", sh.name, t, time_it(s, bytes, [&] { read_block_multiload<<<groups, t, 0, s>>>(p); }));
    }
    fflush(stdout);
    cudaFree(w);
  }
  return 0;
}
