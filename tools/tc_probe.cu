// Tensor-core probe. Two questions decide the batched kernel design:
//  1. Is an mma result for one (row, lane) independent of where that row and lane
//     sit in the 16x16 tile? If so, tile kernels can be batch-invariant.
//  2. What arithmetic does the hardware perform? If a CPU model reproduces it
//     bit for bit, CPU/GPU equality survives on the tensor-core path too.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

using namespace nvcuda;

using bf16 = __nv_bfloat16;

// C' = A * B + C for `count` independent 16x16x16 tiles.
__global__ void mma_tiles(const bf16* A, const bf16* B, const float* C, float* out) {
  const int t = blockIdx.x;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> b;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
  wmma::load_matrix_sync(a, A + t * 256, 16);
  wmma::load_matrix_sync(b, B + t * 256, 16);
  wmma::load_matrix_sync(c, C + t * 256, 16, wmma::mem_row_major);
  wmma::mma_sync(c, a, b, c);
  wmma::store_matrix_sync(out + t * 256, c, 16, wmma::mem_row_major);
}

static uint16_t f32_to_bf16_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  // Round to nearest, ties to even.
  uint32_t r = u + 0x7FFF + ((u >> 16) & 1);
  return static_cast<uint16_t>(r >> 16);
}
static float bf16_bits_to_f32(uint16_t b) {
  uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// GEMM over a packed matrix: W is [row tile][col tile][16 rows][16 cols], x is
// [lane tile][col tile][16 lanes][16 cols]. LT lane tiles share one weight load.
template <int LT>
__global__ void tile_gemm(const bf16* W, const bf16* X, float* Y, int row_tiles, int col_tiles, int lane_tiles, int inflight) {
  const int part_count = (lane_tiles + LT - 1) / LT;
  const int rt = blockIdx.x / part_count;
  const int lt0 = (blockIdx.x % part_count) * LT;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[LT];
  for (int l = 0; l < LT; ++l) wmma::fill_fragment(c[l], 0.0f);
  const bf16* w = W + static_cast<int64_t>(rt) * col_tiles * 256;
  for (int ct = 0; ct < col_tiles; ++ct) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a;
    wmma::load_matrix_sync(a, w + static_cast<int64_t>(ct) * 256, 16);
    for (int l = 0; l < LT; ++l) {
      const int lt = lt0 + l;
      if (lt >= lane_tiles) break;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> b;
      wmma::load_matrix_sync(b, X + (static_cast<int64_t>(lt) * col_tiles + ct) * 256, 16);
      wmma::mma_sync(c[l], a, b, c[l]);
    }
  }
  for (int l = 0; l < LT; ++l) {
    const int lt = lt0 + l;
    if (lt >= lane_tiles) break;
    wmma::store_matrix_sync(Y + (static_cast<int64_t>(rt) * lane_tiles + lt) * 256, c[l], 16, wmma::mem_row_major);
  }
}

int main() {
  const int tiles = 4096;
  std::mt19937 rng(42);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<uint16_t> Ah(tiles * 256), Bh(tiles * 256);
  std::vector<float> Ch(tiles * 256);
  for (auto& v : Ah) v = f32_to_bf16_bits(nd(rng) * 0.05f);
  for (auto& v : Bh) v = f32_to_bf16_bits(nd(rng));
  for (auto& v : Ch) v = nd(rng);

  bf16 *A, *B;
  float *C, *O;
  cudaMalloc(&A, tiles * 512);
  cudaMalloc(&B, tiles * 512);
  cudaMalloc(&C, tiles * 1024);
  cudaMalloc(&O, tiles * 1024);
  std::vector<float> base(tiles * 256), perm(tiles * 256);
  const auto run = [&](const std::vector<uint16_t>& a, const std::vector<uint16_t>& b, const std::vector<float>& c, std::vector<float>& o) {
    cudaMemcpy(A, a.data(), tiles * 512, cudaMemcpyHostToDevice);
    cudaMemcpy(B, b.data(), tiles * 512, cudaMemcpyHostToDevice);
    cudaMemcpy(C, c.data(), tiles * 1024, cudaMemcpyHostToDevice);
    mma_tiles<<<tiles, 32>>>(A, B, C, O);
    cudaDeviceSynchronize();
    cudaMemcpy(o.data(), O, tiles * 1024, cudaMemcpyDeviceToHost);
  };
  run(Ah, Bh, Ch, base);

  // Question 1: permute lanes (B columns and C columns) and rows (A rows and C rows)
  // independently per tile, and compare each element against its unpermuted result.
  std::vector<uint16_t> Ap(Ah.size()), Bp(Bh.size());
  std::vector<float> Cp(Ch.size());
  std::vector<int> rperm(tiles * 16), lperm(tiles * 16);
  for (int t = 0; t < tiles; ++t) {
    std::vector<int> r(16), l(16);
    std::iota(r.begin(), r.end(), 0);
    std::iota(l.begin(), l.end(), 0);
    std::shuffle(r.begin(), r.end(), rng);
    std::shuffle(l.begin(), l.end(), rng);
    for (int i = 0; i < 16; ++i) {
      rperm[t * 16 + i] = r[i];
      lperm[t * 16 + i] = l[i];
    }
    for (int i = 0; i < 16; ++i)
      for (int k = 0; k < 16; ++k) Ap[t * 256 + i * 16 + k] = Ah[t * 256 + r[i] * 16 + k];
    // B is col_major: element (k, j) sits at j*16 + k.
    for (int j = 0; j < 16; ++j)
      for (int k = 0; k < 16; ++k) Bp[t * 256 + j * 16 + k] = Bh[t * 256 + l[j] * 16 + k];
    for (int i = 0; i < 16; ++i)
      for (int j = 0; j < 16; ++j) Cp[t * 256 + i * 16 + j] = Ch[t * 256 + r[i] * 16 + l[j]];
  }
  run(Ap, Bp, Cp, perm);
  int64_t same = 0, total = 0;
  for (int t = 0; t < tiles; ++t)
    for (int i = 0; i < 16; ++i)
      for (int j = 0; j < 16; ++j) {
        const float a = perm[t * 256 + i * 16 + j];
        const float b = base[t * 256 + rperm[t * 16 + i] * 16 + lperm[t * 16 + j]];
        same += std::memcmp(&a, &b, 4) == 0;
        ++total;
      }
  printf("Q1 row/lane permutation: %lld / %lld elements bit-identical\n", (long long)same, (long long)total);

  // Question 2: arithmetic models against the hardware result.
  int64_t m_seq = 0, m_rev = 0, m_exact = 0, m_tree = 0, m_cfirst = 0;
  float worst_ulp = 0;
  for (int t = 0; t < tiles; ++t) {
    for (int i = 0; i < 16; ++i) {
      for (int j = 0; j < 16; ++j) {
        float p[16];
        for (int k = 0; k < 16; ++k)
          p[k] = bf16_bits_to_f32(Ah[t * 256 + i * 16 + k]) * bf16_bits_to_f32(Bh[t * 256 + j * 16 + k]);
        const float c = Ch[t * 256 + i * 16 + j];
        const float hw = base[t * 256 + i * 16 + j];
        float s = c;
        for (int k = 0; k < 16; ++k) s = s + p[k];
        m_seq += s == hw;
        s = c;
        for (int k = 15; k >= 0; --k) s = s + p[k];
        m_rev += s == hw;
        long double e = c;
        for (int k = 0; k < 16; ++k) e += static_cast<long double>(p[k]);
        m_exact += static_cast<float>(e) == hw;
        // Balanced adder tree over products, then the accumulator.
        float lv[16];
        std::copy(p, p + 16, lv);
        for (int n = 16; n > 1; n /= 2)
          for (int q = 0; q < n / 2; ++q) lv[q] = lv[2 * q] + lv[2 * q + 1];
        m_tree += (lv[0] + c) == hw;
        // Exact product sum rounded once, then added to the accumulator.
        long double se = 0;
        for (int k = 0; k < 16; ++k) se += static_cast<long double>(p[k]);
        m_cfirst += (static_cast<float>(se) + c) == hw;
        const float d = std::fabs(static_cast<float>(e) - hw);
        const float ulp = std::nextafter(std::fabs(hw), INFINITY) - std::fabs(hw);
        worst_ulp = std::max(worst_ulp, ulp > 0 ? d / ulp : 0.0f);
      }
    }
  }
  printf("Q2 models vs hardware, out of %lld:\n", (long long)total);
  printf("   sequential ascending %lld\n   sequential descending %lld\n   exact sum then round %lld\n   balanced tree + c %lld\n   exact products rounded, + c %lld\n   worst distance from exact: %.2f ulp\n",
         (long long)m_seq, (long long)m_rev, (long long)m_exact, (long long)m_tree, (long long)m_cfirst, worst_ulp);

  // Question 3: speed on the real shapes, for comparison with cuBLAS.
  cudaStream_t s;
  cudaStreamCreate(&s);
  printf("shape,batch,LT,us,T_MAC_per_s\n");
  struct Shape {
    const char* name;
    int rows, cols;
  };
  for (const Shape sh : {Shape{"lm_head", 128256, 2048}, Shape{"gate", 8192, 2048}, Shape{"down", 2048, 8192}, Shape{"q", 2048, 2048}, Shape{"kv", 512, 2048}}) {
    const int row_tiles = sh.rows / 16, col_tiles = sh.cols / 16;
    const size_t wbytes = static_cast<size_t>(row_tiles) * col_tiles * 512;
    bf16* W;
    cudaMalloc(&W, wbytes);
    {
      std::vector<uint16_t> h(wbytes / 2);
      for (auto& v : h) v = static_cast<uint16_t>(rng());
      for (auto& v : h) v = f32_to_bf16_bits((static_cast<int16_t>(v) / 32768.0f) * 0.05f);
      cudaMemcpy(W, h.data(), wbytes, cudaMemcpyHostToDevice);
    }
    for (int b : {16, 64, 128}) {
      const int lane_tiles = b / 16;
      const size_t xbytes = static_cast<size_t>(lane_tiles) * col_tiles * 512;
      const size_t ybytes = static_cast<size_t>(lane_tiles) * row_tiles * 1024;
      bf16* X;
      float* Y;
      cudaMalloc(&X, xbytes);
      cudaMalloc(&Y, ybytes);
      std::vector<uint16_t> h(xbytes / 2);
      for (auto& v : h) v = f32_to_bf16_bits(nd(rng));
      cudaMemcpy(X, h.data(), xbytes, cudaMemcpyHostToDevice);
      const auto bench = [&](int lt_per_block, auto launch) {
        std::vector<double> ts;
        for (int r = 0; r < 12; ++r) {
          auto t0 = std::chrono::steady_clock::now();
          launch();
          cudaStreamSynchronize(s);
          ts.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
        }
        std::sort(ts.begin(), ts.end());
        const double t = ts[ts.size() / 2];
        printf("%s,%d,%d,%.1f,%.2f\n", sh.name, b, lt_per_block, t * 1e6, static_cast<double>(sh.rows) * sh.cols * b / t / 1e12);
        fflush(stdout);
      };
      bench(1, [&] { tile_gemm<1><<<row_tiles * lane_tiles, 32, 0, s>>>(W, X, Y, row_tiles, col_tiles, lane_tiles, 1); });
      if (lane_tiles >= 2) bench(2, [&] { tile_gemm<2><<<row_tiles * ((lane_tiles + 1) / 2), 32, 0, s>>>(W, X, Y, row_tiles, col_tiles, lane_tiles, 1); });
      if (lane_tiles >= 4) bench(4, [&] { tile_gemm<4><<<row_tiles * ((lane_tiles + 3) / 4), 32, 0, s>>>(W, X, Y, row_tiles, col_tiles, lane_tiles, 1); });
      if (lane_tiles >= 8) bench(8, [&] { tile_gemm<8><<<row_tiles * ((lane_tiles + 7) / 8), 32, 0, s>>>(W, X, Y, row_tiles, col_tiles, lane_tiles, 1); });
      cudaFree(X);
      cudaFree(Y);
    }
    cudaFree(W);
  }
  return 0;
}
