#include <cstdio>
__global__ void touch(float* x, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] = fmaf(x[i], 1.5f, 2.0f); }
int main() { const int n = 1 << 20; float* x; cudaMalloc(&x, n * 4); cudaMemset(x, 0, n * 4); for (int r = 0; r < 4; ++r) touch<<<(n + 255) / 256, 256>>>(x, n); cudaDeviceSynchronize(); printf("ok\n"); }
