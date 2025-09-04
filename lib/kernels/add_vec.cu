#include <kernels/add_vec.cuh>

namespace gpu {

namespace detail {
__global__ void add_vec(float *__restrict__ res, float *__restrict__ const a,
                        float *__restrict__ const b, size_t n) {
  size_t i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n) {
    res[i] = a[i] + b[i];
  }
}
} // namespace detail

void add_vec(float *r, const float *a, const float *b, size_t n) {
  float *d_r, *d_a, *d_b;
  cudaMalloc(&d_r, n * sizeof(float));
  cudaMalloc(&d_a, n * sizeof(float));
  cudaMalloc(&d_b, n * sizeof(float));
  cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);
  const size_t block_size = 256;
  const size_t grid_size = (block_size + n - 1) / block_size;
  detail::add_vec<<<grid_size, block_size>>>(d_r, d_a, d_b, n);
  cudaMemcpy(r, d_r, n * sizeof(float), cudaMemcpyDeviceToHost);
  return;
}
} // namespace gpu
