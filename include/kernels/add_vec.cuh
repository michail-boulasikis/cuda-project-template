#ifndef KERNELS_ADD_VEC_CUH
#define KERNELS_ADD_VEC_CUH
#include <cstddef>
#include <cuda_macros.hpp>

namespace gpu {

namespace detail {
CUDA_GLOBAL void add_vec(float *__restrict__ res, float *__restrict__ const a,
                         float *__restrict__ const b, size_t n);
}

void add_vec(float *r, const float *a, const float *b, size_t n);

} // namespace gpu
#endif // KERNELS_ADD_VEC_CUH
