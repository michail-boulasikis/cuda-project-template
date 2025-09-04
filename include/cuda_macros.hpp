#ifndef CUDA_MACROS_HPP
#define CUDA_MACROS_HPP

#ifdef __CUDACC__
#define CUDA_HOST __host__
#define CUDA_DEVICE __device__
#define CUDA_GLOBAL __global__
#else
#define CUDA_HOST
#define CUDA_DEVICE
#define CUDA_GLOBAL
#endif

#ifdef NDEBUG
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA Error: %s (%s:%d)\n",                         \
                   cudaGetErrorString(err), __FILE__, __LINE__);               \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t err = (call);                                               \
    if (err != CUBLAS_STATUS_SUCCESS) {                                        \
      std::fprintf(stderr, "cuBLAS Error %d (%s:%d)\n", err, __FILE__,         \
                   __LINE__);                                                  \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)
#else
#define CUDA_CHECK(call) (call)
#define CUBLAS_CHECK(call) (call)
#endif

#endif // CUDA_MACROS_HPP
