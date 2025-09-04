#ifndef CUDA_MACROS_CUH
#define CUDA_MACROS_CUH

// This hack here is done to avoid exposing CUDA code to the host compiler.
// When we include a header compiled with `nvcc` in a .cpp source file, we also include declarations of CUDA kernels.
// The host compiler does not know what CUDA kernels or calls are and refuses to compile.
// But we also need to include these files to be able to access GPU functions.
// This is why we do this:

#ifdef __CUDACC__           // If the compiler processing the including file is a CUDA compiler...
                            // ...then we define these macros
#define CUDA_HOST __host__
#define CUDA_DEVICE __device__
#define CUDA_GLOBAL __global__
#else                       // If, on the other hand, this is a host compiler...
                            // ... then there is no macros, and it is seen as a normal declaration!
#define CUDA_HOST
#define CUDA_DEVICE
#define CUDA_GLOBAL
#endif

// If we are compiling using DEBUG mode, we make checks, otherwise we do not need
// this extra overhead.
#ifdef NDEBUG
#include <cstdio>
#include <cstdlib>
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA Error: %s (%s:%d)\n",                         \
                   cudaGetErrorString(err), __FILE__, __LINE__);               \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)
#else
#define CUDA_CHECK(call) (call)
#endif

#endif // CUDA_MACROS_CUH
