# It's time to declare our next target, this time a GPU library named kernels.
# Much like the CPU library, it's going to be a STATIC library, so the same things apply here.
add_library(kernels STATIC)

# Much like the CPU library, we will build the GPU library using one source file. If you have more files, just add them:
# target_sources(kernels PRIVATE add_vec.cu dot_product.cu, ...)
target_sources(kernels PRIVATE add_vec.cu)

# Same as the CPU library.
target_include_directories(kernels PUBLIC ${PROJECT_SOURCE_DIR}/include)

# This line defines compilation options (think flags that are to be passed to the compiler) for our library. The options are PRIVATE, which means that only our library requires them.
# The library's consumers are free to compile in whatever way they like.
# The expression on the right uses something called a "generator expression" (*).
# Generator expressions are a CMake feature that allow you to make build settings conditional — they are evaluated *at build time*, not during the initial CMake configuration step.
#
# This particular generator expression:
#   $<$<COMPILE_LANGUAGE:CUDA>: ... >
# reads as:
#   "If the source file currently being compiled is a CUDA file, apply these options."
#
# So, the compiler flags inside this expression will only be passed to `nvcc`, not to the host C++ compiler.
# This lets us write platform- or language-specific configurations without duplicating CMake logic.
#
# Here, an interesting flag is --relocatable-device-code=true. This will allow us to link to this library using the host C++ compiler.
target_compile_options(kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --generate-line-info --use_fast_math --relocatable-device-code=true>)

# Same as the CPU library, except we require C++20, as nvcc does not yet support C++23.
target_compile_features(kernels PRIVATE cxx_std_20)

# This is a very useful CMake command which links other libraries to our target.
# Here for instance we specify that we want to use the CUDA::cudart cuda runtime library.
# We link it as a PRIVATE library, which means we do not want consumers of our library to also link to CUDA::cudart.
# We could also link to our CPU library here by just appending it to the list of libraries.
target_link_libraries(kernels PRIVATE CUDA::cudart)

# Same as the CPU library, except that we have specified CUDA_SEPARABLE_COMPILATION as ON.
# With separable compilation ON, CMake uses nvcc to compile GPU code to an intermediate
# format and later links the device code together. This makes separate compilation and
# linking of CUDA modules possible, just like with regular C++ object files.
# This is not so useful here as we only have one source file, but is good to know in general when developing CUDA codes.
set_target_properties(kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)


# (*) Tip: Generator expressions can also be used for other conditions, like:
#   - $<CONFIG:Debug>          only in Debug builds
#   - $<BOOL:${SOME_VAR}>      if SOME_VAR is ON
#   - $<TARGET_EXISTS:foo>     if target "foo" exists
#
# They can be nested and combined to make CMake more dynamic and portable.
