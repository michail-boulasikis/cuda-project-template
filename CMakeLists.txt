# This is the first line in any top-level CMakeLists.txt - if the version of CMake that we use to generate the build files is lower than the given version,
# an error will be reported. This makes sure that the used CMake used is capable enough to understand the commands we have used.
cmake_minimum_required(VERSION 4.0.0)

# Then follows the project declaration. Here we also specify the version as well as the languages used in this project.
project(cuda-project-template VERSION 0.1.0 LANGUAGES CXX CUDA)

# We require that the build environment provides the CUDA toolkit. We need this because we link to CUDA libraries in our executable and GPU libraries.
find_package(CUDAToolkit REQUIRED)

# Finally, we include the source code subdirectories.
# Personally I like to split the library source code and the executable source code in two different directories.
# This is not necessary, but it might help organize things if your project gets too big.
# Note that this means that these folders should also have a CMakeLists.txt in them - failing to provide it will result in an error.
# This is necessary even if the only thing these CMakeLists.txt do is add subdirectories beneath them.
add_subdirectory(tools)
add_subdirectory(lib)
