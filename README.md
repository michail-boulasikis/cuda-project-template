# Modern CUDA Project Template

This is a CUDA project template that you can clone and get started writing CUDA. It has 3 goals:
- Be as understandable, pedagogical and simple as possible.
- Make use of modern CMake capabilities.
- Invoke the CUDA compiler (for example `nvcc`), as little as possible. Here, the CUDA compiler is only invoked to compile the code found under `./lib/kernels/add_vec.cu`.

## Requirements

I make no guarantee that the project builds in any environment - its purpose is more pedagogical. Nevertheless, in order to be able to build this project, you need the CUDA toolkit, CMake 4.0.0 or above and a C++ compiler capable of compiling C++23.

The project can then be built by running the following commands in the top level directory:

```
mkdir target && cd target
cmake ..
make
```


## Guide

In order to showcase the most common CMake and CUDA patterns, this template has three targets: two libraries and one executable.
- Library `hostutils` is a pure CPU library.
- Library `kernels` is a GPU library that defines a kernel and a CPU interface function for that kernel.
- Executable `vec-adder` inside the `tools` directory links against these two libraries and makes use of them.

The CMakeLists.txt in this project have comments on every line giving thorough explanations of what is going on.
I have written them with the expectation that you will start by looking into the [top level CMakeLists.txt](CMakeLists.txt). Then I recommend looking into the [CPU library CMakeLists.txt](lib/hostutils/CMakeLists.txt), then the [GPU library CMakeLists.txt](lib/kernels/CMakeLists.txt) and finally [the executable CMakeLists.txt](tools/vec-adder/CMakeLists.txt). The CMakeLists.txt which are under the `lib/` and `tools/` directories just include the subdirectories below them.
The `include/` folder does not have a CMakeLists.txt; this is normal, as these are simply header files which are included as text in the source files.

I hope you learn something through this template!
