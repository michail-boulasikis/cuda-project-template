# This is the final target in our project, this time an executable.
# The executable's name is going to be vec-adder.
add_executable(vec-adder)

# Same as the CPU and GPU library targets.
target_sources(vec-adder PRIVATE main.cpp)

# Remember that we are linking to libraries that PUBLICly expose headers in "include/".
# So we do not need this line, even though we are including files from that directory.
#target_include_directories(vec-adder PRIVATE ${PROJECT_SOURCE_DIR}/include)

# Same as the GPU library target, but here we link to the two other libraries we have already built: hostutils and kernels!
target_link_libraries(vec-adder PRIVATE hostutils kernels)

# Same as the CPU and GPU library targets.
target_compile_features(vec-adder PRIVATE cxx_std_23)

# Similar to the CPU and GPU library targets, except this time we are building an executable, so we put it in <target-dir>/bin.
# We can then find our executable at <target-dir>/bin/vec-adder
set_target_properties(vec-adder PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
