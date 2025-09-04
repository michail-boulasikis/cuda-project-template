
# Welcome to the CPU library CMakeLists.txt.
# It is here we specify our first target. A target is CMake jargon for something that will be built.
# In this case, we build a library which is going to be named `hostutils`.
# It will be a static library, which means that it will end up in a binary file named `libhostutils.a`, and then linked into an executable that wants to use it at build time.
# Unlike shared libraries, it won't be dynamically loaded at runtime.
add_library(hostutils STATIC)

# This library will be built using just a single source file, `utils.cpp`.
# In CMake, the words PRIVATE, INTERFACE and PUBLIC specify visibility and propagation of properties.
# Here, PRIVATE means that only hostutils needs utils.cpp - this source file is not going to be needed by whatever target uses hostutils.
target_sources(hostutils PRIVATE utils.cpp)

# Here we specify the C++ standard. Our library uses C++23 features, so we require CMake to enable support for them.
# Since this requirement is PRIVATE, consumers of our library can use whatever C++ standard they want.
# If we used PUBLIC instead, our consumers would also be required to use at least C++23.
target_compile_features(hostutils PRIVATE cxx_std_23)

# This is going to be added to our include path. PROJECT_SOURCE_DIR is the root directory of our project.
# For example, if we have a file include/hostutils/utils.hpp, we can include it in utils.cpp as:
# #include <hostutils/utils.hpp>
# We also want consumers of our library to be able to include these headers, so we make them PUBLIC (*).
# If this were a header-only library, we might use INTERFACE here instead of PUBLIC,
# since the library would have no compiled sources and only expose headers.
target_include_directories(hostutils PUBLIC ${PROJECT_SOURCE_DIR}/include)

# Finally, we specify where the target is going to end up.
# Personally I like putting all the static and shared library targets into a lib/ directory.
# This is not necessary of course, it's up to you how you want to organize your project.
set_target_properties(hostutils PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# (*) Note that this will make all headers under include/ accessible to consumers, not just the ones under include/hostutils/
