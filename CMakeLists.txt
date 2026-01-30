cmake_minimum_required(VERSION 3.20)
project(ams_point_interpolant)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Download pybind11 and external libraries
include(FetchContent)

# Fetch pybind11
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
)

# Fetch PoissonRecon
FetchContent_Declare(
    poisson_recon
    GIT_REPOSITORY https://github.com/mkazhdan/PoissonRecon.git
    GIT_TAG cd6dc7d33f028b2e6496f5cd999c25cecd56aff2
)

# Install dependencies of PoissonRecon
find_package(Threads REQUIRED)
find_package(OpenMP)

# Compile options (from original PoissonRecon's makefile)
set(COMMON_WARNINGS
    -Wno-deprecated
    -Wno-invalid-offsetof
)

set(RELEASE_FLAGS
    -O3
    -DRELEASE
    -funroll-loops
    -ffast-math
    -g
)

set(DEBUG_FLAGS
    -DDEBUG
    -g3
)

# OpenMP handling
if(OpenMP_CXX_FOUND)
    set(OPENMP_LIB OpenMP::OpenMP_CXX)
endif()

# Make all libraries available
FetchContent_MakeAvailable(pybind11 poisson_recon)

# Create the Python module
pybind11_add_module(ams_point_interpolant cpp/ams_point_interpolant.cpp)

# Include directories of PoissonRecon
target_include_directories(ams_point_interpolant PRIVATE 
    ${poisson_recon_SOURCE_DIR}/Src
)

# Set output directory
set_target_properties(ams_point_interpolant PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/py_ams_point_interpolant
)
