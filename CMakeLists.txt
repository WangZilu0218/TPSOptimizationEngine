cmake_minimum_required(VERSION 3.1)
project(optimize C CXX CUDA)
set(CMAKE_CUDA_STANDARD 14)
find_package(CUDA 9.0 REQUIRED)
message(STATUS ${CUDA_INCLUDE_DIRS})

add_subdirectory(common)
add_subdirectory(pybind11)
add_subdirectory(optEngine)