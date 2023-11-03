cmake_minimum_required(VERSION 3.1)
project(optimize CXX CUDA)
#set(CMAKE_CUDA_STANDARD 14)
find_package(CUDA 9.0 REQUIRED)
message(STATUS ${CUDA_INCLUDE_DIRS})

add_subdirectory(common)
add_subdirectory(pybind11)
add_subdirectory(optEngine)

link_directories(${CMAKE_BINARY_DIR}/optEngine)
add_library(opti MODULE optimize.cpp)
target_include_directories(opti PUBLIC ${CUDA_INCLUDE_DIRS} /usr/local/cuda/samples/common/inc ${PROJECT_SOURCE_DIR} ${PROJECT_BINARY_DIR})
target_link_libraries(opti PUBLIC optEngine pybind11::module)

set_target_properties(opti PROPERTIES
        CXX_VISIBILITY_PRESET "hidden"
        INTERPROCEDURAL_OPTIMIZATION TRUE
        PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
        )