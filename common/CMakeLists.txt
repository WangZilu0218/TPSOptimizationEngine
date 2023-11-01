add_library(csc STATIC csc.cpp)
target_include_directories(csc PUBLIC ${CUDA_INCLUDE_DIRS} /usr/local/cuda/samples/common/inc ${PROJECT_SOURCE_DIR} ${PROJECT_BINARY_DIR})
set_target_properties(csc PROPERTIES CUDA_ARCHITECTURES "70;72;75")
target_link_libraries(csc PRIVATE cusparse)