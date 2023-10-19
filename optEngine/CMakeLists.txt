
add_library(DoseLoss SHARED IMPORTED)
set_target_properties(DoseLoss PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/enzymeAutoGrad/doseLoss.so)


add_library(optEngine fista.cpp fista.cu)
target_link_libraries(optEngine DoseLoss)
target_include_directories(optEngine PUBLIC ${CUDA_INCLUDE_DIRS} /usr/local/cuda/samples/common/inc ${PROJECT_SOURCE_DIR} ${PROJECT_BINARY_DIR})
set_target_properties(optEngine PROPERTIES CUDA_ARCHITECTURES "35;50;70;72")