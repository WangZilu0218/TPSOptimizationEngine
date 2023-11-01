
add_library(DoseLoss SHARED IMPORTED)
set_target_properties(DoseLoss PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/enzymeAutoGrad/doseLoss.so)

add_library(DVHLoss SHARED IMPORTED)
set_target_properties(DVHLoss PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/enzymeAutoGrad/DVHLoss.so)

add_library(EUDLoss SHARED IMPORTED)
set_target_properties(EUDLoss PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/enzymeAutoGrad/EUDLoss.so)

add_library(UniformLoss SHARED IMPORTED)
set_target_properties(UniformLoss PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/enzymeAutoGrad/UniformLoss.so)

link_directories(${CMAKE_BINARY_DIR}/common)
add_library(optEngine fista.cpp fista.cu)
target_link_libraries(optEngine PRIVATE DoseLoss DVHLoss EUDLoss cublas UniformLoss csc)
target_include_directories(optEngine PUBLIC ${CUDA_INCLUDE_DIRS} /usr/local/cuda/samples/common/inc ${PROJECT_SOURCE_DIR} ${PROJECT_BINARY_DIR})
set_target_properties(optEngine PROPERTIES CUDA_ARCHITECTURES "70;72;75")

add_library(opti optimize.cpp)
target_include_directories(opti PUBLIC ${CUDA_INCLUDE_DIRS} /usr/local/cuda/samples/common/inc ${PROJECT_SOURCE_DIR} ${PROJECT_BINARY_DIR})
target_link_libraries(opti PUBLIC optEngine pybind11::module)