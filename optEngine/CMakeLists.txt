
add_library(DoseLoss STATIC IMPORTED)
set_target_properties(DoseLoss PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/enzymeAutoGrad/DoseLoss.so)

add_library(DVHLoss STATIC IMPORTED)
set_target_properties(DVHLoss PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/enzymeAutoGrad/DVHLoss.so)

add_library(EUDLoss STATIC IMPORTED)
set_target_properties(EUDLoss PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/enzymeAutoGrad/EUDLoss.so)

add_library(UniformLoss STATIC IMPORTED)
set_target_properties(UniformLoss PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/enzymeAutoGrad/UniformDoseLoss.so)

link_directories(${CMAKE_BINARY_DIR}/common)
add_library(optEngine STATIC fista.cpp fista.cu)
set_property(TARGET optEngine PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(optEngine PRIVATE DoseLoss DVHLoss EUDLoss cublas UniformLoss csc)
target_include_directories(optEngine PUBLIC ${CUDA_INCLUDE_DIRS} /usr/local/cuda/samples/common/inc ${PROJECT_SOURCE_DIR} ${PROJECT_BINARY_DIR})
set_target_properties(optEngine PROPERTIES CUDA_ARCHITECTURES "70;72;75")

