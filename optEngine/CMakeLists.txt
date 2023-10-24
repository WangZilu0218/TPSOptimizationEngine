
add_library(DoseLoss SHARED IMPORTED)
set_target_properties(DoseLoss PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/enzymeAutoGrad/doseLoss.so)

add_library(DVHLoss SHARED IMPORTED)
set_target_properties(DVHLoss PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/enzymeAutoGrad/DVHLoss.so)

add_library(EUDLoss SHARED IMPORTED)
set_target_properties(EUDLoss PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/enzymeAutoGrad/EUDLoss.so)

add_library(optEngine fista.cpp fista.cu)
target_link_libraries(optEngine DoseLoss DVHLoss EUDLoss)
target_include_directories(optEngine PUBLIC ${CUDA_INCLUDE_DIRS} /usr/local/cuda/samples/common/inc ${PROJECT_SOURCE_DIR} ${PROJECT_BINARY_DIR})
set_target_properties(optEngine PROPERTIES CUDA_ARCHITECTURES "70;72;75")