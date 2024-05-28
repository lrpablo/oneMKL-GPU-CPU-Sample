# Copyright (C) 2024 Codeplay Software Limited

#find_package(CUDA REQUIRED)

set(CMAKE_CXX_COMPILER icpx)
set(CMAKE_CXX_STANDARD 17)
set(INCLUDES include ${SYCL_MKLROOT}/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -L${SYCL_MKLROOT}/lib")

add_custom_target(release DEPENDS ${BINARY_NAME})
add_executable(${BINARY_NAME} gemm_usm_gpu_cpu.cpp)
target_include_directories(
  ${BINARY_NAME} PRIVATE ${SYCL_INCLUDE_DIR} ${INCLUDES} ${CUDA_INCLUDE_DIRS})
set_target_properties(${BINARY_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                ${CMAKE_SOURCE_DIR}/bin)
target_link_libraries(${BINARY_NAME} PRIVATE ${CUDA_LIBRARIES} onemkl)
