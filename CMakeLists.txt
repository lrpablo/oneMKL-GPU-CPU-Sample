# Copyright (C) 2024 Codeplay Software Limited

cmake_minimum_required(VERSION 3.16)

project(geem-demo LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

if(NOT DEFINED SYCL_MKLROOT)
  message(FATAL_ERROR "SYCL_MKLROOT not defined")
endif()

if(NOT DEFINED ENV{ONEAPI_ROOT})
  message(
    FATAL_ERROR
      "ONEAPI_ROOT not defined - oneAPI not instaled or setvars.sh not executed"
  )
endif()

set(BINARY_NAME
    "gemm-demo"
    CACHE STRING "Binary name")

add_subdirectory(src)

set_target_properties(${BINARY_NAME} PROPERTIES CUDA_ARCHITECTURES "50;72;89")
