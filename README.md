# Demo GEMM execution in multiple platforms

## Introduction

This program executes the same GEMM function on a GPU and CPU, demonstrating
 SYCL's capability to execute the same code on different platforms.

It prints the first elements of the resulting matrix.

## Building and running

From the source root directory:

```
sh scripts/build.sh
bin/gemm_demo
```

## Building and running the SYCL version

Build and install oneMKL for CUDA
(https://oneapi-src.github.io/oneMKL/building_the_project.html#building-with-cmake)

From the root directory:

The script will take the path to the SYCL mkl root path from the *SYCL_MKL*
environmental variable or as an argument.

```
ssh scripts/build.sh [path to the sycl mkl root dir]
export LD_LIBRARY_PATH=<path to the sycl mkl lib>:$LD_LIBRARY_PATH
bin/gemm_usm_gpu_cpu
```
