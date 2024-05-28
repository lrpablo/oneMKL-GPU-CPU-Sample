
/*
 *
 *       This example demonstrates use of DPCPP API oneapi::mkl::blas::gemm
 *       using unified shared memory to perform General Matrix-Matrix
 *       Multiplication on two SYCL devices (CPU and GPU) if available, or on
 *       the available ones if any.
 *
 */

#include "oneapi/mkl.hpp"
#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

using mklTrans = oneapi::mkl::transpose;

#include "support.hpp"

/**
 * This code snippet demonstrates how to perform a matrix-matrix multiplication
 * (GEMM) using the oneAPI Math Kernel Library (oneMKL) and SYCL. The `run_gemm`
 * function takes in the device, input matrices A and B, output matrix C, and
 * other parameters required for the GEMM operation. It first creates an
 * execution queue and a context using the provided device. Then, it allocates
 * memory on the device using `sycl::malloc_device`. If the memory allocation
 * fails, it throws a `std::runtime_error`. Next, it copies the input matrices A
 * and B from the host to the device using `q.memcpy`. After that, it executes
 * the GEMM operation using `oneapi::mkl::blas::column_major::gemm`. It waits
 * for the calculations to complete using `q.wait_and_throw`. Then, it copies
 * the output matrix C from the device to the host using `q.memcpy`. Finally, it
 * prints the values of matrices A, B, and C using the `print_values` function
 * and frees the allocated device memory using `sycl::free`.
 *
 * The `main` function initializes the input matrices A, B, and C, and sets the
 * parameters for the GEMM operation. It first tries to run the GEMM operation
 * on a GPU device and then tries to run it on a CPU device. It catches any
 * synchronous SYCL exceptions and prints the error message and error code. If
 * any other exception occurs, it prints the exception message.
 *
 * @param dev The SYCL device to use for the GEMM operation.
 * @param A The input matrix A.
 * @param B The input matrix B.
 * @param C The output matrix C. The matrix passed by value.
 * @param tA The transpose operation to apply to matrix A.
 * @param tB The transpose operation to apply to matrix B.
 * @param m The number of rows in matrix A and matrix C.
 * @param n The number of columns in matrix B and matrix C.
 * @param k The number of columns in matrix A and rows in matrix B.
 * @param ldA The leading dimension of matrix A.
 * @param ldB The leading dimension of matrix B.
 * @param ldC The leading dimension of matrix C.
 * @param alpha The scalar alpha.
 * @param beta The scalar beta.
 */
void run_gemm(const sycl::device &dev, std::vector<float> &A,
              std::vector<float> &B, std::vector<float> C, mklTrans tA,
              mklTrans tB, std::int64_t m, std::int64_t n, std::int64_t k,
              std::int64_t ldA, std::int64_t ldB, std::int64_t ldC, float alpha,
              float beta) {

  // Check if input vectors are empty
  if (A.empty() || B.empty() || C.empty()) {
    throw std::runtime_error("Input vectors A, B, or C are empty.");
  }

  // Catch asynchronous exceptions
  auto exception_handler = [](sycl::exception_list exceptions) {
    for (std::exception_ptr const &e : exceptions) {
      try {
        std::rethrow_exception(e);
      } catch (sycl::exception const &e) {
        std::cerr << "Caught asynchronous SYCL exception during GEMM:"
                  << std::endl;
        std::cerr << "\t" << e.what() << std::endl;
      }
    }
    throw std::runtime_error("Failed to execute GEMM.");
  };

  // create execution queue
  sycl::queue q(dev, exception_handler);
  sycl::event gemm_done;
  sycl::context cxt = q.get_context();

  // allocate memory on device
  auto dev_A = sycl::malloc_device<float>(A.size(), q);
  auto dev_B = sycl::malloc_device<float>(B.size(), q);
  auto dev_C = sycl::malloc_device<float>(C.size(), q);

  // copy data from host to device
  q.memcpy(dev_A, A.data(), A.size() * sizeof(float)).wait();
  q.memcpy(dev_B, B.data(), B.size() * sizeof(float)).wait();
  q.memcpy(dev_C, C.data(), C.size() * sizeof(float)).wait();

  //
  // Execute Gemm
  gemm_done = oneapi::mkl::blas::column_major::gemm(
      q, tA, tB, m, n, k, alpha, dev_A, ldA, dev_B, ldB, beta, dev_C, ldC);

  // Wait until calculations are done
  q.wait_and_throw();

  // copy data from device back to host
  q.memcpy(C.data(), dev_C, C.size() * sizeof(float)).wait();

  print_values(C.data(), ldC, "C");

  sycl::free(dev_C, q);
  sycl::free(dev_B, q);
  sycl::free(dev_A, q);
}
/**
 * This program  demonstrates the use of the oneAPI Math Kernel Library
 * (oneMKL) and SYCL running in multiple devices.
 *
 * It initializes input matrices A, B, and C, and sets the parameters for the
 * GEMM operation.
 *
 * It first tries to run the GEMM operation on a GPU device and then tries to
 * run it on a CPU device.
 *
 * If a GPU device is available, it executes the GEMM operation on the GPU
 * device using the run_gemm function.
 *
 * If a CPU device is available, it executes the GEMM operation on the CPU
 * device using the run_gemm function.
 *
 * If any synchronous SYCL exception occurs, it prints the error message and
 * error code.
 *
 * @return 0 if the program executed successfully, 1 otherwise.
 */
int main(void) {
  try {
    std::vector<float> A;
    std::vector<float> B;
    std::vector<float> C;
    sycl::device dev;

    auto transA = mklTrans::trans;
    auto transB = mklTrans::nontrans;
    auto transC = mklTrans::nontrans;

    // matrix data sizes
    std::int64_t m = 60;
    std::int64_t n = 100;
    std::int64_t k = 70;

    // leading dimensions of data
    std::int64_t ldA = 105;
    std::int64_t ldB = 110;
    std::int64_t ldC = 115;

    size_t sizeA =
        ((transA == mklTrans::nontrans) ? ldA * k : ldA * m) * sizeof(float);
    size_t sizeB =
        ((transB == mklTrans::nontrans) ? ldB * n : ldB * k) * sizeof(float);
    size_t sizeC = (ldC * n) * sizeof(float);

    init_data(A, transA, sizeA, m, k, ldA);
    init_data(B, transB, sizeB, k, n, ldB);
    init_data(C, transC, sizeC, m, n, ldC);

    float alpha = 2.0;
    float beta = 3.0;

    std::cout << "Matrixes\n";
    print_values(A.data(), ldA, "A");
    print_values(B.data(), ldB, "B");
    print_values(C.data(), ldC, "C");
    std::cout << std::endl;

    /**
     * Retrieves the available GPU devices using SYCL and checks if there is at
     * least one GPU device available.
     */
    auto gpuDevices = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (!gpuDevices.empty()) {
      for (int i = 0; i < gpuDevices.size(); ++i) {
        dev = gpuDevices[i];
        std::cerr << "Running on GPU device." << std::endl;
        std::cerr << "Device name is: "
                  << dev.get_info<sycl::info::device::name>() << std::endl;

        run_gemm(dev, A, B, C, transA, transB, m, n, k, ldA, ldB, ldC, alpha,
                 beta);

        std::cerr << "GPU example successfully executed.\n" << std::endl;
      }
    } else {
      std::cerr << "No GPU detected.\n\n" << std::endl;
    }

    auto cpuDevices = sycl::device::get_devices(sycl::info::device_type::cpu);
    if (!cpuDevices.empty()) {
      dev = cpuDevices[0];
      std::cerr << "Running on CPU device." << std::endl;
      std::cerr << "Device name is: "
                << dev.get_info<sycl::info::device::name>() << std::endl;

      run_gemm(dev, A, B, C, transA, transB, m, n, k, ldA, ldB, ldC, alpha,
               beta);

      std::cerr << "CPU example successfully executed." << std::endl;
    } else {
      std::cout
          << "No CPU detected, check setting of ONEAPI_DEVICE_SELECTOR.\n\n"
          << std::endl;
    }
  } catch (sycl::exception const &e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  return 0;
}
