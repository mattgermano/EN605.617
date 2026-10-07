//!
//! @file assignment.cu
//! @author Matt Germano (mgerman8@jh.edu)
//! @brief
//! @version 0.1
//! @date 2026-10-06
//!
//! @copyright Copyright (c) 2026
//!

#include <cstdint>
#include <cstdlib>
#include <cuda/cmath>
#include <cuda_runtime.h>
#include <exception>
#include <format>
#include <iostream>
#include <source_location>
#include <string>

//!
//! @brief Helper function for checking for CUDA errors
//!
//! @param[in] result The result of a CUDA API call
//! @param[in] loc    The source code location of the function call
//!
inline void
checkCudaErrors(cudaError_t result,
                std::source_location loc = std::source_location::current()) {
  if (result != cudaSuccess) {
    std::cerr << std::format(
        "CUDA Runtime Error: {}:{}:{} = {}\n", loc.file_name(), loc.line(),
        static_cast<int>(result), cudaGetErrorString(result));
    cudaDeviceReset();
    exit(EXIT_FAILURE);
  }
}

//!
//! @brief Converts a sequence of characters to a number
//!
//! @param[in]  arg   The sequence of characters
//! @param[out] value The characters converted to a number
//! @return true if the characters could be converted to a number, else false
//!
bool parse_number(const char *arg, long long &value) {
  try {
    value = std::stoll(arg);
    return true;
  } catch (const std::exception &) {
    std::cerr << std::format("Failed to convert '{}' to a number\n", arg);
    return false;
  }
}

//!
//! @brief Validates and sets the kernel grid/block dimensions
//!
//! @param[out] total_threads The total number of threads
//! @param[out] block_size    The number of threads per block
//! @param[out] num_blocks    The number of blocks in the grid
//! @return true if the grid/block dimensions are valid, else false
//!
bool set_block_dimensions(long long &total_threads, long long block_size,
                          long long &num_blocks) {
  int device_id = 0;
  cudaDeviceProp prop{};
  checkCudaErrors(cudaGetDeviceProperties(&prop, device_id));

  if (block_size <= 0 || block_size > prop.maxThreadsPerBlock) {
    std::cerr << std::format("Block size must be between 1 and {}\n",
                             prop.maxThreadsPerBlock);
    return false;
  }

  if (block_size % prop.warpSize != 0) {
    std::cout << std::format(
        "Warning: Block size is not a multiple of the warp size ({})\n",
        prop.warpSize);
  }

  num_blocks = cuda::ceil_div(total_threads, block_size);
  if (num_blocks > prop.maxGridSize[0]) {
    std::cerr << std::format(
        "Number of blocks ({}) is over the maximum of {}\n", num_blocks,
        prop.maxGridSize[0]);
    return false;
  }

  if (total_threads != num_blocks * block_size) {
    total_threads = num_blocks * block_size;
    std::cout << "Warning: Total thread count is not evenly divisible by the "
                 "block size\n";
    std::cout << std::format(
        "The total number of threads will be rounded up to {}\n",
        total_threads);
  }

  return true;
}

//!
//! @brief Parses CLI arguments
//!
//! @param[in]  argc          The number of arguments
//! @param[in]  argv          The input arguments
//! @param[out] total_threads The total number of threads
//! @param[out] block_size    The number of threads per block
//! @param[out] num_blocks    The number of blocks in the grid
//! @return true if the arguments could be parsed successfully, else false
//!
bool parse_arguments(int argc, char **argv, long long &total_threads,
                     long long &block_size, long long &num_blocks) {
  if (argc > 3) {
    std::cerr << "Too many arguments\n";
    return false;
  }
  if (argc >= 2 && !parse_number(argv[1], total_threads)) {
    return false;
  }
  if (argc >= 3 && !parse_number(argv[2], block_size)) {
    return false;
  }
  if (total_threads <= 0) {
    std::cerr << "Total threads must be greater than 0\n";
    return false;
  }

  return set_block_dimensions(total_threads, block_size, num_blocks);
}

//!
//! @brief Initializes input/output buffers and executes the histogram kernel
//!
//! @param[in] block_size The number of threads per block
//! @param[in] num_blocks The number of blocks in the grid
//! @return true if the kernel executed successfully, else false
//!
bool execute_gpu_functions(long block_size, long long num_blocks) {
  return true;
}

int main(int argc, char **argv) {
  long long total_threads = (1LL << 22);
  long long block_size = 256;
  long long num_blocks = 0;

  if (!parse_arguments(argc, argv, total_threads, block_size, num_blocks)) {
    std::cerr << std::format("Usage: {} [total_threads] [block_size]\n",
                             argv[0]);
    return EXIT_FAILURE;
  }

  std::cout << std::format("[+] Total threads: {}\n", total_threads);
  std::cout << std::format("[+] Total blocks: {}\n", num_blocks);
  std::cout << std::format("[+] Threads per block: {}\n", block_size);
  std::cout << "[+] NOTE: Kernel execution time does not measure CPU/GPU "
               "memory transfers\n\n";

  if (!execute_gpu_functions(block_size, num_blocks)) {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
