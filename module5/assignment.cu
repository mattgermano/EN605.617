//!
//! @file assignment.cu
//! @author Matt Germano (mgerman8@jh.edu)
//! @brief Groups 8-bit pixel values from an image into exposure bins (blacks,
//! shadows, midtones, highlights, and whites) using all CUDA memory types
//! @version 0.1
//! @date 2026-09-27
//!
//! @copyright Copyright (c) 2026
//!

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cuda/cmath>
#include <cuda_runtime.h>
#include <exception>
#include <format>
#include <iostream>
#include <limits>
#include <source_location>
#include <string>
#include <vector>

// Image dimensions
static constexpr std::size_t NUM_ROWS = 32'000;
static constexpr std::size_t NUM_COLS = 16'000;
static constexpr std::size_t NUM_PIXELS = NUM_ROWS * NUM_COLS;
static_assert(
    NUM_PIXELS <= std::numeric_limits<std::uint32_t>::max(),
    "Exposure bin counter is restricted to 32-bits (4,294,967,295 pixels)");

// Number of possible values for an 8-bit pixel
static constexpr int NUM_VALUES = (1 << 8);

// The starting pixel value for each exposure bin. For example, if a pixel is
// between 0 and 17 it falls into the first exposure bin, if it's between 18 and
// 65 it falls into the next bin, etc.
static constexpr std::array<int, 5> BIN_STARTS = {0, 18, 66, 194, 242};
static constexpr int NUM_BINS = static_cast<int>(BIN_STARTS.size());
// The actual names corresponding to each of the bins. For example, a pixel
// value between 18 and 65 maps to the "Shadows" exposure bin.
static constexpr std::array<std::string, NUM_BINS> BIN_NAMES = {
    "Blacks", "Shadows", "Midtones", "Highlights", "Whites"};

// Constant lookup table to map a pixel value to its corresponding exposure bin.
// For example, c_bin_lut[193] = 3. This maps the pixel to the "Highlights"
// exposure bin (BIN_NAMES[3]).
// The memory is populated in execute_gpu_functions() using cudaMemcpyToSymbol()
__constant__ std::uint8_t c_bin_lut[NUM_VALUES]; // NOLINT

// Common function pointer type declaration shared between kernels
using HistogramKernel = void (*)(const std::uint8_t *, unsigned int *,
                                 std::size_t);

//!
//! @brief Helper function for checking for CUDA errors
//!
//! @param[in] result The result of a CUDA API call
//! @param[in] loc    The source code location of the function call
//!
inline void
checkCudaErrors(cudaError_t result,
                std::source_location loc = std::source_location::current()) {
  // Reference documentation:
  // https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#error-checking-in-cuda
  // Reference code:
  // https://github.com/NVIDIA/cuda-samples/blob/5443602d89ed99aede2e4b7bf329daddeadb320e/Common/helper_cuda.h#L585-L598
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
//! @brief Assembles the lookup table to map every possible pixel value to the
//! exposure bin that it falls in
//!
//! @return A 1-D array containing the exposure bin index for each pixel value
//!
std::array<std::uint8_t, NUM_VALUES> build_exposure_bin_lut() {
  std::array<std::uint8_t, NUM_VALUES> lut{};
  int bin = 0;
  for (int i = 0; i < NUM_VALUES; ++i) {
    while (bin + 1 < NUM_BINS && i >= BIN_STARTS[bin + 1]) {
      bin++;
    }
    lut[i] = static_cast<std::uint8_t>(bin);
  }
  return lut;
}

//!
//! @brief Determines the expected histogram given an array of pixel values and
//! a lookup table
//!
//! @param[in] h_image A 1-D array of 8-bit pixel values in the image
//! @param[in] lut     An exposure bin lookup table
//! @return A vector with the number of pixels in each bin
//!
std::vector<unsigned int>
compute_reference_hist(const std::uint8_t *h_image,
                       const std::array<std::uint8_t, NUM_VALUES> &lut) {
  std::vector<unsigned int> ref_hist(NUM_BINS, 0);
  for (std::size_t i = 0; i < NUM_PIXELS; ++i) {
    ref_hist[lut[h_image[i]]]++;
  }
  return ref_hist;
}

//!
//! @brief Fills an image with a test pattern
//!
//! @param[out] h_image A 1-D array of pixel values in the image
//!
//! @note The test pattern fills each row in the image with the current row
//! number (up to 256 before resetting back to 0). For example:
//! Row 0: 0 0 0 0 0 0 0 ... 0
//! Row 1: 1 1 1 1 1 1 1 ... 1
//! Row 2: 2 2 2 2 2 2 2 ... 2
//!
void fill_test_image(std::uint8_t *h_image) {
  for (std::size_t r = 0; r < NUM_ROWS; ++r) {
    for (std::size_t c = 0; c < NUM_COLS; ++c) {
      h_image[r * NUM_COLS + c] = static_cast<std::uint8_t>(r % NUM_VALUES);
    }
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
  if (static_cast<std::size_t>(total_threads) > NUM_PIXELS) {
    std::cout << "Warning: More threads than pixels\n";
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
//! @brief Counts the number of pixels in an image that are in each exposure bin
//! using shared memory, constant memory, global memory, and registers on the
//! GPU.
//!
//! @param[in]  d_image    A 1-D array of 8-bit pixel values in the image
//! @param[out] d_hist     A 1-D array counting the number of pixels in each bin
//! @param[in]  num_pixels The total number of pixels in the image
//!
__global__ void
gpu_histogram_shared_mem(const std::uint8_t *const __restrict__ d_image,
                         unsigned int *const __restrict__ d_hist,
                         std::size_t num_pixels) {
  // Shared memory array to count the number of pixels in each exposure bin
  __shared__ unsigned int s_hist[NUM_BINS]; // NOLINT

  // Need to initialize the count for each bin with zero
  for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
    s_hist[b] = 0;
  }
  // Block until the entire shared memory array is initialized
  __syncthreads();

  // There will almost always be less total threads than pixels, so each thread
  // needs to operate on multiple pixels in a grid-stride loop. The stride is
  // the total number of threads in a grid.
  std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < num_pixels; i += stride) {
    // Get the current pixel, map it to an exposure bin using the lookup table,
    // and increment the count for that bin in the output histogram
    atomicAdd(&s_hist[c_bin_lut[d_image[i]]], 1);
  }
  __syncthreads();

  // Add this block's counts to the global memory histogram
  for (unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
    if (s_hist[b] != 0) {
      atomicAdd(&d_hist[b], s_hist[b]);
    }
  }
}

//!
//! @brief Counts the number of pixels in an image that are in each exposure bin
//! using constant memory, global memory, and registers on the GPU.
//!
//! @param[in]  d_image    A 1-D array of 8-bit pixel values in the image
//! @param[out] d_hist     A 1-D array counting the number of pixels in each bin
//! @param[in]  num_pixels The total number of pixels in the image
//!
__global__ void
gpu_histogram_global_mem(const std::uint8_t *const __restrict__ d_image,
                         unsigned int *const __restrict__ d_hist,
                         std::size_t num_pixels) {
  // There will almost always be less total threads than pixels, so each thread
  // needs to operate on multiple pixels in a grid-stride loop. The stride is
  // the total number of threads in a grid.
  std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t i =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < num_pixels; i += stride) {
    // Get the current pixel, map it to an exposure bin using the lookup table,
    // and increment the count for that bin in the output histogram
    atomicAdd(&d_hist[c_bin_lut[d_image[i]]], 1u);
  }
}

//!
//! @brief Verifies the histogram was calculated correctly
//!
//! @param gpu_hist The histogram output from the GPU
//! @param ref_hist The expected histogram calculated on the CPU
//! @return true if the histograms match, else false
//!
bool verify_results(const std::vector<unsigned int> &gpu_hist,
                    const std::vector<unsigned int> &ref_hist) {
  std::cout << "------------------------------------------------------\n";
  std::cout << std::format("{:<18} {:>9} {:>12} {:>12}\n", "Bin", "Values",
                           "GPU", "CPU");
  std::cout << "------------------------------------------------------\n";

  bool passed = true;
  for (int b = 0; b < NUM_BINS; ++b) {
    const int lo = BIN_STARTS[b];
    const int hi = (b + 1 < NUM_BINS) ? BIN_STARTS[b + 1] - 1 : NUM_VALUES - 1;
    const bool match = (gpu_hist[b] == ref_hist[b]);
    passed = passed && match;
    std::cout << std::format("{:<18} {:>4}-{:<4} {:>12} {:>12}{}\n",
                             BIN_NAMES[b], lo, hi, gpu_hist[b], ref_hist[b],
                             match ? "" : " [MISMATCH]");
  }
  std::cout << "------------------------------------------------------\n";
  std::cout << (passed ? "[+] Exposure bin verification PASSED\n\n"
                       : "[-] Exposure bin verification FAILED\n\n");
  return passed;
}

//!
//! @brief Executes a kernel and measures its exeuction time
//!
//! @param[in] kernel      Function pointer for kernel to execute
//! @param[in] name        An identifying name of the kernel being executed
//! @param[in] d_image     The input image in device memory
//! @param[in] d_hist      The output histogram in device memory
//! @param[in] ref_hist    The expected output histogram
//! @param[in] block_size  The number of threads per block
//! @param[in] num_blocks  The number of blocks in the grid
//! @return true if the results of the kernel are valid, else false
//!
bool benchmark_kernel(HistogramKernel kernel, const std::string &name,
                      const std::uint8_t *d_image, unsigned int *d_hist,
                      const std::vector<unsigned int> &ref_hist,
                      long long block_size, long long num_blocks) {
  const dim3 grid(static_cast<unsigned int>(num_blocks));
  const dim3 block(static_cast<unsigned int>(block_size));

  // "Warm-up" the kernel to mitigate the effects of lazy loading and
  // initialization.
  // https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html#impact-on-performance-measurements
  kernel<<<grid, block>>>(d_image, d_hist, NUM_PIXELS);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemset(d_hist, 0, sizeof(unsigned int) * NUM_BINS));

  cudaEvent_t start = nullptr, stop = nullptr;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  float gpu_elapsed_ms = 0.0f;

  checkCudaErrors(cudaEventRecord(start));
  kernel<<<grid, block>>>(d_image, d_hist, NUM_PIXELS);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));
  checkCudaErrors(cudaEventElapsedTime(&gpu_elapsed_ms, start, stop));

  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));

  // Copy the result histogram back to the host
  std::vector<unsigned int> gpu_hist(NUM_BINS);
  checkCudaErrors(cudaMemcpy(gpu_hist.data(), d_hist,
                             sizeof(unsigned int) * NUM_BINS,
                             cudaMemcpyDeviceToHost));

  std::cout << std::format("[+] {}\n", name);
  std::cout << std::format("[+] Kernel execution time: {:.6f} ms\n",
                           gpu_elapsed_ms);

  return verify_results(gpu_hist, ref_hist);
}

//!
//! @brief Initializes input/output buffers and executes the histogram kernel
//!
//! @param[in] block_size The number of threads per block
//! @param[in] num_blocks The number of blocks in the grid
//! @return true if the kernel executed successfully, else false
//!
bool execute_gpu_functions(long long block_size, long long num_blocks) {
  // Allocate and initialize the host input image pixel values
  std::uint8_t *h_image = nullptr;
  checkCudaErrors(cudaMallocHost(&h_image, sizeof(std::uint8_t) * NUM_PIXELS));
  fill_test_image(h_image);

  // Assemble the lookup table and expected histogram result for the test image
  const auto lut = build_exposure_bin_lut();
  const auto ref_hist = compute_reference_hist(h_image, lut);

  // Allocate device vectors
  std::uint8_t *d_image = nullptr;
  unsigned int *d_hist = nullptr;
  checkCudaErrors(cudaMalloc(&d_image, sizeof(std::uint8_t) * NUM_PIXELS));
  checkCudaErrors(cudaMalloc(&d_hist, sizeof(unsigned int) * NUM_BINS));
  // Zero memory to prevent compute-sanitizer initcheck warnings
  checkCudaErrors(cudaMemset(d_image, 0, sizeof(std::uint8_t) * NUM_PIXELS));
  checkCudaErrors(cudaMemset(d_hist, 0, sizeof(unsigned int) * NUM_BINS));

  // Copy the bin lookup table to constant memory
  // Reference:
  // https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#constant-memory
  checkCudaErrors(cudaMemcpyToSymbol(c_bin_lut, lut.data(), sizeof(c_bin_lut)));

  // Transfer the test image to GPU memory
  checkCudaErrors(cudaMemcpy(d_image, h_image,
                             sizeof(std::uint8_t) * NUM_PIXELS,
                             cudaMemcpyHostToDevice));

  bool passed = benchmark_kernel(gpu_histogram_shared_mem,
                                 "Histogram Shared Memory Kernel", d_image,
                                 d_hist, ref_hist, block_size, num_blocks);
  passed = passed && benchmark_kernel(gpu_histogram_global_mem,
                                      "Histogram Global Memory Kernel", d_image,
                                      d_hist, ref_hist, block_size, num_blocks);

  checkCudaErrors(cudaFreeHost(h_image));
  checkCudaErrors(cudaFree(d_image));
  checkCudaErrors(cudaFree(d_hist));

  return passed;
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

  std::cout << std::format("[+] Image size: {} x {} ({} pixels)\n", NUM_COLS,
                           NUM_ROWS, NUM_PIXELS);
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
