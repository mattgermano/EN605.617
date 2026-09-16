//!
//! @file assignment.cu
//! @author Matt Germano (mgerman8@jh.edu)
//! @brief Benchmarks LLA to ECEF coordinate conversions on the CPU/GPU
//! @version 0.1
//! @date 2026-09-13
//!
//! @copyright Copyright (c) 2026
//!

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cuda/cmath>
#include <cuda_runtime.h>
#include <format>
#include <iostream>
#include <numbers>
#include <random>
#include <source_location>
#include <span>
#include <string>
#include <vector>

// Maximum number of CUDA blocks in the X-dimension
static constexpr long long MAX_BLOCKS = 2147483647LL;

//!
//! @brief Helper function for checking for CUDA errors
//!
//! @param result The result of a CUDA API call
//! @param loc    The source code location of the function call
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

// Constants required for LLA to ECEF conversion
// Values found here: https://en.wikipedia.org/wiki/World_Geodetic_System#WGS_84
static constexpr double A = 6378137.0; // WGS-84 semi-major axis
static constexpr double E2 =
    6.69437999014e-3; // WGS-84 first eccentricity squared
static constexpr double DEG2RAD = std::numbers::pi / 180.0;

struct LLACoordinate {
  double lat_deg = 0.0;
  double lon_deg = 0.0;
  double alt_m = 0.0;
};

struct ECEFCoordinate {
  double x_m = 0.0;
  double y_m = 0.0;
  double z_m = 0.0;
  // This is a dummy value that is populated when testing kernel branching
  double payload = 0.0;
};

//!
//! @brief Shared host/device function used to convert an LLA (latitude,
//! longitude, altitude) coordinate to an ECEF (earth-centered, earth-fixed)
//! coordinate
//!
//! @param[in]  lla      The LLA coordinate
//! @param[out] ecef_x_m The ECEF X position
//! @param[out] ecef_y_m The ECEF Y position
//! @param[out] ecef_z_m The ECEF Z position
//!
__host__ __device__ inline void lla2ecef(const LLACoordinate &lla,
                                         double &ecef_x_m, double &ecef_y_m,
                                         double &ecef_z_m) {
  const double lat_rad = lla.lat_deg * DEG2RAD;
  const double lon_rad = lla.lon_deg * DEG2RAD;

  // The LLA to ECEF formula is referenced from here:
  // https://www.oc.nps.edu/oc2902w/coord/coordcvt.pdf
  const double r_n =
      A / std::sqrt(1 - (E2 * std::sin(lat_rad) * std::sin(lat_rad)));
  ecef_x_m = (r_n + lla.alt_m) * std::cos(lat_rad) * std::cos(lon_rad);
  ecef_y_m = (r_n + lla.alt_m) * std::cos(lat_rad) * std::sin(lon_rad);
  ecef_z_m = ((1 - E2) * r_n + lla.alt_m) * std::sin(lat_rad);
}

//!
//! @brief Converts an array of LLA (latitude, longitude, altitude) coordinates
//! to ECEF (earth-centered, earth-fixed) coordinates on the CPU
//!
//! @param[in]  lla  An array of LLA coordinates
//! @param[out] ecef An array of ECEF coordinates
//! @param[in]  num_coordinates The number of input/output coordinates
//!
void cpu_lla2ecef(const LLACoordinate *lla, ECEFCoordinate *ecef,
                  std::size_t num_coordinates) {
  for (std::size_t i = 0; i < num_coordinates; ++i) {
    double x = 0.0, y = 0.0, z = 0.0;
    lla2ecef(lla[i], x, y, z);

    ecef[i].x_m = x;
    ecef[i].y_m = y;
    ecef[i].z_m = z;
    ecef[i].payload = 0.0;
  }
}

//!
//! @brief Converts an array of LLA (latitude, longitude, altitude) coordinates
//! to ECEF (earth-centered, earth-fixed) coordinates and re-converts back
//! to geodetic latitude if the coordinate is in the northern hemisphere on
//! the CPU
//!
//! @param[in]  lla  An array of LLA coordinates
//! @param[out] ecef An array of ECEF coordinates
//! @param[in]  num_coordinates The number of input/output coordinates
//!
void cpu_lla2ecef_branching(const LLACoordinate *lla, ECEFCoordinate *ecef,
                            std::size_t num_coordinates) {
  for (std::size_t i = 0; i < num_coordinates; ++i) {
    double x = 0.0, y = 0.0, z = 0.0;
    lla2ecef(lla[i], x, y, z);

    ecef[i].x_m = x;
    ecef[i].y_m = y;
    ecef[i].z_m = z;

    // If we're in the northern hemisphere, convert back to geodetic latitude
    // https://www.oc.nps.edu/oc2902w/coord/coordcvt.pdf
    double payload = 0.0;
    if (z > 0.0) {
      double p = std::sqrt(x * x + y * y);
      double phi = std::atan2(z, p);
      for (int k = 0; k < 8; ++k) {
        double sp = std::sin(phi);
        double rn = A / std::sqrt(1.0 - E2 * sp * sp);
        double alt = p / std::cos(phi) - rn;
        phi = std::atan2(z, p * (1.0 - E2 * rn / (rn + alt)));
      }
      payload = phi;
    }
    ecef[i].payload = payload;
  }
}

//!
//! @brief Converts an array of LLA (latitude, longitude, altitude) coordinates
//! to ECEF (earth-centered, earth-fixed) coordinates on the GPU
//!
//! @param[in]  lla  An array of LLA coordinates
//! @param[out] ecef An array of ECEF coordinates
//! @param[in]  num_coordinates The number of input/output coordinates
//! @sa __restrict__ optimization
//! https://developer.nvidia.com/blog/cuda-pro-tip-optimize-pointer-aliasing/
//!
__global__ void gpu_lla2ecef(const LLACoordinate *__restrict__ lla,
                             ECEFCoordinate *__restrict__ ecef,
                             std::size_t num_coordinates) {
  const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (thread_idx >= num_coordinates) {
    return;
  }

  double x = 0.0, y = 0.0, z = 0.0;
  lla2ecef(lla[thread_idx], x, y, z);

  ecef[thread_idx].x_m = x;
  ecef[thread_idx].y_m = y;
  ecef[thread_idx].z_m = z;
  ecef[thread_idx].payload = 0.0;
}

//!
//! @brief Converts an array of LLA (latitude, longitude, altitude) coordinates
//! to ECEF (earth-centered, earth-fixed) coordinates and re-converts back
//! to geodetic latitude if the coordinate is in the northern hemisphere on
//! the GPU
//!
//! @param[in]  lla  An array of LLA coordinates
//! @param[out] ecef An array of ECEF coordinates
//! @param[in]  num_coordinates The number of input/output coordinates
//! @sa __restrict__ optimization
//! https://developer.nvidia.com/blog/cuda-pro-tip-optimize-pointer-aliasing/
//!
__global__ void gpu_lla2ecef_branching(const LLACoordinate *__restrict__ lla,
                                       ECEFCoordinate *__restrict__ ecef,
                                       std::size_t num_coordinates) {
  const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (thread_idx >= num_coordinates) {
    return;
  }

  double x = 0.0, y = 0.0, z = 0.0;
  lla2ecef(lla[thread_idx], x, y, z);

  ecef[thread_idx].x_m = x;
  ecef[thread_idx].y_m = y;
  ecef[thread_idx].z_m = z;

  // If we're in the northern hemisphere, convert back to geodetic latitude
  // https://www.oc.nps.edu/oc2902w/coord/coordcvt.pdf
  double payload = 0.0;
  if (z > 0.0) {
    double p = std::sqrt(x * x + y * y);
    double phi = std::atan2(z, p);
    for (int k = 0; k < 8; ++k) {
      double sp = std::sin(phi);
      double rn = A / std::sqrt(1.0 - E2 * sp * sp);
      double alt = p / std::cos(phi) - rn;
      phi = std::atan2(z, p * (1.0 - E2 * rn / (rn + alt)));
    }
    payload = phi;
  }
  ecef[thread_idx].payload = payload;
}

//!
//! @brief Helper function for running the branching/non-branching GPU kernel
//! and determining its execution time
//!
//! @param[in] d_lla        The LLA device pointer
//! @param[in] d_ecef       The ECEF device pointer
//! @param[in] length       The number of coordinates
//! @param[in] num_blocks   The number of blocks to execute the kernel with
//! @param[in] block_size   The number of threads per block
//! @param[in] branching    When true, executes the branching kernel
//! @param[in] warmup_count The number of times to warmup the kernel
//! @param[in] run_count    The number of times to run the kernel
//! @return The elapsed execution time of the kernel (in ms)
//!
float time_gpu_kernel(LLACoordinate *d_lla, ECEFCoordinate *d_ecef,
                      std::size_t length, unsigned int num_blocks,
                      unsigned int block_size, bool branching = false,
                      std::size_t warmup_count = 1, std::size_t run_count = 5) {
  // Create events used to time kernel execution
  cudaEvent_t start_gpu = nullptr, stop_gpu = nullptr;
  checkCudaErrors(cudaEventCreate(&start_gpu));
  checkCudaErrors(cudaEventCreate(&stop_gpu));
  float gpu_elapsed_ms = 0.0f;

  // "Warm-up" the kernel to mitigate the effects of lazy loading and
  // initialization.
  // https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html#impact-on-performance-measurements
  for (std::size_t i = 0; i < warmup_count; ++i) {
    if (branching) {
      gpu_lla2ecef_branching<<<num_blocks, block_size>>>(d_lla, d_ecef, length);
    } else {
      gpu_lla2ecef<<<num_blocks, block_size>>>(d_lla, d_ecef, length);
    }
  }
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // ...now run the kernel in a loop to measure average timing
  checkCudaErrors(cudaEventRecord(start_gpu));
  for (std::size_t i = 0; i < run_count; ++i) {
    if (branching) {
      gpu_lla2ecef_branching<<<num_blocks, block_size>>>(d_lla, d_ecef, length);
    } else {
      gpu_lla2ecef<<<num_blocks, block_size>>>(d_lla, d_ecef, length);
    }
  }
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaEventRecord(stop_gpu));
  checkCudaErrors(cudaEventSynchronize(stop_gpu));
  checkCudaErrors(cudaEventElapsedTime(&gpu_elapsed_ms, start_gpu, stop_gpu));

  checkCudaErrors(cudaEventDestroy(start_gpu));
  checkCudaErrors(cudaEventDestroy(stop_gpu));

  return gpu_elapsed_ms / static_cast<float>(run_count);
}

int main(int argc, char **argv) {
  // Read command line arguments
  long long total_threads = (1 << 22);
  long long block_size = 256;

  if (argc >= 2) {
    try {
      total_threads = std::stoll(argv[1]);
      if (total_threads <= 0) {
        std::cerr << "Total threads must be greater than 0\n";
        return 1;
      }
    } catch (const std::exception &) {
      std::cerr << std::format("Failed to convert '{}' to a number\n", argv[1]);
      return 1;
    }
  }
  if (argc >= 3) {
    try {
      block_size = std::stoll(argv[2]);
      if (block_size <= 0 || block_size > 1024) {
        std::cerr << "Block size must be between 1 and 1024\n";
        return 1;
      }
    } catch (const std::exception &) {
      std::cerr << std::format("Failed to convert '{}' to a number\n", argv[2]);
      return 1;
    }
  }

  // Validate command line arguments
  long long num_blocks = cuda::ceil_div(total_threads, block_size);
  if (num_blocks > MAX_BLOCKS) {
    std::cerr << std::format(
        "Number of blocks ({}) is over the maximum of {}\n", num_blocks,
        MAX_BLOCKS);
    return 1;
  }

  if (total_threads != (num_blocks * block_size)) {
    total_threads = num_blocks * block_size;
    std::cout << "Warning: Total thread count is not evenly divisible by the "
                 "block size\n";
    std::cout << std::format(
        "The total number of threads will be rounded up to {}\n",
        total_threads);
  }

  std::cout << std::format("[+] Total threads: {}\n", total_threads);
  std::cout << std::format("[+] Total blocks: {}\n", num_blocks);
  std::cout << std::format("[+] Threads per block: {}\n", block_size);

  // ----------------------------------------------------------------
  // BEGIN: Initialize
  // ----------------------------------------------------------------

  // Allocate host vectors
  //
  // The CUDA programming guide recommends using cudaMallocHost when buffers
  // will be used to copy data between CPU and GPU memory.
  // https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html#explicit-memory-management
  LLACoordinate *h_lla = nullptr;
  ECEFCoordinate *h_ecef = nullptr;
  checkCudaErrors(
      cudaMallocHost(&h_lla, total_threads * sizeof(LLACoordinate)));
  checkCudaErrors(
      cudaMallocHost(&h_ecef, total_threads * sizeof(ECEFCoordinate)));

  // Use the same random number generator seed so the results are the same
  // between runs
  std::mt19937 gen(1234567); // NOLINT

  std::uniform_real_distribution<double> lat_n(10.0, 80.0);
  std::uniform_real_distribution<double> lat_s(-80.0, -10.0);
  std::uniform_real_distribution<double> lon(-180.0, 180.0);

  // Initialize the LLA coordinates where half of the coordinates are in the
  // northern hemisphere and half are in the southern hemisphere
  for (std::size_t i = 0; i < total_threads; ++i) {
    h_lla[i] = {.lat_deg = (i < total_threads / 2) ? lat_n(gen) : lat_s(gen),
                .lon_deg = lon(gen),
                .alt_m = 435.0};
  }

  // Allocate device vectors
  LLACoordinate *d_lla = nullptr;
  ECEFCoordinate *d_ecef = nullptr;

  checkCudaErrors(cudaMalloc(&d_lla, total_threads * sizeof(LLACoordinate)));
  checkCudaErrors(cudaMalloc(&d_ecef, total_threads * sizeof(ECEFCoordinate)));
  // Zeroing memory to initialize all elements
  checkCudaErrors(
      cudaMemset(d_ecef, 0, total_threads * sizeof(ECEFCoordinate)));
  checkCudaErrors(cudaMemcpy(d_lla, h_lla,
                             total_threads * sizeof(LLACoordinate),
                             cudaMemcpyHostToDevice));

  // ----------------------------------------------------------------
  // END: Initialize
  // ----------------------------------------------------------------

  // ----------------------------------------------------------------
  // BEGIN: Execute the kernel on the GPU (non-branching)
  // ----------------------------------------------------------------

  float gpu_elapsed_ms =
      time_gpu_kernel(d_lla, d_ecef, total_threads, num_blocks, block_size);

  std::cout << std::format("[+] Avg kernel execution time excluding copies "
                           "(non-branching): {:.6f} ms\n",
                           gpu_elapsed_ms);

  checkCudaErrors(cudaMemcpy(h_ecef, d_ecef,
                             total_threads * sizeof(ECEFCoordinate),
                             cudaMemcpyDeviceToHost));

  // ----------------------------------------------------------------
  // END: Execute the kernel on the GPU (non-branching)
  // ----------------------------------------------------------------

  // ----------------------------------------------------------------
  // BEGIN: Execute the function on the CPU (non-branching)
  // ----------------------------------------------------------------

  auto start_cpu = std::chrono::steady_clock::now();
  cpu_lla2ecef(h_lla, h_ecef, total_threads);
  auto stop_cpu = std::chrono::steady_clock::now();
  auto cpu_elapsed_ms =
      std::chrono::duration<double, std::milli>(stop_cpu - start_cpu);

  std::cout << std::format(
      "[+] CPU execution time (non-branching): {:.6f} ms\n",
      cpu_elapsed_ms.count());

  // ----------------------------------------------------------------
  // END: Execute the function on the CPU (non-branching)
  // ----------------------------------------------------------------

  // ----------------------------------------------------------------
  // BEGIN: Execute the kernel on the GPU (branching)
  // ----------------------------------------------------------------

  gpu_elapsed_ms = time_gpu_kernel(d_lla, d_ecef, total_threads, num_blocks,
                                   block_size, true);

  std::cout << std::format("[+] Avg kernel execution time excluding copies "
                           "(branching, sorted): {:.6f} ms\n",
                           gpu_elapsed_ms);

  checkCudaErrors(cudaMemcpy(h_ecef, d_ecef,
                             total_threads * sizeof(ECEFCoordinate),
                             cudaMemcpyDeviceToHost));

  // Shuffle the coordinates to cause a branching penalty since the
  // northern/southern hemisphere coordinates are now mixed together within a
  // warp (whereas before they were contiguously grouped)
  std::span<LLACoordinate> h_lla_span(h_lla, total_threads);
  std::vector<LLACoordinate> shuffled(h_lla_span.begin(), h_lla_span.end());
  std::ranges::shuffle(shuffled, gen);
  checkCudaErrors(cudaMemcpy(d_lla, shuffled.data(),
                             shuffled.size() * sizeof(LLACoordinate),
                             cudaMemcpyHostToDevice));

  gpu_elapsed_ms = time_gpu_kernel(d_lla, d_ecef, total_threads, num_blocks,
                                   block_size, true);

  std::cout << std::format("[+] Avg kernel execution time excluding copies "
                           "(branching, shuffled): {:.6f} ms\n",
                           gpu_elapsed_ms);

  checkCudaErrors(cudaMemcpy(h_ecef, d_ecef,
                             total_threads * sizeof(ECEFCoordinate),
                             cudaMemcpyDeviceToHost));

  // ----------------------------------------------------------------
  // END: Execute the kernel on the GPU (branching)
  // ----------------------------------------------------------------

  // ----------------------------------------------------------------
  // BEGIN: Execute the function on the CPU (branching)
  // ----------------------------------------------------------------

  start_cpu = std::chrono::steady_clock::now();
  cpu_lla2ecef_branching(h_lla, h_ecef, total_threads);
  stop_cpu = std::chrono::steady_clock::now();
  cpu_elapsed_ms =
      std::chrono::duration<double, std::milli>(stop_cpu - start_cpu);

  std::cout << std::format(
      "[+] CPU execution time (branching, sorted): {:.6f} ms\n",
      cpu_elapsed_ms.count());

  start_cpu = std::chrono::steady_clock::now();
  cpu_lla2ecef_branching(shuffled.data(), h_ecef, total_threads);
  stop_cpu = std::chrono::steady_clock::now();
  cpu_elapsed_ms =
      std::chrono::duration<double, std::milli>(stop_cpu - start_cpu);

  std::cout << std::format(
      "[+] CPU execution time (branching, shuffled): {:.6f} ms\n",
      cpu_elapsed_ms.count());

  // ----------------------------------------------------------------
  // END: Execute the function on the CPU (branching)
  // ----------------------------------------------------------------

  // ----------------------------------------------------------------
  // BEGIN: Cleanup
  // ----------------------------------------------------------------

  checkCudaErrors(cudaFreeHost(h_lla));
  checkCudaErrors(cudaFreeHost(h_ecef));
  checkCudaErrors(cudaFree(d_lla));
  checkCudaErrors(cudaFree(d_ecef));

  return 0;
}
