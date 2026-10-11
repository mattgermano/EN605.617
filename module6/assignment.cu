//!
//! @file assignment.cu
//! @author Matt Germano (mgerman8@jh.edu)
//! @brief Generates a test RF signal, applies a windowing function, runs a
//! FFT, and computes the magnitude of each frequency bin. The input data is
//! chunked across different numbers of CUDA streams to test performance.
//! @version 0.1
//! @date 2026-10-06
//!
//! @copyright Copyright (c) 2026
//!

// I moved some of the common functions that I keep using into a utility header
// file in the EN605.617/common/inc directory. This includes checking for CUDA
// errors and parsing CLI arguments.
#include <Utilities.hpp>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cuda/cmath>
#include <cuda_runtime.h>
#include <cufft.h>
#include <format>
#include <iostream>
#include <numbers>
#include <vector>

// Number of I/Q samples per FFT. This maps to the number of output frequency
// bins. A higher value produces better frequency resolution at the cost of less
// time resolution.
static constexpr int FFT_SIZE = 1024;

// A frame is a slice of the input signal and is equal to the FFT size. cuFFT
// can compute an FFT on many frames at once so we can group them into a larger
// chunk for efficiency.
static constexpr int FRAMES_PER_CHUNK = 256;
static constexpr long long CHUNK_SAMPLES =
    static_cast<long long>(FFT_SIZE) * FRAMES_PER_CHUNK;

// The maximum number of parallel CUDA streams to test the program with
static constexpr int MAX_STREAMS = 8;

// Data from a radio is typically received as int16 complex numbers, but FFT's
// operate on complex floating point numbers. This scaling factor will convert
// the int16 magnitude to a floating point magnitude between -1.0 and 1.0.
static constexpr float INT16_SCALE = 1.0f / 32767.0f;

// Sample rate (samples per second) of the generated signal
static constexpr double SAMPLE_RATE_SPS = 1.024e6;

// Frequency of the generated signal relative to where the radio is tuned to
static constexpr double TONE_FREQUENCY_HZ = 100.0e3;

// The frequency width of each bin in the output FFT
static constexpr double BIN_WIDTH = SAMPLE_RATE_SPS / FFT_SIZE;
// The frequency bin in which the generated signal will appear
static constexpr int EXPECTED_BIN =
    (static_cast<int>(TONE_FREQUENCY_HZ / BIN_WIDTH) + FFT_SIZE) % FFT_SIZE;

// The FFT sums over each sample so it's necessary to divide by the FFT size to
// restore the original amplitude. Multiplying by 2.0 is needed since the Hann
// window reduces the peak amplitude by half.
static constexpr float MAGNITUDE_SCALE = 2.0f / FFT_SIZE;
static constexpr float POWER_SCALE = MAGNITUDE_SCALE * MAGNITUDE_SCALE;

// Prevents taking a log of 0
static constexpr float MIN_POWER = 1.0e-20f;
// Peak magnitude must be within this threshold to be valid
static constexpr float PEAK_THRESHOLD_DB = 0.1f;

// Structure to store data associated with each stream. Each stream has its own
// input/output buffers and events to signal completion.
struct StreamData {
  cudaStream_t stream = nullptr;
  cufftHandle plan = 0;
  short2 *d_input = nullptr;
  cufftComplex *d_output = nullptr;
  float *d_magnitude_db = nullptr;
  cudaEvent_t complete = nullptr;
};

//!
//! @brief Converts a complex int16 input signal to complex floating point and
//! applies a Hann windowing function.
//!
//! @param[in]  input  The int16 complex input signal
//! @param[out] output The floating point complex output signal with a window
//! applied
//! @param[in]  num_samps The number of input samples
//!
__global__ void convert_and_window(const short2 *__restrict__ input,
                                   cufftComplex *__restrict__ output,
                                   std::size_t num_samps) {
  std::size_t thread_idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (thread_idx >= num_samps) {
    return;
  }

  // Get the sample number within the FFT length
  int n = static_cast<int>(thread_idx % FFT_SIZE);

  // Apply a Hann window which tapers the ends of the signal to 0. This is
  // needed since an FFT assumes a signal is periodic where both ends of the
  // signal are close in value. This is often not the case for a real signal.
  // Reference: https://pysdr.org/content/frequency_domain#windowing
  // Formula: https://en.wikipedia.org/wiki/Window_function#Hann_window
  float w = 0.5f * (1.0f - std::cosf(2.0f * std::numbers::pi_v<float> *
                                     static_cast<float>(n) / FFT_SIZE));

  // Scale the int16 value to the range -1.0 to 1.0 and multiply by the
  // windowing function
  output[thread_idx] = make_cuFloatComplex(
      static_cast<float>(input[thread_idx].x) * INT16_SCALE * w,
      static_cast<float>(input[thread_idx].y) * INT16_SCALE * w);
}

//!
//! @brief Calculates the magnitude in dB of each frequency bin.
//!
//! @param[in]  spectrum      The complex FFT output
//! @param[out] magnitude_db  The magnitudes of each frequency bin in dB
//! @param[in]  num_samps     The number of frequency bins
//!
__global__ void compute_magnitude_db(const cufftComplex *__restrict__ spectrum,
                                     float *__restrict__ magnitude_db,
                                     std::size_t num_samps) {
  std::size_t thread_idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (thread_idx >= num_samps) {
    return;
  }

  // Power is just the real component squared + the imaginary component squared
  float power = (spectrum[thread_idx].x * spectrum[thread_idx].x +
                 spectrum[thread_idx].y * spectrum[thread_idx].y) *
                POWER_SCALE;
  // dB formula is 10log10 of the power
  // https://en.wikipedia.org/wiki/Decibel#Power_quantities
  magnitude_db[thread_idx] = 10.0f * std::log10f(std::fmaxf(power, MIN_POWER));
}

//!
//! @brief Generates an int16 complex RF signal at a specific frequency and
//! sample rate
//!
//! @param[out] signal    The generated signal
//! @param[in]  num_samps The number of samples to generate
//!
void generate_signal(short2 *signal, std::size_t num_samps) {
  for (std::size_t i = 0; i < num_samps; ++i) {
    // Get the time of the current sample
    double t = static_cast<double>(i) / SAMPLE_RATE_SPS;
    double phase = 2.0 * std::numbers::pi * TONE_FREQUENCY_HZ * t;

    // Each of these is a coordinate on the complex plane
    // Need to multiply to 32,676 to scale to an int16
    signal[i] =
        make_short2(static_cast<short>(std::lrint(32767.0 * std::cos(phase))),
                    static_cast<short>(std::lrint(32767.0 * std::sin(phase))));
  }
}

//!
//! @brief Verify the peak magnitude within each FFT of the input signal
//! appears in the correct frequency bin
//!
//! @param magnitude  The array of signal magnitudes (in dB)
//! @param num_frames The number of signal frames
//! @return true if the peak resides in the correct frequency bin, else false
//!
bool verify_result(const float *magnitude, long long num_frames) {
  bool passed = true;

  // Loop through each frame and confirm the frequency bin within that frame
  // with the max magnitude matches where the test signal was placed
  for (long long f = 0; f < num_frames; ++f) {
    const float *m = magnitude + f * FFT_SIZE;
    int peak = 0;

    for (int b = 1; b < FFT_SIZE; ++b) {
      if (m[b] > m[peak]) {
        // Found a bin with a new peak magnitude
        peak = b;
      }
    }

    // Confirm peak is in the expected bin and is also at 0 dB. The signal was
    // scaled to -1.0 to 1.0 and 10log10(1.0) is 0 dB.
    if (peak != EXPECTED_BIN || std::abs(m[peak]) > PEAK_THRESHOLD_DB) {
      std::cout << std::format(
          "Frame {}: Peak bin {} (Expected {}), Level {:.3f} dB\n", f, peak,
          EXPECTED_BIN, m[peak]);
      passed = false;
    }
  }

  return passed;
}

//!
//! @brief Executes the processing kernels for each chunk of input signal
//!
//! @param[in] streams     The vector of data associated with each stream
//! @param[in] num_streams The number of streams to chunk the data across
//! @param[in] h_input     The input signal
//! @param[out] h_output   The array containing the magnitude of each freq bin
//! @param[in] num_chunks  The number of times to divide the input signal
//! @param[in] block_size  The number of threads per block
//!
void run_kernels(const std::vector<StreamData> &streams, int num_streams,
                 const short2 *h_input, float *h_output, long long num_chunks,
                 long long block_size) {
  const dim3 grid(
      static_cast<unsigned int>(cuda::ceil_div(CHUNK_SAMPLES, block_size)));
  const dim3 block(static_cast<unsigned int>(block_size));

  for (long long c = 0; c < num_chunks; ++c) {
    const auto &stream = streams[c % num_streams];

    // Copy a specific chunk of the input signal to the device
    checkCudaErrors(cudaMemcpyAsync(stream.d_input, h_input + c * CHUNK_SAMPLES,
                                    CHUNK_SAMPLES * sizeof(short2),
                                    cudaMemcpyHostToDevice, stream.stream));

    // Convert the signal to floating point and apply windowing function
    convert_and_window<<<grid, block, 0, stream.stream>>>(
        stream.d_input, stream.d_output, CHUNK_SAMPLES);
    checkCudaErrors(cudaGetLastError());

    // Execute the FFT on the chunk of samples
    checkCufftErrors(cufftExecC2C(stream.plan, stream.d_output, stream.d_output,
                                  CUFFT_FORWARD));

    // Convert each frequency bin to dB
    compute_magnitude_db<<<grid, block, 0, stream.stream>>>(
        stream.d_output, stream.d_magnitude_db, CHUNK_SAMPLES);
    checkCudaErrors(cudaGetLastError());

    // Copy the dB magnitudes from the device to the host
    checkCudaErrors(cudaMemcpyAsync(
        h_output + c * CHUNK_SAMPLES, stream.d_magnitude_db,
        CHUNK_SAMPLES * sizeof(float), cudaMemcpyDeviceToHost, stream.stream));
  }
}

//!
//! @brief Processes the data through the kernels with a variable number of CUDA
//! streams
//!
//! @param[in]  streams     The vector of data associated with each stream
//! @param[in]  num_streams The number of streams to chunk the data across
//! @param[in]  h_input     The input signal
//! @param[out] h_output    The array containing the magnitude of each freq bin
//! @param[in]  num_chunks  The number of times to divide the input signal
//! @param[in]  block_size  The number of threads per block
//! @return float           The elapsed execution time across all streams
//!
float benchmark_streams(const std::vector<StreamData> &streams, int num_streams,
                        const short2 *h_input, float *h_output,
                        long long num_chunks, long long block_size) {
  cudaEvent_t start = nullptr, stop = nullptr;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  cudaStream_t first_stream = streams[0].stream;
  checkCudaErrors(cudaEventRecord(start, first_stream));
  for (int i = 1; i < num_streams; ++i) {
    // Ensure all streams block until the start event has been reached
    checkCudaErrors(cudaStreamWaitEvent(streams[i].stream, start, 0));
  }

  run_kernels(streams, num_streams, h_input, h_output, num_chunks, block_size);

  for (int i = 1; i < num_streams; ++i) {
    // Wait for all streams besides the first to complete
    checkCudaErrors(cudaEventRecord(streams[i].complete, streams[i].stream));
    checkCudaErrors(cudaStreamWaitEvent(first_stream, streams[i].complete));
  }

  // ...now synchronize with the first stream
  checkCudaErrors(cudaEventRecord(stop, first_stream));
  checkCudaErrors(cudaEventSynchronize(stop));

  float gpu_elapsed_ms = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&gpu_elapsed_ms, start, stop));
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));

  return gpu_elapsed_ms;
}

//!
//! @brief Initializes each CUDA stream and associated memory/events
//!
//! @return The vector of CUDA stream data
//!
std::vector<StreamData> initialize_streams() {
  std::vector<StreamData> streams(MAX_STREAMS);
  for (auto &stream : streams) {
    checkCudaErrors(
        cudaStreamCreateWithFlags(&stream.stream, cudaStreamNonBlocking));
    checkCudaErrors(
        cudaMalloc(&stream.d_input, CHUNK_SAMPLES * sizeof(short2)));
    checkCudaErrors(
        cudaMalloc(&stream.d_output, CHUNK_SAMPLES * sizeof(cufftComplex)));
    checkCudaErrors(
        cudaMalloc(&stream.d_magnitude_db, CHUNK_SAMPLES * sizeof(float)));
    checkCufftErrors(
        cufftPlan1d(&stream.plan, FFT_SIZE, CUFFT_C2C, FRAMES_PER_CHUNK));
    checkCufftErrors(cufftSetStream(stream.plan, stream.stream));
    // None of these events are used for timing and the programming guide
    // recommends disabling timing if it is not needed
    checkCudaErrors(
        cudaEventCreateWithFlags(&stream.complete, cudaEventDisableTiming));
  }

  return streams;
}

//!
//! @brief Cleans up the data associated with a stream
//!
//! @param[in] streams The vector of CUDA stream data
//!
void cleanup_streams(const std::vector<StreamData> &streams) {
  for (auto &stream : streams) {
    checkCufftErrors(cufftDestroy(stream.plan));
    checkCudaErrors(cudaFree(stream.d_input));
    checkCudaErrors(cudaFree(stream.d_output));
    checkCudaErrors(cudaFree(stream.d_magnitude_db));
    checkCudaErrors(cudaEventDestroy(stream.complete));
    checkCudaErrors(cudaStreamDestroy(stream.stream));
  }
}

//!
//! @brief Initializes and benchmarks the CUDA streams
//!
//! @param[in] total_threads The total number of threads
//! @param[in] block_size    The number of threads per block
//! @param[in] num_chunks    The number of times to divide the input signal
//! @return true if the kernels executed successfully, else false
//!
bool execute_gpu_functions(long long total_threads, long long block_size,
                           long long num_chunks) {
  // Allocate host vectors and generate input signal
  short2 *h_input = nullptr;
  float *h_output = nullptr;
  checkCudaErrors(cudaMallocHost(&h_input, total_threads * sizeof(short2)));
  checkCudaErrors(cudaMallocHost(&h_output, total_threads * sizeof(float)));
  generate_signal(h_input, total_threads);

  const auto streams = initialize_streams();

  // Perform a warm-up run to mitigate the effects of lazy loading and cuFFT
  // initialization
  benchmark_streams(streams, 1, h_input, h_output, num_chunks, block_size);

  std::cout << "------------------------------------------------------\n";
  std::cout << std::format("{:>8} {:>10} {:>8}\n", "Streams", "Time (ms)",
                           "Speedup");
  bool passed = true;
  float gpu_baseline_ms = 0.0f;
  for (int num_streams = 1; num_streams <= MAX_STREAMS; num_streams *= 2) {
    if (num_streams > num_chunks) {
      break;
    }
    // Reset the output array between runs
    std::memset(h_output, 0, total_threads * sizeof(float));

    float gpu_elapsed_ms = benchmark_streams(streams, num_streams, h_input,
                                             h_output, num_chunks, block_size);
    if (num_streams == 1) {
      gpu_baseline_ms = gpu_elapsed_ms;
    }
    std::cout << std::format("{:>8} {:>10.3f} {:>7.2f}x\n", num_streams,
                             gpu_elapsed_ms, gpu_baseline_ms / gpu_elapsed_ms);
    passed = passed && verify_result(h_output, total_threads / FFT_SIZE);
  }
  std::cout << "------------------------------------------------------\n";

  // Cleanup resources
  cleanup_streams(streams);
  checkCudaErrors(cudaFreeHost(h_input));
  checkCudaErrors(cudaFreeHost(h_output));

  return passed;
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
  if (total_threads <= 0 || total_threads % CHUNK_SAMPLES != 0) {
    std::cerr << std::format(
        "Total threads must be a positive multiple of {}\n", CHUNK_SAMPLES);
    return false;
  }

  long long chunk_threads = CHUNK_SAMPLES;
  long long blocks_per_chunk = 0;
  if (!set_block_dimensions(chunk_threads, block_size, blocks_per_chunk)) {
    return false;
  }
  num_blocks = (total_threads / CHUNK_SAMPLES) * blocks_per_chunk;
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
  std::cout << "[+] NOTE: Execution time includes CPU/GPU memory transfers\n\n";

  const long long num_chunks = total_threads / CHUNK_SAMPLES;
  const long long num_frames = total_threads / FFT_SIZE;
  std::cout << std::format("[+] Frames: {}, Samples/frame: {}, Chunks: {}\n",
                           num_frames, FFT_SIZE, num_chunks);
  std::cout << std::format(
      "[+] Sample rate: {:.3f} MHz, Bin width: {:.1f} Hz\n",
      SAMPLE_RATE_SPS / 1e6, BIN_WIDTH);
  std::cout << std::format(
      "[+] Tone Frequency: {:.1f} kHz, Frequency Bin: {}\n\n",
      TONE_FREQUENCY_HZ / 1e3, EXPECTED_BIN);

  if (!execute_gpu_functions(total_threads, block_size, num_chunks)) {
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
