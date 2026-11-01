//!
//! @file assignment.cu
//! @author Matt Germano (mgerman8@jh.edu)
//! @brief Simulates astrophotography image stacking in which multiple images
//! are averaged together to reduce noise. A random starfield with random sensor
//! noise is generated using cuRAND. NPP is used to blur the stars and
//! sum/average the individual exposures.
//! @version 0.1
//! @date 2026-10-27
//!
//! @copyright Copyright (c) 2026
//!

#include <Utilities.hpp>

#include <cuda/cmath>
#include <cuda_runtime.h>
#include <curand.h>
#include <npp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Size of the simulated camera sensor in pixels
static constexpr int WIDTH = 6000;
static constexpr int HEIGHT = 4000;
static constexpr int NUM_PIXELS = WIDTH * HEIGHT;

// Number of stars in the simulated image
static constexpr int NUM_STARS = 2000;

// The range of brightness values for a star. This is really a measure of how
// many photons hit the camera sensor.
static constexpr float MIN_STAR_BRIGHTNESS = 20.0f;
static constexpr float MAX_STAR_BRIGHTNESS = 20000.0f;

// Simulate light pollution by adding some background glow to the sky
static constexpr float SKY_BACKGROUND = 100.0f;

// Simulate camera sensor read noise
static constexpr float READ_NOISE = 10.0f;

// Number of exposures to stack and the points at which to save the result
// For example, this will write an output image after 1 exposure (most noise), 4
// exposures (less-noise), and so on.
static constexpr std::array<int, 5> EXPOSURE_COUNTS = {1, 4, 16, 64, 128};
static constexpr int MAX_FRAMES = EXPOSURE_COUNTS.back();

static constexpr unsigned long long RNG_SEED = 7148;
static constexpr int BLOCK_SIZE = 256;

// Image is stored as 1-D array of floats. NPP needs a step size which is just
// the length of one row in bytes.
static constexpr int FLOAT_STEP = WIDTH * sizeof(float);
static constexpr std::size_t IMAGE_BYTES = NUM_PIXELS * sizeof(float);
static constexpr NppiSize IMAGE_SIZE = {.width = WIDTH, .height = HEIGHT};

// Directory the output images are written to if one isn't given
static constexpr const char *DEFAULT_OUTPUT_DIR = "output";

// Structure to make it easier to pass state between functions
struct Simulation {
  cudaStream_t stream = nullptr;
  NppStreamContext npp_ctx{};
  curandGenerator_t generator = nullptr;

  // Device memory
  float *d_star_random = nullptr; // Random numbers for the stars
  float *d_points = nullptr;      // Stars as bright pixels
  float *d_truth = nullptr;       // Noise-free sky
  float *d_noise = nullptr;       // Random noise for one exposure
  float *d_frame = nullptr;       // One noisy exposure
  float *d_sum = nullptr;         // Running total of all exposures
  float *d_stack = nullptr;       // The average of all exposures so far
  uint8_t *d_display = nullptr;   // 8-bit image for saving to a file

  // Directory the output images are written to
  std::filesystem::path output_dir = DEFAULT_OUTPUT_DIR;
};

//!
//! @brief Creates a bright pixel for each star in the image
//!
//! @param[in]  random 3 random numbers per star (x pos, y pos, and brightness)
//! @param[out] image  The image to draw the stars in
//!
__global__ void plot_stars_kernel(const float *random, float *image) {
  const unsigned int star = blockIdx.x * blockDim.x + threadIdx.x;
  if (star >= NUM_STARS) {
    return;
  }

  // Map the random numbers to a pixel location
  const int x = min(static_cast<int>(random[3 * star + 0] * WIDTH), WIDTH - 1);
  const int y =
      min(static_cast<int>(random[3 * star + 1] * HEIGHT), HEIGHT - 1);

  // Generate a brightness for the star between the min/max using a logarithmic
  // scale
  const float brightness =
      MIN_STAR_BRIGHTNESS * std::powf(MAX_STAR_BRIGHTNESS / MIN_STAR_BRIGHTNESS,
                                      random[3 * star + 2]);

  atomicAdd(&image[y * WIDTH + x], brightness);
}

//!
//! @brief Simulates taking one exposure of the sky with a camera and adding a
//! fixed amount of sensor read noise to every pixel in the image
//!
//! @param[in]  truth  The noise-free image of the sky
//! @param[in]  normal One random number per pixel
//! @param[out] frame  The simulated exposure
//!
__global__ void make_exposure_kernel(const float *truth, const float *normal,
                                     float *frame) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= NUM_PIXELS) {
    return;
  }

  frame[i] = truth[i] + READ_NOISE * normal[i];
}

//!
//! @brief Converts an image to 8-bit grayscale so it can be viewed. A "stretch"
//! needs to be applied since most of the image has low brightness values.
//!
//! @param[in]  image   The image with brightness values
//! @param[out] display The 8-bit image
//!
__global__ void stretch_kernel(const float *image, uint8_t *display) {
  const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= NUM_PIXELS) {
    return;
  }

  const float BLACK = SKY_BACKGROUND - 10.0f;
  const float WHITE = SKY_BACKGROUND + 300.0f;

  // Maps the brightness onto a 0 to 1 scale. BLACK becomes 0 and WHITE becomes
  // 1. Anything darker than BLACK is clamped to 0 and anything brighter than
  // WHITE is clamped to 1.
  const float scaled =
      std::fminf(std::fmaxf((image[i] - BLACK) / (WHITE - BLACK), 0.0f), 1.0f);

  // Take the square root to brighten dim pixels, while leaving bright ones
  // largely unchanged (e.g., 0.04 becomes 0.2, but 0.81 only becomes 0.9). That
  // result is then scaled to the nearest 8-bit pixel value.
  display[i] =
      static_cast<std::uint8_t>(std::lroundf(std::sqrtf(scaled) * 255.0f));
}

//!
//! @brief Creates an NPP context required for specific function calls
//!
//! @param[in] stream The CUDA stream to run NPP functions on
//! @return The NPP stream context
//!
static NppStreamContext create_npp_context(cudaStream_t stream) {
  NppStreamContext ctx{};
  checkCudaErrors(cudaGetDevice(&ctx.nCudaDeviceId));

  cudaDeviceProp prop{};
  checkCudaErrors(cudaGetDeviceProperties(&prop, ctx.nCudaDeviceId));

  ctx.hStream = stream;
  ctx.nMultiProcessorCount = prop.multiProcessorCount;
  ctx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
  ctx.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
  ctx.nSharedMemPerBlock = prop.sharedMemPerBlock;
  ctx.nCudaDevAttrComputeCapabilityMajor = prop.major;
  ctx.nCudaDevAttrComputeCapabilityMinor = prop.minor;
  checkCudaErrors(cudaStreamGetFlags(stream, &ctx.nStreamFlags));

  return ctx;
}

//!
//! @brief Creates the CUDA stream, the cuRAND generator, and all device memory
//!
//! @return The GPU resources used by the program
//!
static Simulation create_simulation() {
  Simulation sim;
  checkCudaErrors(cudaStreamCreate(&sim.stream));
  sim.npp_ctx = create_npp_context(sim.stream);

  checkCurandErrors(
      curandCreateGenerator(&sim.generator, CURAND_RNG_PSEUDO_DEFAULT));
  checkCurandErrors(
      curandSetPseudoRandomGeneratorSeed(sim.generator, RNG_SEED));
  checkCurandErrors(curandSetStream(sim.generator, sim.stream));

  checkCudaErrors(
      cudaMalloc(&sim.d_star_random, 3UL * NUM_STARS * sizeof(float)));
  checkCudaErrors(cudaMalloc(&sim.d_points, IMAGE_BYTES));
  checkCudaErrors(cudaMalloc(&sim.d_truth, IMAGE_BYTES));
  checkCudaErrors(cudaMalloc(&sim.d_noise, IMAGE_BYTES));
  checkCudaErrors(cudaMalloc(&sim.d_frame, IMAGE_BYTES));
  checkCudaErrors(cudaMalloc(&sim.d_sum, IMAGE_BYTES));
  checkCudaErrors(cudaMalloc(&sim.d_stack, IMAGE_BYTES));
  checkCudaErrors(cudaMalloc(&sim.d_display, NUM_PIXELS));
  return sim;
}

//!
//! @brief Frees all the GPU resources used by the program
//!
//! @param[in] sim The GPU resources to free
//!
static void cleanup_simulation(Simulation &sim) {
  checkCurandErrors(curandDestroyGenerator(sim.generator));
  checkCudaErrors(cudaFree(sim.d_star_random));
  checkCudaErrors(cudaFree(sim.d_points));
  checkCudaErrors(cudaFree(sim.d_truth));
  checkCudaErrors(cudaFree(sim.d_noise));
  checkCudaErrors(cudaFree(sim.d_frame));
  checkCudaErrors(cudaFree(sim.d_sum));
  checkCudaErrors(cudaFree(sim.d_stack));
  checkCudaErrors(cudaFree(sim.d_display));
  checkCudaErrors(cudaStreamDestroy(sim.stream));
  sim = Simulation{};
}

//!
//! @brief Stretches an image on the GPU and saves it as a grayscale Portable
//! Graymap (PGM) file
//!
//! @param[in] sim      The GPU resources
//! @param[in] d_image  The image to save
//! @param[in] filename The name of the file to write in the output directory
//!
static void save_image(const Simulation &sim, const float *d_image,
                       const std::string &filename) {
  const int num_blocks = cuda::ceil_div(NUM_PIXELS, BLOCK_SIZE);
  stretch_kernel<<<num_blocks, BLOCK_SIZE, 0, sim.stream>>>(d_image,
                                                            sim.d_display);
  checkCudaErrors(cudaGetLastError());

  std::vector<std::uint8_t> h_display(NUM_PIXELS);
  checkCudaErrors(cudaMemcpyAsync(h_display.data(), sim.d_display, NUM_PIXELS,
                                  cudaMemcpyDeviceToHost, sim.stream));
  checkCudaErrors(cudaStreamSynchronize(sim.stream));

  const std::filesystem::path path = sim.output_dir / filename;
  std::ofstream file(path, std::ios::binary);

  // Need to add the PGM header for it to be a valid PGM file
  // First line: P5
  // Second line: Optional comment
  // Third line: Width and height of the image
  // Fourth line: Maximum gray level (255 in this case since it's 8-bit)
  // Reference: https://users.wpi.edu/~cfurlong/me-593n/pgmimage.html
  file << std::format("P5\n{} {}\n255\n", WIDTH, HEIGHT);
  file.write(reinterpret_cast<const char *>(h_display.data()), NUM_PIXELS);
  if (!file) {
    std::cerr << std::format("Failed to write {}\n", path.string());
    exit(EXIT_FAILURE);
  }
}

//!
//! @brief Create a noise-free image of the sky. cuRAND is used to generate the
//! locations and brightness of each star. NPP applies a gaussian blur and adds
//! the background sky brightness.
//!
//! @param[in] sim The GPU resources
//!
static void render_truth(const Simulation &sim) {
  // Generate a random location (x, y) and brightness for every star
  checkCurandErrors(
      curandGenerateUniform(sim.generator, sim.d_star_random, 3UL * NUM_STARS));

  checkCudaErrors(cudaMemsetAsync(sim.d_points, 0, IMAGE_BYTES, sim.stream));
  const dim3 grid(cuda::ceil_div(NUM_STARS, BLOCK_SIZE));
  plot_stars_kernel<<<grid, BLOCK_SIZE, 0, sim.stream>>>(sim.d_star_random,
                                                         sim.d_points);
  checkCudaErrors(cudaGetLastError());

  // Blur each star into a round blob using a gaussian blur
  checkNppErrors(nppiFilterGaussBorder_32f_C1R_Ctx(
      sim.d_points, FLOAT_STEP, IMAGE_SIZE, {.x = 0, .y = 0}, sim.d_truth,
      FLOAT_STEP, IMAGE_SIZE, NPP_MASK_SIZE_7_X_7, NPP_BORDER_REPLICATE,
      sim.npp_ctx));

  // Add the sky background brightness to every pixel
  checkNppErrors(nppiAddC_32f_C1IR_Ctx(SKY_BACKGROUND, sim.d_truth, FLOAT_STEP,
                                       IMAGE_SIZE, sim.npp_ctx));

  save_image(sim, sim.d_truth, "truth.pgm");
}

//!
//! @brief Generate a single noisy exposure and add it to the running sum
//!
//! @param[in] sim The GPU resources
//!
static void capture_exposure(const Simulation &sim) {
  // Randomly generate new noise for each exposure
  checkCurandErrors(
      curandGenerateNormal(sim.generator, sim.d_noise, NUM_PIXELS, 0.0f, 1.0f));

  const dim3 grid(cuda::ceil_div(NUM_PIXELS, BLOCK_SIZE));
  make_exposure_kernel<<<grid, BLOCK_SIZE, 0, sim.stream>>>(
      sim.d_truth, sim.d_noise, sim.d_frame);
  checkCudaErrors(cudaGetLastError());

  // Use NPP to add the new exposure to the running total
  checkNppErrors(nppiAdd_32f_C1IR_Ctx(sim.d_frame, FLOAT_STEP, sim.d_sum,
                                      FLOAT_STEP, IMAGE_SIZE, sim.npp_ctx));
}

//!
//! @brief Average the current exposures and save the result
//!
//! @param[in] sim       The GPU resources
//! @param[in] exposures The number of exposures taken so far
//!
static void save_stack(const Simulation &sim, int exposures) {
  // Use NPP to create a stacked image that is the average of all exposures
  // (stack = sum / exposures)
  checkNppErrors(nppiMulC_32f_C1R_Ctx(
      sim.d_sum, FLOAT_STEP, 1.0f / static_cast<float>(exposures), sim.d_stack,
      FLOAT_STEP, IMAGE_SIZE, sim.npp_ctx));

  const std::string filename = std::format("stack_{:02}.pgm", exposures);
  save_image(sim, sim.d_stack, filename);
  std::cout << std::format("Saved {} ({} exposure{} stacked)\n", filename,
                           exposures, exposures == 1 ? "" : "s");
}

//!
//! @brief Takes MAX_FRAMES exposures and adds them to a running total.
//! Saves the current stack at each configured exposure count.
//!
//! @param[in] sim The GPU resources
//!
static void stack_exposures(const Simulation &sim) {
  checkCudaErrors(cudaMemsetAsync(sim.d_sum, 0, IMAGE_BYTES, sim.stream));

  for (int frames = 1; frames <= MAX_FRAMES; frames++) {
    capture_exposure(sim);

    if (std::ranges::find(EXPOSURE_COUNTS, frames) != EXPOSURE_COUNTS.end()) {
      save_stack(sim, frames);
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc > 2) {
    std::cerr << std::format("Usage: {} [output_dir]\n", argv[0]);
    return EXIT_FAILURE;
  }

  Simulation sim = create_simulation();
  if (argc == 2) {
    sim.output_dir = argv[1];
  }
  std::filesystem::create_directories(sim.output_dir);

  std::cout << std::format("Simulating {}x{} camera sensor with {} stars\n",
                           WIDTH, HEIGHT, NUM_STARS);

  render_truth(sim);
  std::cout << "Saved truth.pgm (noise-free sky)\n";
  stack_exposures(sim);
  std::cout << std::format("\nImages written to {}/\n",
                           sim.output_dir.string());

  cleanup_simulation(sim);
  return EXIT_SUCCESS;
}
