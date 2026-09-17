# Module 3 Assignment

The following directory contains the source code and answers to the assignment
questions for Module 3.

## Build and Run

The code is intended to be automatically built and run using the assignment
runner. This can be accomplished by executing the following command from the
root of the repository:

```bash
./assignment_build_execution/run_assignments.sh
```

The above command requires Python 3, uv, CMake, and CUDA/NVCC. The assignment
runner executes the [`build.sh`](./build.sh) and [`run.sh`](./run.sh) scripts to
build and run the project with different configurations.

Alternatively, the code can be manually built using the following commands:

```bash
cd module3
cmake -S . -B build -D CMAKE_BUILD_TYPE="Release"
cmake --build build --parallel $(nproc --ignore=1)
```

If CMake isn't available, you can also run a simple `make` command from this
directory:

```bash
make
```

It can then be run as follows:

```bash
./build/assignment.exe <total_threads> <block_size>

# ...for example
./build/assignment.exe 4194304 256
```

## Branching Performance Comparison

## Prior Submission Review

The following sections identify good and bad qualities of the code provided in
the assignment text.

### Bad Qualities

1. The host arrays (`int a[N], b[N], c[N];`) are allocated on the stack. On
   Linux, the default max stack size is typically 8 MB which will significantly
   limit the maximum value for `N`. It is common to test CUDA code with large
   datasets and this program will crash on startup if `N` is around 700,000 or
   greater. The buffers can be allocated on the heap instead using `std::vector`
   or by calling `cudaHostMalloc` which the CUDA programming guide recommends
   for data that will need to be transferred between the CPU and the GPU.

1. CUDA kernels are launched asynchronously and return immediately. The code is
   attempting to measure the execution time of the CUDA kernel, but it is
   actually measuring the time it takes to launch the kernel. A synchronization
   mechanism would have to be called (such as `cudaDeviceSynchronize`) prior to
   getting the stop time to wait for the kernel to finish executing.
   Alternatively, [CUDA
   events](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html#timing-operations-in-cuda-streams)
   could be used to record start/stop events.

1. In addition to the prior timing issue, the [lazy
   loading](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html#)
   behavior of CUDA modules will cause the first kernel execution to take longer
   which impacts timing measurements. The CUDA programming guide recommends
   doing a warmup execution of the kernel and preloading the kernel prior to
   measuring its execution time to avoid the loading penalty which can cause
   misleading timing results.

1. Depending on the value of `N`, assigning `i * i` to `b[i]` can overflow an
   integer.

1. There is no error checking on any of the CUDA runtime API function calls.
   Every API function returns a `cudaError_t` enumeration. On success, it is
   equal to `cudaSuccess`. Properly handling errors prevents the program from
   crashing or hidden issues from arising. This extends to checking the result
   of a kernel launch which can be done using `cudaGetLastError` since the
   kernel launch is asynchronous. The input handling also does not check for
   errors so invalid values for blocks and threads can be passed when executing
   the program.

### Good Qualities

1. The code properly frees device memory that was allocated with `cudaMalloc`.

2. The code follows the standard workflow of allocating device memory, copying
   CPU data to device memory, executing the kernel, and then copying the results
   back from the GPU to the CPU.
