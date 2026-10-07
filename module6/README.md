# Module 6 Assignment

The following directory contains the source code for Module 6.

## Build and Run

The code is intended to be automatically built and run using the assignment
runner. This can be accomplished by executing the following commands:

```bash
git clone https://github.com/mattgermano/EN605.617.git
cd EN605.617
./assignment_build_execution/run_assignments.sh
```

The above command requires Python 3,
[uv](https://docs.astral.sh/uv/getting-started/installation/), CMake, CUDA/NVCC,
and g++. These dependencies can be installed on Ubuntu 26.04 using the following
commands:

```bash
sudo apt update && sudo apt install -y wget build-essential cmake python3 git
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2604/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-13-4
wget -qO- https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Also, ensure that `nvcc` is on your `PATH` via the following command:

```bash
export PATH="/usr/local/cuda/bin:$PATH"
```

The assignment runner executes the [`build.sh`](./build.sh) and
[`run.sh`](./run.sh) scripts to build and run the project with different
configurations.

Alternatively, the code can be manually built using the following commands:

```bash
cd module6
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
