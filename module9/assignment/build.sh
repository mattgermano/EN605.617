#!/usr/bin/env bash

BUILD_TYPE="$1"
if [[ -z "$BUILD_TYPE" ]]; then
    echo "Usage: $0 <build_type>"
    exit 1
fi

if ! command -v cmake &> /dev/null; then
    echo "CMake is required to run build script!"
    exit 1
fi

cmake -S . -B build -D CMAKE_BUILD_TYPE="$1" --fresh
cmake --build build --parallel $(nproc --ignore=1)
