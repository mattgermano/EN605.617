cmake_minimum_required(VERSION 3.23)

project(
    module3
    LANGUAGES CUDA
    VERSION 1.0.0
)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

add_executable(assignment.exe assignment.cu)

# For a debug build, also enable debug symbols for device code (-G)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(assignment.exe PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-G -fdevice-sanitize=memcheck>
    )
endif()
