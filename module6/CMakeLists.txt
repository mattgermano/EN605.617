cmake_minimum_required(VERSION 3.24)

project(
    module6
    LANGUAGES CUDA
    VERSION 1.0.0
)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

add_executable(assignment.exe assignment.cu)
target_link_libraries(assignment.exe PRIVATE CUDA::cufft)
target_include_directories(assignment.exe PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../common/inc)
set_property(TARGET assignment.exe PROPERTY CUDA_ARCHITECTURES native)

# For a debug build, also enable debug symbols for device code (-G)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(assignment.exe PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-G -fdevice-sanitize=memcheck>
    )
endif()

add_custom_target(sanitizers
    COMMAND compute-sanitizer --tool memcheck --error-exitcode 1 $<TARGET_FILE:assignment.exe>
    COMMAND compute-sanitizer --tool initcheck --error-exitcode 1 $<TARGET_FILE:assignment.exe>
    COMMAND compute-sanitizer --tool synccheck --error-exitcode 1 $<TARGET_FILE:assignment.exe>
    COMMAND compute-sanitizer --tool racecheck --error-exitcode 1 $<TARGET_FILE:assignment.exe>
    DEPENDS assignment.exe
    COMMENT "Running compute-sanitizer"
)
