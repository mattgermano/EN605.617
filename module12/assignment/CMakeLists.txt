cmake_minimum_required(VERSION 3.24)

project(
    module12
    LANGUAGES CXX
    VERSION 1.0.0
)

find_package(OpenCL REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(assignment.exe assignment.cpp)
target_link_libraries(assignment.exe PRIVATE OpenCL::OpenCL)
