cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(synaptic LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_PARALLEL_LEVEL 8)

enable_testing()

add_subdirectory(${CMAKE_SOURCE_DIR}/test)
