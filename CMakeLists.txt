cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(synaptic)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_PARALLEL_LEVEL 8)
# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include/)
include_directories(${CMAKE_SOURCE_DIR}/src/)
include_directories(${CMAKE_SOURCE_DIR}/test/)

# Source files
file(GLOB SOURCES "src/*.cpp")
file(GLOB test_sources "test/unit_tests/add_op_tests.cpp")
file(GLOB MAIN "src/engine.cpp")

# Enable testing
enable_testing()

# Executables
add_executable(executable ${MAIN})
add_executable(test1 ${test_sources})

# GoogleTest integration
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/releases/download/v1.15.2/googletest-1.15.2.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE  # Fix warning by specifying timestamp behavior
)

# Set GoogleTest configuration
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Link gtest and gtest_main libraries to test1
target_link_libraries(test1 gtest gtest_main)

# Register the test
add_test(NAME basic_add_op_test COMMAND test1)
