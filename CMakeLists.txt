cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(synaptic)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/include/)
include_directories(${CMAKE_SOURCE_DIR}/src/)
include_directories(${CMAKE_SOURCE_DIR}/test/)

file(GLOB SOURCES "src/*.cpp")
file(GLOB test1 "test/unit_tests/add_op_tests.cpp")
file(GLOB MAIN "src/engine.cpp")

enable_testing()

add_executable(executable ${MAIN})
add_executable(test1 ${test1})

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/releases/download/v1.15.2/googletest-1.15.2.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE  # Fix warning by specifying timestamp behavior
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Link gtest to your test executable
target_link_libraries(test1 gtest gtest_main)

# Register the test
add_test(NAME basic_add_op_test COMMAND test1)
