cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(synaptic)

set(CXX_STANDARD 17)  # Update to a higher standard if needed
set(CMAKE_CXX_COMPILER g++)  # Assuming your compiler is g++

#message("CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")

include_directories(${CMAKE_SOURCE_DIR}/include/)
include_directories(${CMAKE_SOURCE_DIR}/src/)
include_directories(${CMAKE_SOURCE_DIR}/test/)
file(GLOB SOURCES "src/*.cpp")
file(GLOB test1 "test/unit_tests/add_op_tests.cpp")
file(GLOB MAIN "src/engine.cpp")
enable_testing()
add_test (NAME basic_add_op_test COMMAND test1)

add_executable(executable ${MAIN})
add_executable(test1 ${test1})
#add_executable(tensor src/tensor.cpp)

