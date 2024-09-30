cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(synaptic LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_PARALLEL_LEVEL 8)


set(DEBUG 0)
set(INFO 1)
set(WARNING 2)
set(ERROR 3)
set(CRITICAL 4)
set(SYNAPTIC_LOG_LEVEL ${INFO})  # INFO level by default

# Set the log file path
if(MSVC)
  set(SYNAPTIC_LOG_FILE "..\\logs.log")  # No need to escape quotes in CMake
else()
  set(SYNAPTIC_LOG_FILE "../logs.log")
endif()

# Add definitions to pass to the compiler
add_compile_definitions(SYNAPTIC_LOG_LEVEL=${SYNAPTIC_LOG_LEVEL})
add_compile_definitions(SYNAPTIC_LOG_FILE=\"${SYNAPTIC_LOG_FILE}\")


# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/src/logging/)
include_directories(${CMAKE_SOURCE_DIR}/test)

# Source files for the library
file(GLOB SOURCES "test/unit_tests/test_runner.cpp")
# Define the shared library
#add_library(synaptic SHARED ${SOURCES})

# Optionally, install the shared library
#install(TARGETS synaptic DESTINATION lib)

# Enable testing
enable_testing()

# Source files for tests
file(GLOB test_sources "test/unit_tests/*.cpp")

# Create the test executable
add_executable(tests ${test_sources})

# Link the synaptic library to the test executable

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

# Link GoogleTest with the test executable
target_link_libraries(tests gtest gtest_main)

# Register the test
add_test(NAME run_all_tests COMMAND tests)
