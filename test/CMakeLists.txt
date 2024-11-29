

set(DEBUG 0)
set(INFO 1)
set(WARNING 2)
set(ERROR 3)
set(CRITICAL 4)
set(LOG_LEVEL ${INFO})  # INFO level by default
set(CMAKE_BUILD_TYPE Debug)
option(LOG_LEVEL_DEBUG "Show warning + error + info + debug messages" ON)
option(LOG_LEVEL_INFO "Show warning + error + info messages" OFF)
option(LOG_LEVEL_WARNING "Show warrning + error messages" OFF)
option(LOG_LEVEL_ERROR "Show error messages" OFF)

if(LOG_LEVEL_DEBUG)
    set(LOG_LEVEL ${DEBUG})
    message(STATUS "Log level: DEBUG")
elseif(LOG_LEVEL_INFO)
    set(LOG_LEVEL ${INFO})
    message(STATUS "Log level: INFO")
elseif(LOG_LEVEL_WARNING)
    set(LOG_LEVEL ${WARNING})
    message(STATUS "Log level: WARNING")
elseif(LOG_LEVEL_ERROR)
    set(LOG_LEVEL ${ERROR})
    message(STATUS "Log level: ERROR")
endif()

if(_WIN32)
    set(LOG_FILE "D:\a\synaptic\synaptic/build/logs.log")
else()
    set(LOG_FILE "${CMAKE_SOURCE_DIR}/logs.log")
endif()
message(STATUS "Log file: ${LOG_FILE}")


include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/logging)
include_directories(${CMAKE_SOURCE_DIR}/include/connections)
include_directories(${CMAKE_SOURCE_DIR}/include/loss_fn)
include_directories(${CMAKE_SOURCE_DIR}/include/layers)
include_directories(${CMAKE_SOURCE_DIR}/include/rng_for_tensor)
include_directories(${CMAKE_SOURCE_DIR}/include/operands)


# GoogleTest integration
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/releases/download/v1.15.2/googletest-1.15.2.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE  
)


enable_testing()

# Set GoogleTest configuration
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)


file(GLOB unit_test_sources "${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/*.cpp")
add_executable(unit_tests "${unit_test_sources}")


target_sources(unit_tests PRIVATE ${CMAKE_SOURCE_DIR}/src/logging/logging.cpp)
add_compile_definitions(unit_tests PRIVATE LOG_LEVEL=${LOG_LEVEL})
add_compile_definitions(unit_tests PRIVATE LOG_FILE="${LOG_FILE}")

file(GLOB integration_test_sources "${CMAKE_CURRENT_SOURCE_DIR}/integration_tests/*.cpp")
add_executable(integration_tests "${integration_test_sources}")

target_sources(integration_tests PRIVATE ${CMAKE_SOURCE_DIR}/src/logging/logging.cpp)
add_compile_definitions(integration_tests PRIVATE LOG_LEVEL=${LOG_LEVEL})
add_compile_definitions(integration_tests PRIVATE LOG_FILE="${LOG_FILE}")

# Link GoogleTest with the test executable
target_link_libraries(unit_tests gtest gtest_main)
target_link_libraries(integration_tests gtest gtest_main)

# Register the test
add_test(NAME unit_tests COMMAND unit_tests)
add_test(NAME integration_tests COMMAND integration_tests)
