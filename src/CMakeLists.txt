cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(synaptic)

set(CXX_STANDARD 17)  # Update to a higher standard if needed
set(CMAKE_CXX_COMPILER g++)  # Assuming your compiler is g++


include_directories(HEADER_FILES)
file(GLOB MAIN
     "engine.cpp"
)
add_executable(executable ${MAIN})
