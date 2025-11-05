cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(projet_vision VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Find required packages
find_package(OpenCV REQUIRED)

# MediaPipe - adjust path as needed
# set(MEDIAPIPE_DIR "/path/to/mediapipe" CACHE PATH "Path to MediaPipe installation")
# include_directories(${MEDIAPIPE_DIR})

# Eigen for linear algebra (numpy replacement)
find_package(Eigen3 3.3 QUIET)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found, will use bundled version or OpenCV for linear algebra")
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

if(Eigen3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()

# Compiler warnings
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Source subdirectory
add_subdirectory(src)

