# CMake minimum version check
cmake_minimum_required(VERSION 3.18)

#C++ standard required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# (Windows) Use folders
IF (WIN32)
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
ENDIF()

# Global variables
set(ProjectName "CUDAFDTD")
set(ExeName     "CUDAFDTD")

# Project name
project(${ProjectName} LANGUAGES CXX CUDA)

# Include source code
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src)
