cmake_minimum_required(VERSION 3.10)

project(julia_sets LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native")

set(DEFAULT_SOURCES
        src/main.cu src/main.hpp
        src/bitmap_image.hpp)

include(CheckLanguage)
check_language(CUDA)

if (CMAKE_CUDA_COMPILER)
    message("CUDA is supported. Enabling CUDA sources.")
    enable_language(CUDA)
    add_definitions(-DUSE_CUDA)
    set(CMAKE_CUDA_STANDARD 11)
    set(CUDA_SOURCES
            src/kernels.cu src/kernels.hpp)
else ()
    message("Could not find CUDA support. Disabling CUDA sources.")
endif ()

add_executable(${PROJECT_NAME} ${DEFAULT_SOURCES} ${CUDA_SOURCES})
