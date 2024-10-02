cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cublas_gemm LANGUAGES C CXX CUDA)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD
    20
    CACHE STRING "The C++ standard whoese features are requested." FORCE)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD
    20
    CACHE STRING "The CUDA standard whose features are requested." FORCE)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set host compiler flags. Enable all warnings and treat them as errors
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall")

find_package(CUDAToolkit QUIET REQUIRED)
enable_language(CUDA)
set(CMAKE_CUDA on)

include_directories("${PROJECT_SOURCE_DIR}/../../3rd-party/TiledCUDA/include")

add_executable(cublas_gemm main.cu)
target_link_libraries(cublas_gemm ${CUDA_CUBLAS_LIBRARIES})
