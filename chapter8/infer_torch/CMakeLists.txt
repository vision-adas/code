cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(torch_inference)

set(USE_GPU "OFF")

if(USE_GPU STREQUAL "ON")
    list(APPEND CMAKE_PREFIX_PATH "./libtorch_gpu/libtorch")
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
else()
    list(APPEND CMAKE_PREFIX_PATH "./libtorch")
endif()

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} )

if(USE_GPU STREQUAL "ON")
    add_executable(torch_inference torch_inference_gpu.cpp)
else()
    add_executable(torch_inference torch_inference.cpp)
endif()

target_link_libraries(torch_inference "${TORCH_LIBRARIES}"  ${OpenCV_LIBS})
set_property(TARGET torch_inference PROPERTY CXX_STANDARD 14)