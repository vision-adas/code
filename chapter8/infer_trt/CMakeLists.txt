cmake_minimum_required(VERSION 3.9)
set(CMAKE_CXX_STANDARD 17)
project(inference_in_trt)

set(CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/")
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

find_package( OpenCV REQUIRED )
find_package(CUDA QUIET REQUIRED)
list(APPEND PLUGINS "nvinfer")
list(APPEND PLUGINS "nvonnxparser")
list(APPEND PLUGINS "nvparsers")
list(APPEND PLUGINS "nvinfer_plugin")

foreach(libName ${PLUGINS})
    find_library(${libName}_lib NAMES ${libName} "/usr" PATH_SUFFIXES lib)
    list(APPEND PLUGIN_LIBS "${${libName}_lib}")
endforeach()

include_directories(${CUDA_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} common)

add_executable(trt_inference trt_inference.cpp)

target_link_libraries(trt_inference ${CUDA_LIBRARIES} nvonnxparser nvinfer nvparsers nvinfer_plugin ${OpenCV_LIBS})