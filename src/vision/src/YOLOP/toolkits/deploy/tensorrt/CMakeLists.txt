cmake_minimum_required(VERSION 3.10)
project(yolop_trt_tools CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED)

# TensorRT 경로 탐색(HINTS에 당신 시스템 경로들 추가)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  PATHS /usr/include /usr/local/include /usr/include/x86_64-linux-gnu /usr/src/tensorrt/include)

find_library(NVINFER nvinfer
  PATHS /lib/x86_64-linux-gnu /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu /usr/src/tensorrt/lib)
find_library(NVINFER_PLUGIN nvinfer_plugin
  PATHS /lib/x86_64-linux-gnu /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu /usr/src/tensorrt/lib)

# CUDA 런타임
find_library(CUDART cudart
  PATHS /usr/local/cuda/targets/x86_64-linux/lib /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib)

if(NOT TENSORRT_INCLUDE_DIR OR NOT NVINFER OR NOT NVINFER_PLUGIN OR NOT CUDART)
  message(FATAL_ERROR "TensorRT/OpenCV/CUDA libs not found. Set hints or pass -D vars.")
endif()

include_directories(${OpenCV_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIR})

# CUDA headers (cuda_runtime_api.h) 경로 탐색
find_path(CUDA_INCLUDE_DIR cuda_runtime_api.h
  PATHS /usr/local/cuda/include /usr/local/cuda-12/include /usr/local/cuda-11/include /usr/include)
if(NOT CUDA_INCLUDE_DIR)
  message(FATAL_ERROR "Could not find cuda_runtime_api.h. Set CUDA_INCLUDE_DIR.")
endif()
include_directories(${CUDA_INCLUDE_DIR})

add_executable(dump_trt_bindings dump_trt_bindings.cpp)
target_link_libraries(dump_trt_bindings
  ${OpenCV_LIBS} ${NVINFER} ${NVINFER_PLUGIN} ${CUDART})

add_executable(yolop_trt_min yolop_trt_min.cpp)
target_link_libraries(yolop_trt_min
  ${OpenCV_LIBS} ${NVINFER} ${NVINFER_PLUGIN} ${CUDART})
