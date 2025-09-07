cmake_minimum_required(VERSION 3.10)
project(yolop_trt_min CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# OpenCV
find_package(OpenCV REQUIRED)

# CUDA (레거시 FindCUDA — CMake 3.10~3.16 호환)
find_package(CUDA REQUIRED)

# TensorRT (헤더/라이브러리 경로 자동 탐색 + 필요시 수동 지정 가능)
# 필요하면 configure 시 -DTENSORRT_INCLUDE_DIR=... -DTENSORRT_LIB_DIR=... 넘겨주세요.
find_path(TENSORRT_INCLUDE_DIR
  NAMES NvInfer.h
  PATHS
    ${TENSORRT_INCLUDE_DIR}
    /usr/include /usr/include/x86_64-linux-gnu /usr/local/include
  PATH_SUFFIXES tensorrt
)

# 라이브러리 찾기
find_library(NVINFER_LIB NAMES nvinfer
  HINTS ${TENSORRT_LIB_DIR} /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib
)
find_library(NVINFER_PLUGIN_LIB NAMES nvinfer_plugin
  HINTS ${TENSORRT_LIB_DIR} /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib
)

# cudart 찾기 (FindCUDA는 ${CUDA_TOOLKIT_ROOT_DIR}를 제공합니다)
# 경로가 다르면 아래 PATHS에 추가하세요.
find_library(CUDA_CUDART_LIBRARY NAMES cudart
  PATHS
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    /usr/local/cuda/lib64
    /usr/lib/x86_64-linux-gnu
    /usr/lib
)

if (NOT TENSORRT_INCLUDE_DIR OR NOT NVINFER_LIB OR NOT NVINFER_PLUGIN_LIB)
  message(FATAL_ERROR "TensorRT headers/libs not found. Hint: -DTENSORRT_INCLUDE_DIR=... -DTENSORRT_LIB_DIR=...")
endif()
if (NOT CUDA_CUDART_LIBRARY)
  message(FATAL_ERROR "cudart not found. Ensure CUDA is installed. Hint: export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda")
endif()

add_executable(yolop_trt_min yolop_trt_min.cpp yolop_trt yolop_trt.cpp)

target_include_directories(yolop_trt_min PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}        # cuda_runtime_api.h, cuda_fp16.h 등
  ${TENSORRT_INCLUDE_DIR}
)

target_link_libraries(yolop_trt_min PRIVATE
  ${OpenCV_LIBS}
  ${NVINFER_LIB}
  ${NVINFER_PLUGIN_LIB}
  ${CUDA_CUDART_LIBRARY}
)

# 최적화 옵션(선택)
target_compile_definitions(yolop_trt_min PRIVATE -D_FORCE_INLINES)
