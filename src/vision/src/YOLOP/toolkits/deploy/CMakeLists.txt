set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

cmake_minimum_required(VERSION 3.10)
project(yolop_trt)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

# CUDA / TensorRT / OpenCV 경로 (필요 시 수정)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# TensorRT 경로 수동 지정 필요할 수 있음
# set(TENSORRT_ROOT /usr/lib/x86_64-linux-gnu)  # 예시 (libnvinfer.so 등이 존재)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h PATHS /usr/include /usr/include/x86_64-linux-gnu /usr/local/include)
find_library(NVINFER     nvinfer     HINTS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)
find_library(NVPARSERS   nvparsers   HINTS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)
find_library(NVINFER_PLUGIN nvinfer_plugin HINTS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)
find_library(CUDART      cudart      HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 /usr/local/cuda/lib64)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
)

add_executable(yolop_trt yolop_trt.cpp yolop_trt.hpp)
target_link_libraries(yolop_trt
  ${OpenCV_LIBS}
  ${NVINFER} ${NVPARSERS} ${NVINFER_PLUGIN} ${CUDART}
)


find_package(catkin REQUIRED COMPONENTS roscpp sensor_msgs cv_bridge)
include_directories(${catkin_INCLUDE_DIRS})

add_executable(yolop_trt_node ../yolop_trt_node.cpp yolop_trt.cpp yolop_trt.hpp)
target_link_libraries(yolop_trt_node ${OpenCV_LIBS} ${NVINFER} ${NVPARSERS} ${NVINFER_PLUGIN} ${CUDART} ${catkin_LIBRARIES})
