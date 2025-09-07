cmake_minimum_required(VERSION 3.10)
project(yolop_trt_core CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# OpenCV
find_package(OpenCV REQUIRED)

# CUDA (legacy FindCUDA for CMake 3.10~3.16)
find_package(CUDA REQUIRED)

# TensorRT: 필요시 -DTENSORRT_INCLUDE_DIR / -DTENSORRT_LIB_DIR 전달
find_path(TENSORRT_INCLUDE_DIR
  NAMES NvInfer.h
  PATHS ${TENSORRT_INCLUDE_DIR}
        /usr/include /usr/include/x86_64-linux-gnu /usr/local/include
  PATH_SUFFIXES tensorrt
)
find_library(NVINFER           nvinfer         HINTS ${TENSORRT_LIB_DIR} /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)
find_library(NVINFER_PLUGIN    nvinfer_plugin  HINTS ${TENSORRT_LIB_DIR} /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)
find_library(CUDART            cudart          HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu)

if (NOT TENSORRT_INCLUDE_DIR OR NOT NVINFER OR NOT NVINFER_PLUGIN OR NOT CUDART)
  message(FATAL_ERROR "TensorRT/CUDA not found. Hint: -DTENSORRT_INCLUDE_DIR=... -DTENSORRT_LIB_DIR=... -DCUDA_TOOLKIT_ROOT_DIR=...")
endif()

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ── 코어 라이브러리 ──
add_library(yolop_trt_core SHARED
  src/yolop_trt.cpp
)
target_link_libraries(yolop_trt_core
  ${OpenCV_LIBS}
  ${NVINFER} ${NVINFER_PLUGIN} ${CUDART}
)
target_compile_definitions(yolop_trt_core PRIVATE -D_FORCE_INLINES)

# ── (선택) CLI 데모 ──
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/yolop_trt_min.cpp")
  add_executable(yolop_trt_min tools/yolop_trt_min.cpp)
  target_link_libraries(yolop_trt_min
    yolop_trt_core
    ${OpenCV_LIBS}
    ${NVINFER} ${NVINFER_PLUGIN} ${CUDART}
  )
endif()

# ── 설치 ──
include(GNUInstallDirs)
install(TARGETS yolop_trt_core
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if (TARGET yolop_trt_min)
  install(TARGETS yolop_trt_min
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
