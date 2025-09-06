cmake_minimum_required(VERSION 3.16)
project(yolop_trt CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- OpenCV 찾기 (우선 CMake, 실패 시 pkg-config, 최후에는 /usr/local fallback) ---
find_package(OpenCV QUIET)
if(NOT OpenCV_FOUND)
  find_package(PkgConfig QUIET)
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(OPENCV4 QUIET opencv4)
    if(OPENCV4_FOUND)
      message(STATUS "OpenCV via pkg-config: ${OPENCV4_VERSION}")
      set(OpenCV_INCLUDE_DIRS ${OPENCV4_INCLUDE_DIRS})
      set(OpenCV_LIBS        ${OPENCV4_LIBRARIES})
      link_directories(${OPENCV4_LIBRARY_DIRS})
    endif()
  endif()

  if(NOT OpenCV_FOUND AND NOT OpenCV_INCLUDE_DIRS)
    # 최후수단: 수동 경로 (당신 머신은 /usr/local 기반으로 빌드되어 있었음)
    message(WARNING "OpenCV not found via CMake/pkg-config. Falling back to /usr/local.")
    set(OpenCV_INCLUDE_DIRS /usr/local/include/opencv4)
    link_directories(/usr/local/lib)
    # OpenCV_LIBS는 나중에 링크 실패하면 수동으로 opencv_world 혹은 core/highgui/imgproc 등을 넣어도 됨
  endif()
endif()

# --- CUDA(TensorRT가 include하는 cuda_runtime_api.h의 'crt/host_defines.h') 경로 해결 ---
# 당신 시스템에 존재하는 targets 경로들 우선 탐색
set(CUDA_CANDIDATE_INCLUDES
  /usr/local/cuda-12.1/targets/x86_64-linux/include
  /usr/local/cuda-11.8/targets/x86_64-linux/include
  /usr/local/cuda/targets/x86_64-linux/include
)

find_path(CUDA_TARGET_INC_DIR
  NAMES cuda_runtime_api.h
  PATHS ${CUDA_CANDIDATE_INCLUDES}
  NO_DEFAULT_PATH
)
if(NOT CUDA_TARGET_INC_DIR)
  message(FATAL_ERROR "cuda_runtime_api.h 를 /usr/local/cuda-*/targets/x86_64-linux/include 에서 찾을 수 없습니다.")
endif()

# cudart도 동일 버전으로 맞춤
set(CUDA_CANDIDATE_LIBS
  /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudart.so
  /usr/local/cuda-11.8/targets/x86_64-linux/lib/libcudart.so
  /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so
)
unset(CUDART CACHE)
foreach(p ${CUDA_CANDIDATE_LIBS})
  if(EXISTS "${p}")
    set(CUDART "${p}")
    break()
  endif()
endforeach()
if(NOT CUDART)
  message(FATAL_ERROR "libcudart.so 를 /usr/local/cuda-*/targets/x86_64-linux/lib 에서 찾을 수 없습니다.")
endif()

# --- TensorRT 라이브러리 (시스템 기본 설치 경로) ---
set(NVINFER        /lib/x86_64-linux-gnu/libnvinfer.so)
set(NVINFER_PLUGIN /lib/x86_64-linux-gnu/libnvinfer_plugin.so)

# --- include 경로 추가 ---
include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_TARGET_INC_DIR}           # ← 중요: /targets/x86_64-linux/include
  /usr/include/x86_64-linux-gnu    # NvInfer*.h 등의 기본 경로
)

# --- 소스 타겟 ---
add_executable(dump_trt_bindings dump_trt_bindings.cpp)
target_link_libraries(dump_trt_bindings
  ${OpenCV_LIBS} ${NVINFER} ${NVINFER_PLUGIN} ${CUDART})

add_executable(yolop_trt_min yolop_trt_min.cpp)
target_link_libraries(yolop_trt_min
  ${OpenCV_LIBS} ${NVINFER} ${NVINFER_PLUGIN} ${CUDART})

# 빌드 시 조금 친절한 메시지
message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS         = ${OpenCV_LIBS}")
message(STATUS "CUDA_TARGET_INC_DIR = ${CUDA_TARGET_INC_DIR}")
message(STATUS "CUDART              = ${CUDART}")
message(STATUS "NVINFER             = ${NVINFER}")
message(STATUS "NVINFER_PLUGIN      = ${NVINFER_PLUGIN}")
