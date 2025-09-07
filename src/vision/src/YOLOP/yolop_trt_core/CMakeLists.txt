cmake_minimum_required(VERSION 3.10)
project(yolop_trt_core CXX)

# --- 맨 위에 추가 (project() 바로 아래 권장) ---
set(YOLOP_CUDA_VERSION "11.8" CACHE STRING "Force CUDA version")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-${YOLOP_CUDA_VERSION}" CACHE PATH "CUDA root" FORCE)
message(STATUS "[YOLOP] Forced CUDA: ${CUDA_TOOLKIT_ROOT_DIR}")


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
list(APPEND CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# ── 아키텍처 감지
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|AARCH64|arm64")
  set(_CUDA_ARCH_DIR aarch64-linux)
else()
  set(_CUDA_ARCH_DIR x86_64-linux)
endif()

# ── CUDA include 후보 경로
set(_CUDA_INC_CANDIDATES
  ${CUDA_TOOLKIT_ROOT_DIR}/include
  ${CUDA_TOOLKIT_ROOT_DIR}/targets/${_CUDA_ARCH_DIR}/include
  /usr/local/cuda/include
  /usr/local/cuda/targets/${_CUDA_ARCH_DIR}/include
)

# 버전 디렉토리도 추가 스캔 (예: /usr/local/cuda-11.8, /usr/local/cuda-12.1 ...)
file(GLOB _CUDA_VERSION_ROOTS /usr/local/cuda-*)
foreach(_v IN LISTS _CUDA_VERSION_ROOTS)
  list(APPEND _CUDA_INC_CANDIDATES
    ${_v}/include
    ${_v}/targets/${_CUDA_ARCH_DIR}/include
  )
endforeach()

# host_defines.h가 있는 CRT 경로를 우선 반영
find_path(CUDA_CRT_INCLUDE_DIR
  NAMES crt/host_defines.h
  PATHS
    ${CUDA_TOOLKIT_ROOT_DIR}/targets/${_CUDA_ARCH_DIR}/include
    /usr/local/cuda/targets/${_CUDA_ARCH_DIR}/include
    /usr/local/cuda*/targets/${_CUDA_ARCH_DIR}/include
)
if (CUDA_CRT_INCLUDE_DIR)
  list(INSERT _CUDA_INC_CANDIDATES 0 ${CUDA_CRT_INCLUDE_DIR})
endif()

# 실제 존재하는 경로만 사용
set(_CUDA_INC_USE "")
foreach(_p IN LISTS _CUDA_INC_CANDIDATES)
  if (EXISTS "${_p}")
    list(APPEND _CUDA_INC_USE "${_p}")
  endif()
endforeach()
if (NOT _CUDA_INC_USE)
  message(FATAL_ERROR "No valid CUDA include path found. Checked: ${_CUDA_INC_CANDIDATES}")
endif()
message(STATUS "[YOLOP] Using CUDA include dirs: ${_CUDA_INC_USE}")

# ── TensorRT
find_path(TENSORRT_INCLUDE_DIR
  NvInfer.h
  HINTS ${TENSORRT_INCLUDE_DIR}
        /usr/include /usr/include/x86_64-linux-gnu /usr/local/include
)
find_library(NVINFER
  nvinfer
  HINTS ${TENSORRT_LIB_DIR} /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib
)
find_library(NVINFER_PLUGIN
  nvinfer_plugin
  HINTS ${TENSORRT_LIB_DIR} /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib
)
find_library(CUDART
  cudart
  HINTS
    ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    ${CUDA_TOOLKIT_ROOT_DIR}/targets/${_CUDA_ARCH_DIR}/lib
    ${CUDA_TOOLKIT_ROOT_DIR}/targets/${_CUDA_ARCH_DIR}/lib64
    /usr/local/cuda/lib64
    /usr/local/cuda/targets/${_CUDA_ARCH_DIR}/lib
    /usr/local/cuda/targets/${_CUDA_ARCH_DIR}/lib64
    /usr/lib/x86_64-linux-gnu
)
if (NOT TENSORRT_INCLUDE_DIR OR NOT NVINFER OR NOT NVINFER_PLUGIN OR NOT CUDART)
  message(FATAL_ERROR "TensorRT/CUDA not found. Hint: -DTENSORRT_INCLUDE_DIR=... -DTENSORRT_LIB_DIR=... -DCUDA_TOOLKIT_ROOT_DIR=...")
endif()
message(STATUS "[YOLOP] TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "[YOLOP] NVINFER: ${NVINFER}")
message(STATUS "[YOLOP] NVINFER_PLUGIN: ${NVINFER_PLUGIN}")
message(STATUS "[YOLOP] CUDART: ${CUDART}")

# ── 코어 라이브러리
add_library(yolop_trt_core SHARED
  src/yolop_trt.cpp
)

# add_library 이후에 딱 한 번만 include 디렉토리 지정!
target_include_directories(yolop_trt_core
  PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${_CUDA_INC_USE}
)
target_compile_definitions(yolop_trt_core PRIVATE -D_FORCE_INLINES)
target_link_libraries(yolop_trt_core
  ${OpenCV_LIBS}
  ${NVINFER} ${NVINFER_PLUGIN} ${CUDART}
)

# ── CLI 데모(선택)
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/yolop_trt_min.cpp")
  add_executable(yolop_trt_min tools/yolop_trt_min.cpp)
elseif (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/yolop_trt_min.cpp")
  add_executable(yolop_trt_min yolop_trt_min.cpp)
endif()

if (TARGET yolop_trt_min)
  target_include_directories(yolop_trt_min
    PRIVATE
      ${OpenCV_INCLUDE_DIRS}
      ${TENSORRT_INCLUDE_DIR}
      ${_CUDA_INC_USE}
      ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(yolop_trt_min
    yolop_trt_core
    ${OpenCV_LIBS}
    ${NVINFER} ${NVINFER_PLUGIN} ${CUDART}
  )
endif()

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
