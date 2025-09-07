cmake_minimum_required(VERSION 3.10)
project(yolop_trt_ros)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

find_package(catkin REQUIRED COMPONENTS
  roscpp sensor_msgs cv_bridge image_transport
)
find_package(OpenCV REQUIRED)

# ── 모드 선택: 설치된 코어 찾기(기본) vs 로컬 벤더링 ──
option(YOLOP_TRT_VENDOR_CORE "Build with vendored yolop_trt_core" OFF)

if (NOT YOLOP_TRT_VENDOR_CORE)
  # ── 모드 A: 설치된 코어 찾기 ──
  find_path(YOLOP_TRT_INCLUDE
    NAMES yolop_trt/yolop_trt.hpp
    PATHS /usr/local/include /usr/include
  )
  find_library(YOLOP_TRT_LIB
    NAMES yolop_trt_core
    PATHS /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu
  )
  if (NOT YOLOP_TRT_INCLUDE OR NOT YOLOP_TRT_LIB)
    message(FATAL_ERROR "yolop_trt_core not found. Either `sudo make install` the core or set -DYOLOP_TRT_VENDOR_CORE=ON.")
  endif()
  set(YOLOP_TRT_CORE_TARGET "${YOLOP_TRT_LIB}")
else()
  # ── 모드 B: 로컬 벤더링 ──
  # 경로를 환경변수 또는 상대경로로 지정
  # 기본값: ../yolop_trt_core
  set(YOLOP_TRT_CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../yolop_trt_core" CACHE PATH "Path to yolop_trt_core root")
  if (NOT EXISTS "${YOLOP_TRT_CORE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "Invalid YOLOP_TRT_CORE_DIR: ${YOLOP_TRT_CORE_DIR}")
  endif()
  add_subdirectory(${YOLOP_TRT_CORE_DIR} ${CMAKE_BINARY_DIR}/yolop_trt_core_build)
  # 위 add_subdirectory로 타겟 yolop_trt_core 생성됨
  set(YOLOP_TRT_INCLUDE "${YOLOP_TRT_CORE_DIR}/include")
  set(YOLOP_TRT_CORE_TARGET yolop_trt_core)
endif()

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${YOLOP_TRT_INCLUDE}
)

add_executable(yolop_trt_node
  src/yolop_trt_node.cpp
)

# 모드 A: ${YOLOP_TRT_CORE_TARGET}는 라이브러리 경로
# 모드 B: ${YOLOP_TRT_CORE_TARGET}는 타겟명(yolop_trt_core)
target_link_libraries(yolop_trt_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${YOLOP_TRT_CORE_TARGET}
)

install(TARGETS yolop_trt_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
