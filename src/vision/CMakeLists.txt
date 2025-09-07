cmake_minimum_required(VERSION 3.0.2)
project(vision)

## 빌드 타입이 설정되지 않았다면 Release로 설정
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

## C++11 표준 사용 (C++11을 기본으로 함)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_definitions(-std=c++11)

## Catkin 패키지 설정 (ROS 관련)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  cv_bridge
  roslib
  image_transport
  dynamic_reconfigure
)

generate_dynamic_reconfigure_options(
  cfg/cam_param.cfg
)

catkin_package(
  # INCLUDE_DIRS include
  # LIBRARIES vision
  CATKIN_DEPENDS roscpp rospy cv_bridge roslib image_transport dynamic_reconfigure
  # DEPENDS system_lib
)

## 기본 include 디렉토리 (ROS 및 로컬 include)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
)

## CUDA 설정
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

## OpenCV 설정
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

## TensorRT 설정
include_directories(/opt/TensorRT-8.5.3.1/include)
link_directories(/opt/TensorRT-8.5.3.1/lib)

## 추가 컴파일 플래그 (필요에 따라 수정)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")
add_definitions(-O3 -pthread)

## CUDA로 플러그인 라이브러리 생성 (예: yololayer.cu)
# cuda_add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/src/tensorrtx/yolop/yololayer.cu)
# target_link_libraries(myplugins nvinfer cudart)

## 메인 실행 파일 생성 (yolop_node.cpp, yolop.cpp, common.cpp)
# add_executable(yolop_node 
#   ${PROJECT_SOURCE_DIR}/src/tensorrtx/yolop/yolop_node.cpp 
#   ${PROJECT_SOURCE_DIR}/src/tensorrtx/yolop/yolop.cpp 
#   ${PROJECT_SOURCE_DIR}/src/tensorrtx/yolop/common.cpp
# )

# ## 메인 실행 파일에 필요한 라이브러리 링크 (TensorRT, CUDA, OpenCV, catkin 라이브러리 포함)
# target_link_libraries(yolop_node 
#   nvinfer 
#   cudart 
#   myplugins 
#   ${OpenCV_LIBS}
#   ${catkin_LIBRARIES}
# )
