cmake_minimum_required(VERSION 3.10)
project(yolop_trt_ros)
set(CMAKE_CXX_STANDARD 17)

find_package(catkin REQUIRED COMPONENTS roscpp sensor_msgs cv_bridge image_transport)
find_package(OpenCV REQUIRED)

# 방식 A) 코어를 시스템에 설치했을 때:
find_path(YOLOP_TRT_INCLUDE yolop_trt/yolop_trt.hpp PATHS /usr/local/include)
find_library(YOLOP_TRT_LIB yolop_trt_core PATHS /usr/local/lib)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${YOLOP_TRT_INCLUDE})
add_executable(yolop_trt_node src/yolop_trt_node.cpp)
target_link_libraries(yolop_trt_node ${catkin_LIBRARIES} ${OpenCV_LIBS} ${YOLOP_TRT_LIB})

# 방식 B) 코어를 서브모듈/벤더링:
# add_subdirectory(../yolop_trt_core ${CMAKE_BINARY_DIR}/yolop_trt_core_build)
# target_link_libraries(yolop_trt_node yolop_trt_core ...)
