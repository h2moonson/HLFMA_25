cmake_minimum_required(VERSION 3.0.2)
project(decision)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  roscpp
  rospy
  sensor_msgs
  morai_msgs
  std_msgs
  message_generation
)

add_service_files(
  FILES
  InitializePose.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(
  PROGRAMS
    src/pure_pursuit.py
    src/mode_decider.py
    src/mode_decider_Morai.py
    src/pure_pursuit_Henes.py
    src/pure_pursuit_Morai.py
    src/path_maker.py
    src/morai_maker.py
    src/wgs84_to_utmk.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
