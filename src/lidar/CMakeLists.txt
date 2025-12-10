cmake_minimum_required(VERSION 3.0.2)
project(lidar)

## Compile as C++11, supported in ROS Kinetic and newer
ADD_COMPILE_OPTIONS(-std=c++14 )
ADD_COMPILE_OPTIONS(-std=c++14 )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -B/usr/lib/gold-ld/")
set(CMAKE_BUILD_TYPE "Release")
## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages

find_package(PCL 1.10 REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  pcl_conversions
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf
)
catkin_package(
   LIBRARIES lidar
   CATKIN_DEPENDS geometry_msgs pcl_conversions roscpp sensor_msgs std_msgs
#  DEPENDS system_lib
)
add_definitions(${PCL_DEFINATIONS})

add_executable(raycast src/raycast.cpp)

target_link_libraries (
  raycast 
  ${catkin_LIBRARIES} 
  ${PCL_LIBRARIES}
  )

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)


