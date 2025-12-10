cmake_minimum_required(VERSION 2.8.3)
project(map_generator)

set(CMAKE_BUILD_TYPE "Release")
#set(CMAKE_CXX_FLAGS "-std=c++11")
ADD_COMPILE_OPTIONS(-std=c++11 )
ADD_COMPILE_OPTIONS(-std=c++14 )
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -B/usr/lib/gold-ld/")

#set(PCL_DIR "/usr/include/pcl-1.10/pcl/")
set(PCL_DIR "/home/quant/pcl-1.14/")
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  pcl_conversions
)
find_package(PCL 1.10 REQUIRED)
find_package(Eigen3 REQUIRED)

catkin_package()

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

add_executable (dynamic_env src/dynamic_env.cpp )
target_link_libraries(dynamic_env
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES})