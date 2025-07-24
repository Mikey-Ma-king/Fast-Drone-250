cmake_minimum_required(VERSION 3.0.2)
project(planning)

add_compile_options(-std=c++14)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")

find_package(Eigen3 REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  quadrotor_msgs
  roscpp
)

catkin_package(
  INCLUDE_DIRS include ${EIGEN3_INCLUDE_DIR}
  CATKIN_DEPENDS quadrotor_msgs roscpp
  LIBRARIES ${PROJECT_NAME}
  DEPENDS EIGEN3
)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
  ${DECOMP_UTIL_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${quadrotor_msgs_INCLUDE_DIRS}

)

add_dependencies(${catkin_EXPORTED_TARGETS}
)



add_executable(planning
  src/traj_server.cpp
  src/callback.cpp
  )

target_link_libraries(planning
  ${catkin_LIBRARIES}
)