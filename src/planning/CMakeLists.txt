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

add_dependencies(${catkin_EXPORTED_TARGETS})

add_executable(planning
  src/traj_server.cpp
  src/callback.cpp
  )

target_link_libraries(planning
  ${catkin_LIBRARIES}
)

# 1. Cascade KF + R/t (Ours)
add_executable(dog_pos_processor_sim
  src/dog_pos_processor_sim.cpp
  )
target_link_libraries(dog_pos_processor_sim
  ${catkin_LIBRARIES}
)

# 2. Whole joint EKF + R/t (baseline)
add_executable(dog_pos_processor_kf
  src/dog_pos_processor_kf.cpp
  )
target_link_libraries(dog_pos_processor_kf
  ${catkin_LIBRARIES}
)

# 3. Single LKF + raw increment (baseline)
add_executable(dog_pos_processor_lkf
  src/dog_pos_processor_lkf_base.cpp
  )
target_link_libraries(dog_pos_processor_lkf
  ${catkin_LIBRARIES}
)

# 4. Vision-only, fixed R/t (baseline)
add_executable(dog_pos_processor_vis
  src/dog_pos_processor_vis_base.cpp
  )
target_link_libraries(dog_pos_processor_vis
  ${catkin_LIBRARIES}
)

# legacy（与 sim 同源时可不用）
add_executable(dog_pos_processor
  src/dog_pos_processor.cpp
  )
target_link_libraries(dog_pos_processor
  ${catkin_LIBRARIES}
)

# 降落/仿真实验用 traj_server（放宽 mode 切换，无 1->0 / 2->1 回退）
add_executable(planning_sim
  src/traj_server_sim.cpp
  src/callback_sim.cpp
  )
target_link_libraries(planning_sim
  ${catkin_LIBRARIES}
)
