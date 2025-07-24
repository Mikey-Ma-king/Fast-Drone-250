cmake_minimum_required(VERSION 3.0.2)
project(read)

add_compile_options(-std=c++14)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")
#set(target_prediction_DIR "/")

find_package(Eigen3 REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  nodelet
  quadrotor_msgs
  
)

find_package(OpenCV 3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem thread)
find_package(cv_bridge REQUIRED)

catkin_package(
  INCLUDE_DIRS include ${EIGEN3_INCLUDE_DIR}
  CATKIN_DEPENDS  roscpp nodelet
  LIBRARIES ${PROJECT_NAME}
  DEPENDS EIGEN3
)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIR}
  ${DECOMP_UTIL_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${cv_bridge_INCLUDE_DIRS}
)
include_directories(/home/pc/Fast-Perching/src/utils/target_prediction/include/bezier_prediction)


# planning node
add_library(${PROJECT_NAME}
src/${PROJECT_NAME}.cpp
/home/pc/Fast-Perching/src/utils/target_prediction/src/bezier_predict.cpp  # 目标预测功能实现的文件

)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${Boost_LIBRARIES} 
  ${cv_bridge_LIBRARIES}
)

add_dependencies(${PROJECT_NAME}
  ${catkin_EXPORTED_TARGETS}
)
# 找到库和头文件
include_directories(/usr/local/include/ooqp)
link_directories(/usr/local/lib)

target_link_libraries(${PROJECT_NAME}
    ${catkin_LIBRARIES}  # 如果你在用 catkin
    /usr/local/lib/libooqpbase.a
    /usr/local/lib/libooqpdense.a
    /usr/local/lib/libooqpgendense.a
    /usr/local/lib/libooqpgensparse.a
    /usr/local/lib/libooqpgondzio.a
    /usr/local/lib/libooqpsparse.a
    /usr/local/lib/libooqpmehrotra.a
    /usr/local/lib/libooqpbound.a
)
target_link_libraries(${PROJECT_NAME}
    /usr/local/lib/libma27.a
)