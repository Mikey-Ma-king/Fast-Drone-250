cmake_minimum_required(VERSION 3.0.2)
project(flow_publisher)

## 查找catkin的包
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  message_generation
)

## 添加消息文件
add_message_files(
  FILES
  FlowDataMsg.msg
)

## 生成消息
generate_messages(
  DEPENDENCIES
  std_msgs
)

## 声明catkin包
catkin_package(
  CATKIN_DEPENDS roscpp std_msgs message_runtime
)

## 包含头文件目录
include_directories(
  include/flow_publisher
  ${catkin_INCLUDE_DIRS}
)

## 编译C代码
add_library(check_lib src/check.c)
add_library(flow_decode_lib src/flow_decode.c)

target_link_libraries(flow_decode_lib
  check_lib
)

## 添加可执行文件
add_executable(flow_publisher_node src/flow_publisher_node.cpp)

## 链接库
target_link_libraries(flow_publisher_node
  check_lib
  flow_decode_lib
  ${catkin_LIBRARIES}
)
