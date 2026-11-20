cmake_minimum_required(VERSION 3.8)
project(nlink_utils LANGUAGES C CXX)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_runtime REQUIRED)

# add_subdirectory(protocol_extracter)

add_library(nlink_utils STATIC)

target_sources(nlink_utils PRIVATE
  src/nlink_protocol.cpp
  src/nutils.cpp
  src/init_serial.cpp
  src/linktrack_protocols.cpp
  
  protocol_extracter/nprotocol_base.cpp
  protocol_extracter/nprotocol_extracter.cpp
  

  nlink_unpack/nlink_linktrack_anchorframe0.c
  nlink_unpack/nlink_iot_frame0.c
  nlink_unpack/nlink_linktrack_aoa_nodeframe0.c
  nlink_unpack/nlink_linktrack_nodeframe0.c
  nlink_unpack/nlink_linktrack_nodeframe1.c
  nlink_unpack/nlink_linktrack_nodeframe2.c
  nlink_unpack/nlink_linktrack_nodeframe3.c
  nlink_unpack/nlink_linktrack_nodeframe4.c
  nlink_unpack/nlink_linktrack_nodeframe5.c
  nlink_unpack/nlink_linktrack_nodeframe6.c

  src/nlink_linktrack_nodeframe7/nlink_linktrack_nodeframe7.c

  nlink_unpack/nlink_linktrack_tagframe0.c
  nlink_unpack/nlink_tofsense_frame0.c
  nlink_unpack/nlink_tofsensem_frame0.c
  nlink_unpack/nlink_utils.c
)

target_compile_features(nlink_utils PUBLIC
  c_std_11
  cxx_std_17
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(nlink_utils PRIVATE 
        -Wno-unused-parameter
        -Wno-sign-compare
    )
endif()

target_include_directories(nlink_utils PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${CMAKE_CURRENT_BINARY_DIR}/../../rosidl_generator_cpp
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/protocol_extracter>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/nlink_unpack>
)

ament_target_dependencies(nlink_utils
  rclcpp
  std_msgs
  rosidl_default_runtime
)

target_link_libraries(nlink_utils
  serial
#   nlink_protocol
)

add_dependencies(nlink_utils
  nlink_parser2__rosidl_generator_cpp
  nlink_parser2__rosidl_typesupport_cpp
)
