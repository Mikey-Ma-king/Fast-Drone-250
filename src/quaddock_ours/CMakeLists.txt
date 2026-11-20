cmake_minimum_required(VERSION 3.0.2)
project(quaddock_ours)

find_package(catkin REQUIRED COMPONENTS nav_msgs rospy std_msgs)
catkin_python_setup()
catkin_package()

catkin_install_python(PROGRAMS
  scripts/ours_sw3_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
