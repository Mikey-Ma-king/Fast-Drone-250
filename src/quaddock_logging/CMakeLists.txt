cmake_minimum_required(VERSION 3.0.2)
project(quaddock_logging)
find_package(catkin REQUIRED)
catkin_package()
catkin_install_python(PROGRAMS scripts/record_experiment.py scripts/mark_event.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY config DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
