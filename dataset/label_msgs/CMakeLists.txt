cmake_minimum_required(VERSION 3.0.2)
project(label_msgs)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  message_runtime
  std_msgs
)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  ImageLabels.msg
  ObjectLabel.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
)


catkin_package(
)
