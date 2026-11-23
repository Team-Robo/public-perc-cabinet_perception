cmake_minimum_required(VERSION 3.8)
project(object_grouping_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  msg/DetectedObject.msg
  msg/Shelf.msg
  msg/ObjectGroup.msg
  msg/CabinetScene.msg
  msg/GroupMatch.msg
  srv/GetObjectGroup.srv
  DEPENDENCIES std_msgs sensor_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
