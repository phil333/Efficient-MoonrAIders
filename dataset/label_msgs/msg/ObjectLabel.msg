# Name of the detected object
string object_name

# 4x4 transformation matrix from object to base_footprint of the robot
float64[16] object__to__base_footprint # 4x4 row-major matrix

# 4x4 transformation matrix from base_footprint to camera of the the robot
float64[16] base_footprint__to__camera # 4x4 row-major matrix

# 4x4 transformation matrix from object to camera of the the robot
float64[16] object__to__camera # 4x4 row-major matrix

# top left-corner and bottom right-corner of object's bounding box
# (x,y) coordinates
float64[4] bounding_box # (x1, y1), (x2, y2)

# if object is fully in Field-of-View
bool object_fully_in_fov

# if object is occluded by another object
bool occluded_by_another_object
