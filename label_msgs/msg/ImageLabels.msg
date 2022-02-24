# Header matching the image's header
Header header

uint32 height         # image height, that is, number of rows
uint32 width          # image width, that is, number of columns

string image_encoding  # Encoding of pixels -- channel meaning, ordering, size

string base_footprint_frame
string camera_frame

# Projection/camera matrix taken from camera_info
#     [fx'  0  cx' Tx]
# P = [ 0  fy' cy' Ty]
#     [ 0   0   1   0]
float64[12] P # 3x4 row-major matrix

# Array of object labels
ObjectLabel[] object_labels
