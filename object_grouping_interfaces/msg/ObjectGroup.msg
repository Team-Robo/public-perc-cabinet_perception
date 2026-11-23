# One group of objects the referee arranged together on a shelf side
# (rulebook 5.2: "arranged in groups, either by category or likeliness").

string   group_id                      # stable id within the scene, e.g. "s2_left"
uint8    shelf_id                      # Shelf.shelf_id this group sits on
string   side                          # "left" | "right" | "" (whole shelf)
string   grouping                      # "category" | "likeliness" | "" (unknown)
string   category                      # dominant rulebook category; "" if unknown / likeliness group
DetectedObject[] members      
sensor_msgs/Image[] member_crops       # RGB crops, same order/length as members; EMPTY list = label-only group,
                                       # a 0x0 image = no photo for that member
                                       # rgba8 = alpha channel is the member's segmentation mask
