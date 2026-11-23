# Local copy of perc_ws perception_msgs/DetectedObject (same fields, same layout),
# so this prototype builds without the perc_ws workspace. To integrate with the
# team stack later, switch ObjectGroup.msg / GetObjectGroup.srv and the Python
# imports back to perception_msgs/DetectedObject (or its successor, see
# perc_ws/docs/INTERFACE_DISCUSSION.md) and delete this file.
#
# One segmented object with a metric pose. The contract boundary between
# perception and manipulation: manipulation consumes `pose`/`dimensions`/
# `cloud`, never pixels.

int32       track_id          # segmenter tracker id; -1 if untracked
string      class_name        # detector class name (e.g. "cup")
string      category          # rulebook category from the runtime category map ("tableware", ...); "" if unmapped
float64     confidence        # detector confidence [0, 1]
float64[4]  bbox              # [x1, y1, x2, y2] px in the source image
geometry_msgs/PoseWithCovariance pose   # oriented-bounding-box centre + axes, in the parent array's header.frame_id (metres)
                                        # covariance diagonal [0],[7],[14] = position variance per axis, [35] = yaw variance,
                                        # measured across aggregated frames when produced by PerceiveObjects
geometry_msgs/Vector3 dimensions        # OBB extents along its own x/y/z axes (metres)
uint32      num_points        # points that survived filtering
sensor_msgs/PointCloud2 cloud # the object's filtered points in header.frame_id; empty unless publish_clouds
