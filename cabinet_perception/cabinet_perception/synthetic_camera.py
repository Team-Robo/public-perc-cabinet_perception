"""Fake RealSense + segmenter for testing without a robot.

Publishes the synthetic demo cabinet (core/synthetic.py) on the same topics
and types as the real stack, plus TF cabinet -> camera_color_optical_frame:

  /camera/camera/color/image_raw, .../aligned_depth_to_color/image_raw (16UC1 mm),
  .../color/camera_info, /perc/grasp/instances (ground-truth masks)

Parameter `scene` picks the cabinet (demo: 4 levels x left/right, grid: 3 x 3 with
two physical dividers); `look_at_z` re-aims the camera (a head tilt) at runtime.
"""
import numpy as np
import rclpy
from core_r1_interfaces.msg import InstanceSegmentation, SegmentedInstance
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster

from .core.geometry import look_at
from .core.synthetic import SCENES

FRAME = "camera_color_optical_frame"


def _img(arr, enc, header):
    m = Image(header=header, height=arr.shape[0], width=arr.shape[1], encoding=enc)
    m.step = arr.shape[1] * arr.itemsize * (arr.shape[2] if arr.ndim == 3 else 1)
    m.data = np.ascontiguousarray(arr).tobytes()
    return m


class SyntheticCamera(Node):
    def __init__(self):
        super().__init__("synthetic_camera")
        self.declare_parameter("scene", "demo")          # demo (4 x 2 cubbies) | grid (3 x 3)
        self.scene = SCENES[self.get_parameter("scene").value]()
        cx = self.scene.prior.width / 2
        self.declare_parameter("eye", [cx, -1.0, 1.35])  # camera position, cabinet frame
        self.declare_parameter("look_at_z", 0.7)
        self.declare_parameter("rate", 2.0)
        self.pub_c = self.create_publisher(Image, "/camera/camera/color/image_raw", 5)
        self.pub_d = self.create_publisher(Image, "/camera/camera/aligned_depth_to_color/image_raw", 5)
        self.pub_i = self.create_publisher(CameraInfo, "/camera/camera/color/camera_info", qos_profile_sensor_data)
        self.pub_s = self.create_publisher(InstanceSegmentation, "/perc/grasp/instances", 10)
        self.tf = TransformBroadcaster(self)
        self._cache = None
        self.create_timer(1.0 / self.get_parameter("rate").value, self._tick)

    def _render(self):
        eye = list(self.get_parameter("eye").value)
        z = float(self.get_parameter("look_at_z").value)
        key = (tuple(eye), z)
        if self._cache is None or self._cache[0] != key:
            T = look_at(eye, (self.scene.prior.width / 2, 0.2, z))
            cap, _ = self.scene.render(T)
            self._cache = (key, T, cap)
        return self._cache[1], self._cache[2]

    def _tick(self):
        T, cap = self._render()
        stamp = self.get_clock().now().to_msg()
        header = Image().header
        header.stamp, header.frame_id = stamp, FRAME

        tf = TransformStamped()
        tf.header.stamp, tf.header.frame_id, tf.child_frame_id = stamp, self.scene.prior.frame_id, FRAME
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = (float(v) for v in T[:3, 3])
        q = Rotation.from_matrix(T[:3, :3]).as_quat()
        tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w = (float(v) for v in q)
        self.tf.sendTransform(tf)

        K = cap.K
        info = CameraInfo(header=header, width=K.width, height=K.height, distortion_model="plumb_bob")
        info.k = [K.fx, 0.0, K.cx, 0.0, K.fy, K.cy, 0.0, 0.0, 1.0]
        info.d = [0.0] * 5
        info.p = [K.fx, 0.0, K.cx, 0.0, 0.0, K.fy, K.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        depth_mm = np.nan_to_num(cap.depth_m * 1000.0, nan=0.0).clip(0, 65535).astype(np.uint16)

        seg = InstanceSegmentation(header=header, source_id="synthetic")
        for i, inst in enumerate(cap.instances):
            seg.instances.append(SegmentedInstance(instance_id=str(i), track_id=i, class_label=inst.label,
                                                   confidence=inst.confidence,
                                                   mask=_img(inst.mask.astype(np.uint8) * 255, "mono8", header)))
        self.pub_i.publish(info)
        self.pub_c.publish(_img(cap.rgb, "rgb8", header))
        self.pub_d.publish(_img(depth_mm, "16UC1", header))
        self.pub_s.publish(seg)


def main():
    rclpy.init()
    node = SyntheticCamera()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
