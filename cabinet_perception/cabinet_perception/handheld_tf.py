"""Gravity-aligned TF map -> camera_link from the RealSense IMU, for a handheld camera.

Bench testing without a robot: there is no robot TF tree, so this stands in
for it. Yaw is 0 (map x = camera forward, level), pitch/roll from the
low-pass-filtered accelerometer, z = `camera_height`. Pair it with cabinet_node
`localize: true, relocalize: true`, which finds the floor and the cabinet in
the cloud every frame, so `camera_height` needn't be measured: only gravity
(the IMU) matters. `map` moves with the camera; that is fine for relocalize.

Needs the RealSense IMU: rs_launch.py enable_accel:=true enable_gyro:=true unite_imu_method:=2.
"""
import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster


class HandheldTf(Node):
    def __init__(self):
        super().__init__("handheld_tf")
        self.declare_parameter("camera_height", 1.0)        # m, a guess is fine (floor found in the cloud)
        self.declare_parameter("imu_topic", "/camera/camera/imu")
        self.declare_parameter("parent_frame", "map")
        self.declare_parameter("child_frame", "camera_link")
        self.br = TransformBroadcaster(self)
        self.g = None
        self.k = 0
        self.create_subscription(Imu, self.get_parameter("imu_topic").value, self._on_imu, qos_profile_sensor_data)
        self.create_timer(1 / 30, self._tick)

    def _on_imu(self, m):
        a = np.array([m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z])  # imu optical
        self.g = a if self.g is None else 0.98 * self.g + 0.02 * a

    def _tick(self):
        if self.g is None:
            return
        a = self.g
        u = np.array([a[2], -a[0], -a[1]])               # "up" in camera_link (x fwd, y left, z up)
        u /= np.linalg.norm(u)
        f = np.array([1.0, 0.0, 0.0])
        ex = f - f.dot(u) * u
        ex /= np.linalg.norm(ex)
        ey = np.cross(u, ex)
        R = np.stack([ex, ey, u], 1).T                   # map <- camera_link
        q = Rotation.from_matrix(R).as_quat()
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.get_parameter("parent_frame").value
        t.child_frame_id = self.get_parameter("child_frame").value
        t.transform.translation.z = float(self.get_parameter("camera_height").value)
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = (float(v) for v in q)
        self.br.sendTransform(t)
        self.k += 1
        if self.k % 300 == 1:
            pitch = np.degrees(np.arcsin(-R[2, 0]))
            roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
            self.get_logger().info(f"camera pitch {-pitch:+.1f} deg (+ = up), roll {roll:+.1f} deg, "
                                   f"height {t.transform.translation.z} m")


def main():
    rclpy.init()
    node = HandheldTf()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
