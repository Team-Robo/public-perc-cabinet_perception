"""Cabinet perception node: RGB-D + instance masks -> CabinetScene (v0).

  sub  color / aligned depth / camera_info       (RealSense, /camera/camera/...)
  sub  /perc/grasp/instances                     core_r1_interfaces/InstanceSegmentation
  pub  /perception/cabinet/scene                 object_grouping_interfaces/CabinetScene (latched)
  pub  ~/markers                                 visualization_msgs/MarkerArray (cabinet frame)
  pub  ~/cloud                                   sensor_msgs/PointCloud2: fused cloud, cabinet frame,
                                                 field `object` = 1 on detector masks
  pub  ~/overlay                                 sensor_msgs/Image: last capture with boards, column
                                                 splits, free space and members drawn on it
  srv  ~/scan                                    std_srvs/Trigger: add the newest capture whose
                                                 masks have arrived, rebuild, publish (revision++)
  srv  ~/reset                                   std_srvs/Trigger: forget all captures (and the
                                                 estimated cabinet pose)

The BT calls ~/scan once per head viewpoint; captures are fused in the cabinet
frame. Needs TF cabinet <- camera optical frame: either a static
`fixed_frame -> cabinet` from the prior (launch file), or `localize: true`,
which estimates the cabinet pose from the first scan's cloud (core/localize.py)
and broadcasts it (`relocalize: true`: from every scan - a handheld camera has
no odometry, so its `map` moves with it). The floor is found in the cloud, so
the fixed frame's z = 0 doesn't have to be the floor. `live_period > 0` rebuilds from the newest single frame
every period - for watching it in Foxglove/RViz, not for the BT.
"""
import dataclasses
import threading
from collections import deque

import numpy as np
import rclpy
import yaml
from core_r1_interfaces.msg import InstanceSegmentation
from geometry_msgs.msg import TransformStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from object_grouping_interfaces.msg import CabinetScene
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import Header
from std_srvs.srv import Trigger
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformListener
from visualization_msgs.msg import MarkerArray

from .core.geometry import Intrinsics, depth_to_metres, make_transform, quat_to_matrix
from .core.localize import BASE_GAP, estimate_cabinet
from .core.pipeline import CabinetPerception, Capture, Instance, summary
from .core.prior import CabinetPrior
from .core.stabilize import Stabilizer
from .ros_conv import cloud_to_msg, image_to_np, model_to_markers, model_to_scene, np_to_image

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     reliability=ReliabilityPolicy.RELIABLE)


def _stamp_ns(stamp) -> int:
    return stamp.sec * 1_000_000_000 + stamp.nanosec


class CabinetNode(Node):
    def __init__(self):
        super().__init__("cabinet_perception")
        g = self.declare_parameter
        g("prior_file", "")
        g("object_list", "")
        g("color_topic", "/camera/camera/color/image_raw")
        g("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        g("camera_info_topic", "/camera/camera/color/camera_info")
        g("instances_topic", "/perc/grasp/instances")
        g("scene_topic", "/perception/cabinet/scene")
        g("require_instances", True)    # False: boards/free space only, from the newest frame
        g("depth_scale", 0.001)
        g("tf_timeout", 0.3)
        g("buffer_frames", 30)
        # masks -> RGB-D frame: nearest buffered frame within this many seconds. The Python
        # subscriber does not receive every 30 Hz frame, so exact stamp matching misses often.
        g("mask_slop", 0.04)
        g("size_window", 10)            # localize: cabinet size = median of the last N estimates
        # relocalize: keep the last valid pose/size and cubby structure until a change is confirmed
        # over several frames (core/stabilize.py); single bad frames no longer move the cubbies
        g("stabilize", True)
        g("image_qos", "reliable")      # "reliable" (realsense default) | "sensor" (best effort)
        g("live_period", 0.0)           # > 0: rebuild from the newest frame every N s (visualisation)
        g("localize", False)            # estimate fixed_frame -> cabinet from the cloud
        g("relocalize", False)          # ... on every scan (handheld: the map moves with the camera)
        g("fixed_frame", "map")         # gravity aligned (z up); the floor is found in the cloud
        g("viz", True)                  # publish ~/cloud and ~/overlay
        p = lambda n: self.get_parameter(n).value  # noqa: E731

        self.prior = CabinetPrior.load(p("prior_file")) if p("prior_file") else CabinetPrior()
        self._size_from_prior = bool(p("prior_file"))
        self.categories = {}
        if p("object_list"):
            with open(p("object_list")) as f:
                self.categories = {str(k): str(v) for k, v in ((yaml.safe_load(f) or {}).get("categories") or {}).items()}
        self.perc = CabinetPerception(self.prior)
        self.revision = 0
        self._T_fixed_cab = None        # estimated cabinet pose (localize)
        self._sizes = deque(maxlen=max(1, p("size_window")))   # recent (width, depth, height) estimates
        self._stab = Stabilizer()
        self._lock = threading.Lock()
        self._frames = deque(maxlen=p("buffer_frames"))        # (stamp_ns, color, depth)
        self._instances = deque(maxlen=p("buffer_frames"))     # (stamp_ns, msg)
        self._info = None
        self._last_live_stamp = None

        self.tf_buffer = Buffer()
        # own thread: ~/scan blocks on the lookup inside a callback
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.tf_static = StaticTransformBroadcaster(self)
        self.create_subscription(CameraInfo, p("camera_info_topic"), self._on_info, qos_profile_sensor_data)
        self.create_subscription(InstanceSegmentation, p("instances_topic"), self._on_instances, 10)
        # Best-effort ~1 MB images are often dropped by DDS (fragment loss); reliable by default.
        img_qos = qos_profile_sensor_data if p("image_qos") == "sensor" else QoSProfile(depth=5)
        subs = [Subscriber(self, Image, p("color_topic"), qos_profile=img_qos),
                Subscriber(self, Image, p("depth_topic"), qos_profile=img_qos)]
        self._sync = ApproximateTimeSynchronizer(subs, 10, 0.02)
        self._sync.registerCallback(self._on_rgbd)

        self.scene_pub = self.create_publisher(CabinetScene, p("scene_topic"), LATCHED)
        self.marker_pub = self.create_publisher(MarkerArray, "~/markers", LATCHED)
        self.cloud_pub = self.create_publisher(PointCloud2, "~/cloud", LATCHED)
        self.overlay_pub = self.create_publisher(Image, "~/overlay", LATCHED)
        self.create_service(Trigger, "~/scan", self._scan)
        self.create_service(Trigger, "~/reset", self._reset)
        if p("live_period") > 0:
            self.create_timer(p("live_period"), self._live)
        self.create_timer(5.0, self._check_graph)
        self.get_logger().info(
            f"cabinet prior {self.prior.width}x{self.prior.depth}x{self.prior.height} m, "
            f"{len(self.prior.boards_z)} prior boards, frame '{self.prior.frame_id}'"
            + (f", localize in '{p('fixed_frame')}'" if p("localize") else "")
            + (f", live every {p('live_period')} s" if p("live_period") > 0 else ""))

    def _check_graph(self):
        """Two launches at once (e.g. synthetic + realsense) mix frames from both cameras."""
        twins = [n for n, ns in self.get_node_names_and_namespaces()
                 if n == self.get_name() and ns == self.get_namespace()]
        topic = self.get_parameter("color_topic").value
        pubs = sorted({i.node_name for i in self.get_publishers_info_by_topic(topic)})
        if len(twins) > 1 or len(pubs) > 1:
            self.get_logger().error(
                f"{len(twins)} '{self.get_name()}' nodes and {len(pubs)} publishers on {topic} ({', '.join(pubs)}): "
                "frames from different cameras get mixed. Is another launch still running? Stop all but one.",
                throttle_duration_sec=30.0)

    def _on_info(self, msg):
        self._info = msg

    def _on_rgbd(self, color, depth):
        with self._lock:
            self._frames.append((_stamp_ns(color.header.stamp), color, depth))

    def _on_instances(self, msg):
        if not msg.invalidated:
            with self._lock:
                self._instances.append((_stamp_ns(msg.header.stamp), msg))

    def _pick(self):
        """Newest masks with an RGB-D frame within mask_slop (or newest frame if masks are optional)."""
        with self._lock:
            frames, insts = list(self._frames), list(self._instances)
        slop = int(self.get_parameter("mask_slop").value * 1e9)
        if frames:
            stamps = np.array([f[0] for f in frames], dtype=np.int64)
            for stamp, msg in reversed(insts):
                i = int(np.argmin(np.abs(stamps - stamp)))
                if abs(int(stamps[i]) - stamp) <= slop:
                    return frames[i][1], frames[i][2], msg
        if frames and not self.get_parameter("require_instances").value:
            return frames[-1][1], frames[-1][2], None
        return None

    def _scan(self, req, resp):
        try:
            resp.success, resp.message = self._do_scan()
        except Exception as exc:  # noqa: BLE001 - a service must answer
            self.get_logger().error(f"scan failed: {exc}")
            resp.success, resp.message = False, str(exc)
        return resp

    def _live(self):
        try:
            ok, msg = self._do_scan(live=True)
            if not ok:
                self.get_logger().warn(f"live: {msg}", throttle_duration_sec=5.0)
                self._publish_status(msg)
        except Exception as exc:  # noqa: BLE001 - keep the timer alive
            self.get_logger().warn(f"live scan failed: {exc}", throttle_duration_sec=5.0)
            self._publish_status(str(exc))

    def _publish_status(self, text: str):
        """Live view never freezes: newest colour frame with a red banner saying why no scene."""
        if not self.get_parameter("viz").value:
            return
        with self._lock:
            newest = self._frames[-1][1] if self._frames else None
        if newest is None:
            return
        try:
            import cv2
        except ImportError:
            return
        img = np.ascontiguousarray(image_to_np(newest)[..., :3])
        h = max(28, img.shape[0] // 16)
        cv2.rectangle(img, (0, 0), (img.shape[1], h), (200, 0, 0), -1)
        cv2.putText(img, text[:90], (8, int(h * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, h / 45.0, (255, 255, 255), 1,
                    cv2.LINE_AA)
        self.overlay_pub.publish(np_to_image(img, newest.header))

    def _lookup(self, target: str, source: str, stamp) -> np.ndarray:
        tf = self.tf_buffer.lookup_transform(target, source, Time.from_msg(stamp),
                                             timeout=Duration(seconds=self.get_parameter("tf_timeout").value))
        t, q = tf.transform.translation, tf.transform.rotation
        return make_transform(quat_to_matrix((q.x, q.y, q.z, q.w)), (t.x, t.y, t.z))

    def _cabinet_from_camera(self, cam_frame: str, stamp, depth_m: np.ndarray, K: Intrinsics) -> np.ndarray:
        """T cabinet <- camera, localising the cabinet first if asked to."""
        if not self.get_parameter("localize").value:
            return self._lookup(self.prior.frame_id, cam_frame, stamp)
        fixed = self.get_parameter("fixed_frame").value
        T_fixed_cam = self._lookup(fixed, cam_frame, stamp)
        if self._T_fixed_cab is None or self.get_parameter("relocalize").value:
            from .demo import _cloud     # same range cropping as the offline bag runner
            # no z crop: the fixed frame's z = 0 need not be the floor (handheld camera_height is a guess)
            P, N, cam = _cloud(depth_m, K, T_fixed_cam, (0.3, 2.5, np.inf), z_min=-np.inf)
            hint = dict(width_hint=self.prior.width,          # setup-day width: snap the sides to the walls
                        base_gap=self.prior.boards_z[0] if self.prior.boards_z else BASE_GAP) \
                if self._size_from_prior else {}
            est = estimate_cabinet(P, N, robot_xy=cam[:2], **hint)
            if est is None:
                if self._T_fixed_cab is None:
                    raise RuntimeError("could not localise a cabinet (need visible board tops and walls)")
                # relocalize: one bad frame keeps the last pose instead of freezing the scene
                self.get_logger().warn("cabinet not found in this frame, keeping the last pose",
                                       throttle_duration_sec=5.0)
                return np.linalg.inv(self._T_fixed_cab) @ T_fixed_cam
            first = self._T_fixed_cab is None
            stab = self.get_parameter("stabilize").value and self.get_parameter("relocalize").value
            if stab:
                dims = (self.prior.width, self.prior.depth, self.prior.height) if self._size_from_prior \
                    else (est.width, est.depth, est.height)
                T, size, state = self._stab.update_pose(est.T_frame_cab, dims)
                if state == "outlier":
                    self.get_logger().debug("localisation outlier, keeping the last valid pose")
                elif state == "moved":
                    self.get_logger().info("camera moved: new cabinet pose adopted")
                self._T_fixed_cab = T
            else:
                self._T_fixed_cab = est.T_frame_cab
            if not self._size_from_prior:         # no setup-day prior: size from the cloud
                if stab:
                    w, d, h = size
                else:                             # median of recent estimates
                    self._sizes.append((est.width, est.depth, est.height))
                    w, d, h = np.median(np.array(self._sizes), axis=0)
                self.prior.width, self.prior.depth, self.prior.height = float(w), float(d), float(h) + 0.05
                self.perc.base_board_z = est.base_board_z
            self._broadcast_cabinet(fixed)
            (self.get_logger().info if first else self.get_logger().debug)(
                f"cabinet localised in '{fixed}': {est.width:.2f} x {est.depth:.2f} x {est.height:.2f} m, "
                f"yaw {np.degrees(est.yaw):.1f} deg, origin {np.round(est.T_frame_cab[:3, 3], 3).tolist()}, "
                + ("floor seen" if est.floor_seen else "no floor in view: heights relative to the bottom board"))
        return np.linalg.inv(self._T_fixed_cab) @ T_fixed_cam

    def _broadcast_cabinet(self, fixed: str):
        T = self._T_fixed_cab
        m = TransformStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id, m.child_frame_id = fixed, self.prior.frame_id
        m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = (float(v) for v in T[:3, 3])
        q = Rotation.from_matrix(T[:3, :3]).as_quat()
        m.transform.rotation.x, m.transform.rotation.y, m.transform.rotation.z, m.transform.rotation.w = (float(v) for v in q)
        self.tf_static.sendTransform(m)

    def _do_scan(self, live: bool = False):
        if self._info is None:
            return False, "no camera_info yet"
        picked = self._pick()
        if picked is None:
            return False, "no RGB-D frame with matching instance masks yet"
        color, depth, inst_msg = picked
        if live:
            if depth.header.stamp == self._last_live_stamp:
                return True, "no new frame"
            self._last_live_stamp = depth.header.stamp
        depth_m = depth_to_metres(image_to_np(depth), self.get_parameter("depth_scale").value)
        K = Intrinsics.from_k(self._info.k, self._info.width, self._info.height)
        T = self._cabinet_from_camera(depth.header.frame_id or self._info.header.frame_id, depth.header.stamp,
                                      depth_m, K)
        instances = []
        if inst_msg is not None:
            for si in inst_msg.instances:
                mask = image_to_np(si.mask) > 0
                if mask.shape != depth_m.shape:
                    self.get_logger().warn(f"mask {mask.shape} != depth {depth_m.shape}, skipping {si.class_label}")
                    continue
                instances.append(Instance(si.class_label, si.confidence, mask, int(si.track_id)))
        if live:
            self.perc.reset()
        cap = Capture(depth_m=depth_m, K=K, T_cab_cam=T, rgb=image_to_np(color), instances=instances)
        n_new = self.perc.add_capture(cap)
        model = self.perc.build()
        if live and self.get_parameter("stabilize").value and self.get_parameter("relocalize").value:
            _, structure = self._stab.vote(model)
            if structure is not None:            # keep the last valid cubbies; place this frame's objects in them
                saved = self.perc.prior
                boards_z, splits = structure
                self.perc.prior = dataclasses.replace(saved, boards_z=boards_z, dividers_x=splits)
                try:
                    model = self.perc.build()
                finally:
                    self.perc.prior = saved
        self.revision += 1
        header = Header(frame_id=self.prior.frame_id, stamp=depth.header.stamp)
        self.scene_pub.publish(model_to_scene(model, header, self.categories))
        self.marker_pub.publish(model_to_markers(model, header))
        if self.get_parameter("viz").value:
            self._publish_viz(cap, model, header, color.header)
        text = summary(model)
        log = self.get_logger().debug if live else self.get_logger().info
        for note in self.perc.notes:
            log(note)
        log(f"revision {self.revision} ({self.perc.n_captures} captures):\n{text}")
        return True, (f"revision {self.revision}: {len(model.boards)} boards, {len(model.cubbies)} cubbies, "
                      f"{sum(len(c.members) for c in model.cubbies)} members (+{n_new} detections)")

    def _publish_viz(self, cap, model, header, image_header):
        P, N = self.perc.points
        self.cloud_pub.publish(cloud_to_msg(P, header, ~np.any(N != 0, axis=1)))
        if cap.rgb is None:
            return
        try:
            from .demo import draw_overlay
            self.overlay_pub.publish(np_to_image(draw_overlay(cap, model), image_header))
        except ImportError as exc:      # no OpenCV: skip the overlay, keep everything else
            self.get_logger().warn(f"no overlay: {exc}", once=True)

    def _reset(self, req, resp):
        self.perc.reset()
        self.perc.base_board_z = None
        self._T_fixed_cab = None
        self._sizes.clear()
        self._stab.reset()
        resp.success, resp.message = True, "captures cleared"
        return resp


def main():
    rclpy.init()
    node = CabinetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
