"""Offline runner: synthetic cabinet or a recorded rosbag2 -> cabinet model + overlay PNGs.

  ros2 run cabinet_perception cabinet_demo --synthetic --out /tmp/cab
  ros2 run cabinet_perception cabinet_demo --synthetic --scene grid --out /tmp/cab   # 3 x 3 cubbies
  ros2 run cabinet_perception cabinet_demo --bag BAG --list                    # topics + frames
  ros2 run cabinet_perception cabinet_demo --bag BAG --scan-every 2.0          # where is a cabinet?
  ros2 run cabinet_perception cabinet_demo --bag BAG --at 31.5 34.0 --out /tmp/cab   # fuse these times

Bag mode has no detector masks: it reports boards, cubbies and free space
(all occupied points still block free space). The cabinet frame is estimated
from the cloud (core/localize.py) unless --prior is given with a pose in
--fixed-frame. Needs a gravity-aligned --fixed-frame (z up, floor z = 0).
"""
import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from .core.geometry import (Intrinsics, depth_to_metres, look_at, make_transform, normal_map, point_map,
                            quat_to_matrix, rotate_vectors, transform_points)
from .core.localize import estimate_cabinet
from .core.pipeline import CabinetPerception, Capture, summary
from .core.prior import CabinetPrior
from .core.cubbies import free_box_height
from .core.types import CabinetModel


# --------------------------------------------------------------------------- overlay
def overlay(cap: Capture, model: CabinetModel, path: str):
    import cv2
    img = draw_overlay(cap, model)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cv2.imwrite(path, img[..., ::-1])


def draw_overlay(cap: Capture, model: CabinetModel) -> np.ndarray:
    """RGB image: boards (cyan seen / yellow underside / orange prior or from localisation), column splits
    (magenta), free space boxes (green next to a group / blue empty; thick base = board top seen), members (red boxes)."""
    import cv2
    img = cap.rgb.copy() if cap.rgb is not None else np.zeros((cap.K.height, cap.K.width, 3), np.uint8)
    if img.shape[:2] != (cap.K.height, cap.K.width):
        img = cv2.resize(img, (cap.K.width, cap.K.height))
    T_cam_cab = np.linalg.inv(cap.T_cab_cam)

    def px(p):
        q = transform_points(T_cam_cab, np.asarray(p, float))
        if q[2] <= 0.05:
            return None
        return int(cap.K.fx * q[0] / q[2] + cap.K.cx), int(cap.K.fy * q[1] / q[2] + cap.K.cy)

    def line(a, b, col, w=2):
        pa, pb = px(a), px(b)
        if pa and pb:
            cv2.line(img, pa, pb, col, w, cv2.LINE_AA)

    for b in model.boards:
        col = (255, 140, 0) if b.source in ("prior", "base") else (255, 255, 0) if b.source == "underside" else (0, 255, 255)
        line((b.x_min, 0, b.z), (b.x_max, 0, b.z), col, 3)
        line((b.x_min, 0, b.z), (b.x_min, b.y_max, b.z), col, 1)
        line((b.x_max, 0, b.z), (b.x_max, b.y_max, b.z), col, 1)
    boxes = []                                  # free-space front faces, filled translucent at the end
    for c in model.cubbies:
        z = c.board_z
        if c.column > 1:
            line((c.x_min, 0, z), (c.x_min, 0, z + min(c.clearance_m, 0.4)), (255, 0, 255), 2)
        for s in c.free:
            col = (0, 255, 0) if s.adjacent_to_group else (0, 160, 255)
            # free space is a box on the board: base x range, usable depth, up to the board above
            x0, x1, y0, y1 = s.x_min, s.x_max, 0.02, max(0.03, s.usable_depth_m)
            z0, z1 = z + 0.005, z + 0.005 + free_box_height(c)
            front = [px(q) for q in ((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1))]
            if all(front):
                boxes.append((np.array(front, np.int32), col))
            for a, b in (((x0, y0, z1), (x1, y0, z1)), ((x0, y0, z0), (x0, y0, z1)), ((x1, y0, z0), (x1, y0, z1)),
                         ((x0, y1, z0), (x1, y1, z0)), ((x0, y1, z1), (x1, y1, z1)), ((x0, y0, z0), (x0, y1, z0)),
                         ((x1, y0, z0), (x1, y1, z0)), ((x0, y0, z1), (x0, y1, z1)), ((x1, y0, z1), (x1, y1, z1))):
                line(a, b, col, 1)
            # base: thick = board top seen there; thin = inferred (in view, nothing stands, surface too shallow)
            line((x0, y0, z0), (x1, y0, z0), col, 5 if s.surface_seen else 3)
        p = px(((c.x_min + c.x_max) / 2, 0, z + 0.03))
        if p:
            cv2.putText(img, c.cubby_id, p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        for m in c.members:
            a, b = px((m.lo[0], m.lo[1], m.hi[2])), px((m.hi[0], m.lo[1], m.lo[2]))
            if a and b:
                cv2.rectangle(img, a, b, (255, 60, 60), 2)
                cv2.putText(img, m.label, (min(a[0], b[0]), min(a[1], b[1]) - 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (255, 255, 255), 1, cv2.LINE_AA)
    if boxes:
        layer = img.copy()
        for poly, col in boxes:
            cv2.fillPoly(layer, [poly], col)
        img = cv2.addWeighted(layer, 0.25, img, 0.75, 0)
    return img


# --------------------------------------------------------------------------- synthetic
def run_synthetic(out: str, scene: str = "demo"):
    from .core.synthetic import SCENES
    sc = SCENES[scene]()
    perc = CabinetPerception(sc.prior)
    caps = []
    cx, h = sc.prior.width / 2, sc.prior.height
    for z in (0.7 * h, 0.25 * h):
        cap, _ = sc.render(look_at((cx, -1.0, 1.35), (cx, 0.2, z)))
        perc.add_capture(cap)
        caps.append(cap)
    model = perc.build()
    print(summary(model))
    for n in perc.notes:
        print("  note:", n)
    for i, cap in enumerate(caps):
        overlay(cap, model, os.path.join(out, f"synthetic_{i}.png"))
    print(f"overlays -> {out}")


# --------------------------------------------------------------------------- bag
class BagFrames:
    """Reads colour, depth, camera_info and TF from a rosbag2 (no ROS graph needed)."""

    def __init__(self, bag: str, color: str, depth: str, info: str, depth_info: str = ""):
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
        from tf2_ros import Buffer
        from rclpy.duration import Duration

        storage = "mcap" if bag.endswith(".mcap") else "sqlite3"
        r = rosbag2_py.SequentialReader()
        r.open(rosbag2_py.StorageOptions(uri=bag, storage_id=storage),
               rosbag2_py.ConverterOptions("cdr", "cdr"))
        self.types = {t.name: t.type for t in r.get_all_topics_and_types()}
        want = {color, depth, info, depth_info, "/tf", "/tf_static"} - {""}
        missing = [t for t in (color, depth) if t not in self.types]
        if missing:
            raise SystemExit(f"topics not in bag: {missing}\navailable: {sorted(self.types)}")
        r.set_filter(rosbag2_py.StorageFilter(topics=[t for t in want if t in self.types]))
        self.tf = Buffer(cache_time=Duration(seconds=36000))
        self.color: List[Tuple[float, object]] = []
        self.depth: List[Tuple[float, object]] = []
        self.info = self.depth_info = None
        self.t0 = None
        cls = {t: get_message(self.types[t]) for t in want if t in self.types}
        while r.has_next():
            topic, data, t_ns = r.read_next()
            msg = deserialize_message(data, cls[topic])
            t = t_ns * 1e-9
            self.t0 = t if self.t0 is None else self.t0
            if topic in ("/tf", "/tf_static"):
                for tr in msg.transforms:
                    if topic == "/tf_static":
                        self.tf.set_transform_static(tr, "bag")
                    else:
                        self.tf.set_transform(tr, "bag")
            elif topic == color:
                self.color.append((t, msg))
            elif topic == depth:
                self.depth.append((t, msg))
            elif topic == info and self.info is None:
                self.info = msg
            elif topic == depth_info and self.depth_info is None:
                self.depth_info = msg
        print(f"bag: {len(self.color)} colour, {len(self.depth)} depth frames, "
              f"{(self.depth[-1][0] - self.t0) if self.depth else 0:.1f} s")

    def lookup(self, target: str, source: str, stamp) -> Optional[np.ndarray]:
        from rclpy.time import Time
        try:
            tf = self.tf.lookup_transform(target, source, Time.from_msg(stamp))
        except Exception as exc:  # noqa: BLE001
            print(f"  tf {target} <- {source}: {exc}", file=sys.stderr)
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        return make_transform(quat_to_matrix((q.x, q.y, q.z, q.w)), (t.x, t.y, t.z))

    def frame_at(self, t_rel: float, fixed: str, depth_scale: float):
        """(depth_m, rgb, K, T_fixed_cam, stamp) for the depth frame nearest t_rel."""
        from .ros_conv import image_to_np
        t = self.t0 + t_rel
        td, dm = min(self.depth, key=lambda x: abs(x[0] - t))
        rgb = None
        if self.color:
            tc, cm = min(self.color, key=lambda x: abs(x[0] - td))
            if abs(tc - td) < 0.1:
                rgb = image_to_np(cm)
        info = self.depth_info or self.info
        depth = depth_to_metres(image_to_np(dm), depth_scale)
        sx, sy = depth.shape[1] / info.width, depth.shape[0] / info.height
        k = np.array(info.k, float).reshape(3, 3)
        K = Intrinsics(k[0, 0] * sx, k[1, 1] * sy, k[0, 2] * sx, k[1, 2] * sy, depth.shape[1], depth.shape[0])
        T = self.lookup(fixed, dm.header.frame_id, dm.header.stamp)
        return depth, rgb, K, T, td - self.t0


def _cloud(depth, K, T, crop, z_min=-0.05):
    """crop = (min range, max range, max z); z limits assume floor z = 0 unless z_min/max are inf."""
    P_cam = point_map(depth, K)
    N_cam = normal_map(P_cam)
    P = transform_points(T, P_cam).reshape(-1, 3)
    N = rotate_vectors(T, N_cam).reshape(-1, 3)
    cam = T[:3, 3]
    d = np.linalg.norm(P[:, :2] - cam[:2], axis=1)
    ok = np.isfinite(P).all(1) & (d > crop[0]) & (d < crop[1]) & (P[:, 2] > z_min) & (P[:, 2] < crop[2])
    return P[ok], N[ok], cam


def run_bag(a):
    bag = BagFrames(a.bag, a.color, a.depth, a.info, a.depth_info)
    if a.list:
        for k, v in sorted(bag.types.items()):
            print(f"  {k:<50} {v}")
        return
    crop = (a.min_range, a.max_range, a.max_z)
    if a.scan_every:
        print("t[s]   boards (z, fixed frame)   cabinet estimate")
        end = bag.depth[-1][0] - bag.t0
        for t in np.arange(0, end, a.scan_every):
            depth, rgb, K, T, tt = bag.frame_at(t, a.fixed_frame, a.depth_scale)
            if T is None:
                continue
            P, N, cam = _cloud(depth, K, T, crop)
            est = estimate_cabinet(P, N, robot_xy=cam[:2])
            if est is None:
                continue
            perc = CabinetPerception(CabinetPrior(width=est.width, depth=est.depth, height=est.height + 0.05))
            perc.base_board_z = est.base_board_z
            perc.add_capture(Capture(depth, K, np.linalg.inv(est.T_frame_cab) @ T, rgb))
            m = perc.build()
            if m.boards:
                print(f"{tt:6.1f}  {len(m.boards)}: " + " ".join(f"{b.z:.2f}" for b in m.boards)
                      + f"   {est.width:.2f}x{est.depth:.2f}x{est.height:.2f} m")
        return

    frames = [bag.frame_at(t, a.fixed_frame, a.depth_scale) for t in a.at]
    frames = [f for f in frames if f[3] is not None]
    if not frames:
        raise SystemExit("no usable frames (TF missing?)")
    if a.prior:
        prior = CabinetPrior.load(a.prior)
        T_fixed_cab = make_transform(quat_to_matrix(_rpy_quat(prior.pose_rpy)), prior.pose_xyz)
    else:
        P, N, cam = _cloud(frames[0][0], frames[0][2], frames[0][3], crop)
        est = estimate_cabinet(P, N, robot_xy=cam[:2])
        if est is None:
            raise SystemExit("could not localise a cabinet in the first frame (try --max-range / --at)")
        prior = CabinetPrior(width=est.width, depth=est.depth, height=est.height + 0.05)
        T_fixed_cab = est.T_frame_cab
        print(f"cabinet estimate: {est.width:.2f} x {est.depth:.2f} x {est.height:.2f} m, "
              f"yaw {np.degrees(est.yaw):.1f} deg, origin {np.round(T_fixed_cab[:3, 3], 3)}")
    perc = CabinetPerception(prior)
    if not a.prior:
        perc.base_board_z = est.base_board_z
    caps = []
    for depth, rgb, K, T, tt in frames:
        cap = Capture(depth, K, np.linalg.inv(T_fixed_cab) @ T, rgb)
        perc.add_capture(cap)
        caps.append((tt, cap))
    model = perc.build()
    print(summary(model))
    for n in perc.notes:
        print("  note:", n)
    if a.out:
        for tt, cap in caps:
            overlay(cap, model, os.path.join(a.out, f"bag_{tt:07.2f}.png"))
        print(f"overlays -> {a.out}")


def _rpy_quat(rpy):
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler("xyz", rpy).as_quat()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--scene", default="demo", choices=["demo", "grid"], help="synthetic cabinet")
    ap.add_argument("--bag")
    ap.add_argument("--list", action="store_true", help="list bag topics")
    ap.add_argument("--color", default="/camera/camera/color/image_raw")
    ap.add_argument("--depth", default="/camera/camera/aligned_depth_to_color/image_raw")
    ap.add_argument("--info", default="/camera/camera/color/camera_info")
    ap.add_argument("--depth-info", default="", help="depth camera_info if depth is not registered to colour")
    ap.add_argument("--fixed-frame", default="map", help="gravity-aligned frame (z up, floor z=0)")
    ap.add_argument("--depth-scale", type=float, default=0.001)
    ap.add_argument("--scan-every", type=float, default=0.0, help="report boards every N seconds")
    ap.add_argument("--at", type=float, nargs="*", default=[], help="bag times (s from start) to fuse")
    ap.add_argument("--prior", help="cabinet prior YAML (pose in --fixed-frame)")
    ap.add_argument("--min-range", type=float, default=0.3)
    ap.add_argument("--max-range", type=float, default=2.5)
    ap.add_argument("--max-z", type=float, default=2.2)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    if a.synthetic:
        run_synthetic(a.out or "cabinet_demo_out", a.scene)
    elif a.bag:
        if not a.list and not a.scan_every and not a.at:
            ap.error("give --list, --scan-every or --at")
        run_bag(a)
    else:
        ap.error("give --synthetic or --bag")


if __name__ == "__main__":
    main()
