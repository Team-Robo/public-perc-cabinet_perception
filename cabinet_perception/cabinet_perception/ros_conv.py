"""CabinetModel -> ROS messages: v0 CabinetScene (object_grouping_interfaces)
and an RViz MarkerArray. Images are hand-rolled (no cv_bridge), as in
object_grouping, so numpy ABI clashes stay out of the node.

v0 mapping (design note §7, no message change):
  Shelf        one per level: height_m = board top z (cabinet frame), free_width_m = widest free segment
  ObjectGroup  one per NON-EMPTY cubby: group_id = cubby_id ("s2_left", "s1_middle", "s3_c4"),
               side = column name, members + crops; category = the one shared by >= 2/3 of the
               members (object_list.yaml), else empty and object_grouping judges the group by appearance
"""
from typing import Dict, Optional

import numpy as np
from geometry_msgs.msg import Point, Vector3
from object_grouping_interfaces.msg import CabinetScene, DetectedObject, ObjectGroup, Shelf
from sensor_msgs.msg import Image, PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray

from .core.cubbies import free_box_height
from .core.types import CabinetModel, Member


def cloud_to_msg(P: np.ndarray, header, is_object: Optional[np.ndarray] = None) -> PointCloud2:
    """(M, 3) points -> PointCloud2 with x, y, z and `object` (1 = inside a detector mask)."""
    pts = np.zeros(len(P), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("object", "f4")])
    pts["x"], pts["y"], pts["z"] = P[:, 0], P[:, 1], P[:, 2]
    if is_object is not None:
        pts["object"] = is_object.astype(np.float32)
    fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
              for i, n in enumerate(("x", "y", "z", "object"))]
    return PointCloud2(header=header, height=1, width=len(pts), fields=fields, is_bigendian=False,
                       point_step=16, row_step=16 * len(pts), is_dense=True, data=pts.tobytes())


def np_to_image(img, header=None) -> Image:
    msg = Image()
    if header is not None:
        msg.header = header
    if img is None:
        return msg
    img = np.ascontiguousarray(img, dtype=np.uint8)
    msg.height, msg.width = img.shape[:2]
    msg.encoding, msg.step = "rgb8", img.shape[1] * 3
    msg.data = img.tobytes()
    return msg


def image_to_np(msg: Image) -> np.ndarray:
    """rgb8/bgr8/mono8/16UC1/32FC1 -> numpy (colour always RGB)."""
    enc = msg.encoding.lower()
    dt = {"16uc1": np.uint16, "mono16": np.uint16, "32fc1": np.float32}.get(enc, np.uint8)
    a = np.frombuffer(bytes(msg.data), dt)
    ch = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}.get(enc, 1)
    row = msg.step // np.dtype(dt).itemsize
    a = a.reshape(msg.height, row)[:, :msg.width * ch].reshape(msg.height, msg.width, ch)
    if enc in ("bgr8", "bgra8"):
        a = a[..., [2, 1, 0]]
    elif ch == 4:
        a = a[..., :3]
    elif ch == 1:
        a = a[..., 0]
    return np.ascontiguousarray(a)


def member_to_msg(m: Member, categories: Dict[str, str]) -> DetectedObject:
    d = DetectedObject(track_id=int(m.track_id), class_name=m.label,
                       category=categories.get(m.label, ""), confidence=float(m.confidence),
                       num_points=int(m.n_points))
    d.bbox = [float(v) for v in m.bbox]
    c = (m.lo + m.hi) / 2
    d.pose.pose.position.x, d.pose.pose.position.y, d.pose.pose.position.z = (float(v) for v in c)
    d.pose.pose.orientation.w = 1.0
    d.dimensions = Vector3(x=float(m.dims[0]), y=float(m.dims[1]), z=float(m.dims[2]))
    return d


def model_to_scene(model: CabinetModel, header, categories: Optional[Dict[str, str]] = None) -> CabinetScene:
    categories = categories or {}
    msg = CabinetScene(header=header)
    for level in sorted({c.level for c in model.cubbies}):
        cs = [c for c in model.cubbies if c.level == level]
        free = max((s.width for c in cs for s in c.free), default=0.0)
        msg.shelves.append(Shelf(shelf_id=level, height_m=float(cs[0].board_z), free_width_m=float(free)))
    for c in model.cubbies:
        if not c.members:
            continue
        ms = sorted(c.members, key=lambda m: m.centroid[0])
        g = ObjectGroup(group_id=c.cubby_id, shelf_id=c.level, side=c.side, grouping="", category="")
        # group category: shared by >= 2/3 of its listed members; mixed groups stay open and are
        # judged by appearance / labels in object_grouping
        cats = [categories.get(m.label, "") for m in c.members]
        cats = [x for x in cats if x]
        if cats:
            top = max(set(cats), key=cats.count)
            if cats.count(top) >= 2 / 3 * len(cats):
                g.grouping, g.category = "category", top
        g.members = [member_to_msg(m, categories) for m in ms]
        if any(m.crop is not None for m in ms):
            g.member_crops = [np_to_image(m.crop, header) for m in ms]
        msg.groups.append(g)
    return msg


def _cube(header, ns, i, lo, hi, rgba) -> Marker:
    m = Marker(header=header, ns=ns, id=i, type=Marker.CUBE, action=Marker.ADD)
    c = (np.asarray(lo) + np.asarray(hi)) / 2
    s = np.maximum(np.asarray(hi) - np.asarray(lo), 0.002)
    m.pose.position = Point(x=float(c[0]), y=float(c[1]), z=float(c[2]))
    m.pose.orientation.w = 1.0
    m.scale = Vector3(x=float(s[0]), y=float(s[1]), z=float(s[2]))
    m.color.r, m.color.g, m.color.b, m.color.a = rgba
    return m


def model_to_markers(model: CabinetModel, header) -> MarkerArray:
    arr = MarkerArray(markers=[Marker(header=header, action=Marker.DELETEALL)])
    i = 0
    for b in model.boards:
        rgba = (0.6, 0.6, 0.6, 0.5) if b.source not in ("prior", "base") else (1.0, 0.5, 0.0, 0.5)
        arr.markers.append(_cube(header, "boards", i, (b.x_min, b.y_min, b.z - 0.005), (b.x_max, b.y_max, b.z), rgba))
        i += 1
    for c in model.cubbies:
        if c.column > 1:          # split between this column and the one on its left
            top = c.board_z + (c.clearance_m if np.isfinite(c.clearance_m) else 0.3)
            arr.markers.append(_cube(header, "splits", i, (c.x_min - 0.003, 0.0, c.board_z),
                                     (c.x_min + 0.003, 0.02, top), (1.0, 0.0, 1.0, 0.8)))
            i += 1
        for s in c.free:
            alpha = 0.35 if s.surface_seen else 0.2         # fainter: inferred, board top not seen
            rgba = (0.1, 0.9, 0.1, alpha) if s.adjacent_to_group else (0.1, 0.6, 1.0, alpha)
            # free space as a box: base on the board, up to the board above (capped)
            arr.markers.append(_cube(header, "free", i, (s.x_min, 0.0, c.board_z),
                                     (s.x_max, s.usable_depth_m, c.board_z + max(0.01, free_box_height(c))), rgba))
            i += 1
        for m in c.members:
            arr.markers.append(_cube(header, "members", i, m.lo, m.hi, (0.2, 0.3, 1.0, 0.4)))
            i += 1
            t = Marker(header=header, ns="labels", id=i, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD)
            t.pose.position = Point(x=float(m.centroid[0]), y=float(m.lo[1]), z=float(m.hi[2] + 0.03))
            t.pose.orientation.w = 1.0
            t.scale.z = 0.03
            t.color.r = t.color.g = t.color.b = t.color.a = 1.0
            t.text = m.label
            arr.markers.append(t)
            i += 1
        t = Marker(header=header, ns="cubbies", id=i, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD)
        t.pose.position = Point(x=float((c.x_min + c.x_max) / 2), y=-0.03, z=float(c.board_z + 0.02))
        t.pose.orientation.w = 1.0
        t.scale.z = 0.035
        t.color.r, t.color.g, t.color.b, t.color.a = 1.0, 1.0, 0.0, 1.0
        t.text = c.cubby_id
        arr.markers.append(t)
        i += 1
    return arr
