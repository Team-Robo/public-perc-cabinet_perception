"""Detector masks -> 3D shelf members, and multi-view de-duplication."""
from typing import List, Optional

import numpy as np
from scipy import ndimage

from .types import Member


def member_from_mask(mask: np.ndarray, P_cab: np.ndarray, depth_m: np.ndarray, label: str,
                     confidence: float, rgb: Optional[np.ndarray] = None, track_id: int = -1,
                     erode_px: int = 3, min_points: int = 30, crop_pad: int = 8) -> Optional[Member]:
    """mask: (H, W) bool. P_cab: (H, W, 3) points in the cabinet frame.
    Background that leaks through the mask edge is removed by a range test
    around the median depth of the ERODED mask; the extent is then taken from
    the full mask (erosion alone would shrink every object by a few cm and
    make touching objects look separated)."""
    full = mask.astype(bool) & np.isfinite(depth_m)
    core = full
    if erode_px > 0:
        eroded = ndimage.binary_erosion(mask.astype(bool), iterations=erode_px) & full
        if eroded.sum() >= min_points:
            core = eroded
    if core.sum() < min_points:
        return None
    d = depth_m[core]
    med = np.median(d)
    mad = 1.4826 * np.median(np.abs(d - med))
    keep = np.abs(depth_m[full] - med) < max(0.06, 3 * mad)
    pts = P_cab[full][keep]
    if len(pts) < min_points:
        return None
    lo = np.percentile(pts, 1, axis=0)
    hi = np.percentile(pts, 99, axis=0)
    ys, xs = np.nonzero(mask)
    bbox = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
    crop = None
    if rgb is not None:
        h, w = mask.shape
        x0, y0 = max(int(bbox[0]) - crop_pad, 0), max(int(bbox[1]) - crop_pad, 0)
        x1, y1 = min(int(bbox[2]) + crop_pad + 1, w), min(int(bbox[3]) + crop_pad + 1, h)
        crop = np.ascontiguousarray(rgb[y0:y1, x0:x1])
    return Member(label=label, confidence=float(confidence), centroid=np.median(pts, axis=0),
                  lo=lo, hi=hi, n_points=len(pts), bbox=bbox, crop=crop, track_id=track_id)


def fuse_members(members: List[Member], dist: float = 0.06) -> List[Member]:
    """Greedy merge of detections of the same physical object across views:
    centroids closer than `dist` in x/y AND overlapping z extents (the x/y
    test alone would merge objects stacked on different shelves; z centroids
    shift between views of tall objects). Label = most confident view."""
    out: List[Member] = []
    for m in sorted(members, key=lambda m: -m.confidence):
        hit = next((o for o in out if np.linalg.norm(o.centroid[:2] - m.centroid[:2]) < dist
                    and o.lo[2] < m.hi[2] and m.lo[2] < o.hi[2]), None)
        if hit is None:
            out.append(Member(**{**m.__dict__}))
            continue
        w0, w1 = hit.views, m.views
        hit.centroid = (hit.centroid * w0 + m.centroid * w1) / (w0 + w1)
        hit.lo = np.minimum(hit.lo, m.lo)
        hit.hi = np.maximum(hit.hi, m.hi)
        hit.views = w0 + w1
        if m.n_points > hit.n_points and m.crop is not None:
            hit.crop, hit.bbox = m.crop, m.bbox
        hit.n_points = max(hit.n_points, m.n_points)
    return out
