"""Depth image -> organised point map + normals, and rigid transforms.

Normals come from the organised depth image (central differences), so no
Open3D / PCL is needed: cheap, and good enough to tell horizontal from vertical.
Lens distortion is ignored (RealSense colour streams are close to pinhole);
rectify upstream if that ever matters.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_k(cls, k, width, height):
        k = np.asarray(k, dtype=float).reshape(3, 3)
        return cls(k[0, 0], k[1, 1], k[0, 2], k[1, 2], int(width), int(height))

    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])


def depth_to_metres(depth: np.ndarray, scale: float = 0.001) -> np.ndarray:
    """uint16 (mm) or float (m) depth -> float32 metres, invalid = NaN."""
    if np.issubdtype(depth.dtype, np.integer):
        d = depth.astype(np.float32) * scale
    else:
        d = depth.astype(np.float32)
    d[~np.isfinite(d) | (d <= 0)] = np.nan
    return d


def point_map(depth_m: np.ndarray, K: Intrinsics) -> np.ndarray:
    """(H, W) metres -> (H, W, 3) points in the optical frame (x right, y down, z forward)."""
    h, w = depth_m.shape
    u = np.arange(w, dtype=np.float32)[None, :]
    v = np.arange(h, dtype=np.float32)[:, None]
    z = depth_m
    return np.stack([(u - K.cx) * z / K.fx, (v - K.cy) * z / K.fy, z], axis=-1)


def normal_map(P: np.ndarray, step: int = 3, max_jump: float = 0.03) -> np.ndarray:
    """Organised point map -> unit normals (H, W, 3), oriented towards the camera.
    NaN where a neighbour is missing or a depth discontinuity (> max_jump, scaled
    by range) makes the local surface undefined."""
    h, w, _ = P.shape
    N = np.full_like(P, np.nan)
    s = step
    c = P[s:h - s, s:w - s]
    dx = P[s:h - s, 2 * s:] - P[s:h - s, :w - 2 * s]
    dy = P[2 * s:, s:w - s] - P[:h - 2 * s, s:w - s]
    n = np.cross(dx, dy)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = n / norm
    jump = max_jump * np.maximum(c[..., 2], 1.0)
    bad = (np.abs(dx[..., 2]) > 2 * jump) | (np.abs(dy[..., 2]) > 2 * jump) | ~(norm[..., 0] > 0)
    n[bad] = np.nan
    flip = np.einsum("ijk,ijk->ij", n, c) > 0          # face the camera (origin)
    n[flip] *= -1
    N[s:h - s, s:w - s] = n
    return N


def transform_points(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Apply a 4x4 transform to (..., 3) points."""
    return P @ T[:3, :3].T + T[:3, 3]


def rotate_vectors(T: np.ndarray, V: np.ndarray) -> np.ndarray:
    return V @ T[:3, :3].T


def quat_to_matrix(q) -> np.ndarray:
    """(x, y, z, w) -> 3x3."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def make_transform(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def look_at(eye, target, up=(0, 0, 1)) -> np.ndarray:
    """Pose of an optical camera frame (z forward, y down) at `eye` looking at
    `target`, as a 4x4 parent<-camera transform."""
    eye, target, up = (np.asarray(a, float) for a in (eye, target, up))
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return make_transform(np.stack([x, y, z], axis=1), eye)
