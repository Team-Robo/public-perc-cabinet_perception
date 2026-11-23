"""Estimate the cabinet frame from a gravity-aligned cloud (design note §2, option B).

Input: points + normals in a frame with z up (e.g. base_footprint, or the
IMU-levelled handheld `map`), cropped roughly by range. The frame's z = 0 need
NOT be the floor: the floor is found in the cloud (the lowest horizontal plane,
if it is much larger than a shelf board). If no floor is in view, the
cabinet base is put `base_gap` below the lowest board, so heights are then
relative to the bottom shelf. Output: T_frame_cab with the cabinet convention
(x right facing it, y into it, origin bottom-front-left) plus width / depth /
height.

Yaw comes from the dominant horizontal normal direction of vertical surfaces
facing the robot (back panel, box fronts, board front edges); extents come
from the horizontal (board) points in that rotated frame. Use it when there
is no setup-day prior, or to check the prior's pose.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .geometry import make_transform


@dataclass
class CabinetEstimate:
    T_frame_cab: np.ndarray
    width: float
    depth: float
    height: float
    yaw: float
    floor_seen: bool = True
    # Top of the bottom board in the cabinet frame when the floor is not in view (the base
    # is then BASE_GAP below it); None when the floor was seen. Shallow views see this
    # board almost edge-on, so board detection alone often misses it.
    base_board_z: Optional[float] = None


def _bands(z: np.ndarray, min_points: int = 30) -> list:
    """z of dense horizontal bands (1 cm histogram peaks), bottom first."""
    edges = np.arange(z.min() - 0.01, z.max() + 0.02, 0.01)
    if len(edges) < 3:
        return []
    h, _ = np.histogram(z, edges)
    hs = np.convolve(h, [1, 2, 1], mode="same") / 4.0
    thr = max(min_points, 0.05 * hs.max())
    idx = [i for i in range(len(hs)) if hs[i] >= thr
           and hs[i] >= hs[max(i - 1, 0)] and hs[i] >= hs[min(i + 1, len(hs) - 1)]]
    out = []
    for i in idx:
        zc = 0.5 * (edges[i] + edges[i + 1])
        if not out or zc - out[-1] > 0.03:
            out.append(zc)
    return out


BASE_GAP = 0.08     # cabinet base below the bottom board top when the floor is not in view


def find_base(P: np.ndarray, N: np.ndarray, robot_xy=(0.0, 0.0), normal_tol: float = 0.9,
              floor_area: float = 0.4, base_gap: float = BASE_GAP) -> Tuple[Optional[float], bool]:
    """(z of the cabinet base, floor seen?). The lowest horizontal band is the floor
    if it covers >= floor_area m2 (5 cm cells; a board seen from ~1 m covers ~0.1).
    Otherwise it is the bottom board (or a ledge it stands on), and the base is
    base_gap below it - heights are then relative to the bottom shelf."""
    H = P[N[:, 2] > normal_tol]
    if len(H) < 100:
        return None, False
    bands = _bands(H[:, 2])
    if not bands:
        return None, False
    low = H[np.abs(H[:, 2] - bands[0]) < 0.015]
    z0 = float(np.median(low[:, 2]))
    area = len(np.unique(np.floor(low[:, :2] / 0.05).astype(np.int64), axis=0)) * 0.05 ** 2
    if area >= floor_area:
        return z0, True
    return z0 - base_gap, False


def estimate_cabinet(P: np.ndarray, N: np.ndarray, robot_xy=(0.0, 0.0),
                     normal_tol: float = 0.9, width_hint: Optional[float] = None,
                     base_gap: float = BASE_GAP) -> Optional[CabinetEstimate]:
    """width_hint: measured cabinet width (setup-day prior). The cabinet's sides are then snapped
    to the pair of tall walls that far apart, so the pose holds when only part of the board tops
    is seen (shallow or angled views) and next to other furniture columns.
    base_gap: bottom board top above the cabinet base, used when the floor is not in view."""
    ok = np.isfinite(P).all(1) & np.isfinite(N).all(1)
    P, N = P[ok], N[ok]
    z_base, floor_seen = find_base(P, N, robot_xy, normal_tol, base_gap=base_gap)
    if z_base is None:
        return None
    est = _estimate_floor_aligned(P - np.array([0.0, 0.0, z_base]), N, robot_xy, normal_tol,
                                  width_hint=width_hint)
    if est is None:
        return None
    est.T_frame_cab[2, 3] += z_base
    est.floor_seen = floor_seen
    est.base_board_z = None if floor_seen else base_gap
    return est


def _walls_x(A: np.ndarray, N: np.ndarray, x_ax: np.ndarray, depth: float, z_top: float,
             normal_tol: float = 0.9, coverage: float = 0.4) -> list:
    """x (cabinet axes) of tall vertical walls facing +-x: side walls and dividers. A wall spans
    most of the height and the depth; object sides do not."""
    sel = (np.abs(N @ x_ax) > normal_tol) & (A[:, 1] > 0.0) & (A[:, 1] < depth + 0.02) \
        & (A[:, 2] > 0.03) & (A[:, 2] < z_top)
    X, Y, Z = A[sel, 0], A[sel, 1], A[sel, 2]
    if len(X) < 100:
        return []
    h, e = np.histogram(X, np.arange(X.min() - 0.02, X.max() + 0.03, 0.01))
    hs = np.convolve(h, [1, 2, 1], mode="same") / 4.0
    walls = []
    for i in range(1, len(hs) - 1):
        if hs[i] < 60 or hs[i] < hs[i - 1] or hs[i] < hs[i + 1]:
            continue
        xc = 0.5 * (e[i] + e[i + 1])
        m = np.abs(X - xc) < 0.015
        zb = np.unique(np.floor(Z[m] / 0.02))
        if len(zb) * 0.02 < coverage * z_top or np.ptp(Y[m]) < coverage * depth:
            continue
        if not walls or xc - walls[-1] > 0.03:
            walls.append(float(np.median(X[m])))
    return walls


def _snap_to_walls(walls: list, width: float, lo: float, hi: float, tol: float = 0.06) -> Optional[float]:
    """Left edge of the wall pair `width` apart (+-tol) that best contains the seen boards [lo, hi]."""
    best = None
    for a in walls:
        for b in walls:
            if b <= a or abs((b - a) - width) > tol:
                continue
            overlap = min(b, hi) - max(a, lo)
            key = (overlap, -abs((b - a) - width))
            if best is None or key > best[0]:
                best = (key, 0.5 * (a + b) - 0.5 * width)
    return None if best is None else best[1]


def _estimate_floor_aligned(P: np.ndarray, N: np.ndarray, robot_xy, normal_tol,
                            max_depth: float = 0.8, width_hint: Optional[float] = None) -> Optional[CabinetEstimate]:
    """Same as estimate_cabinet, but P must already have the floor (base) at z = 0."""
    ok = np.isfinite(P).all(1) & np.isfinite(N).all(1)
    P, N = P[ok], N[ok]
    vert = np.abs(N[:, 2]) < 1 - normal_tol
    horiz = N[:, 2] > normal_tol
    if vert.sum() < 100 or horiz.sum() < 100:
        return None
    # Coarse: boards are wider than deep, so the major axis of the board points
    # is the cabinet x axis; "out" is the perpendicular on the robot's side.
    # Only dense z-bands (boards): far floor points scatter above any fixed
    # floor cut at range and would dominate the PCA.
    H = P[horiz & (P[:, 2] > 0.05)]
    if len(H) < 100:
        return None
    hist, edges = np.histogram(H[:, 2], np.arange(0.05, H[:, 2].max() + 0.02, 0.01))
    dense = np.nonzero(hist >= 0.2 * hist.max())[0]
    keep = np.isin(np.clip(np.digitize(H[:, 2], edges) - 1, 0, len(hist) - 1), dense)
    B = H[keep, :2]
    if len(B) < 100:
        return None
    c = B.mean(0)
    w, v = np.linalg.eigh(np.cov((B - c).T))
    minor = v[:, 0]
    if np.dot(minor, np.asarray(robot_xy) - c) < 0:
        minor = -minor
    a0 = float(np.arctan2(minor[1], minor[0]))
    # Fine: normals of vertical surfaces facing the robot within +-20 deg of that
    # (back panel, box fronts); side-wall inner faces are ~90 deg off and drop out.
    nv = N[vert, :2] / np.linalg.norm(N[vert, :2], axis=1, keepdims=True)
    ang = np.arctan2(nv[:, 1], nv[:, 0])
    # Mode (1 deg bins, smoothed) inside +-25 deg of the seed, then a tight mean
    # around it: a plain windowed mean is pulled towards the (biased) seed.
    d = np.degrees(np.angle(np.exp(1j * (ang - a0))))
    d = d[np.abs(d) < 25]
    a = a0
    if len(d) >= 50:
        h, e = np.histogram(d, np.arange(-25, 26, 1.0))
        h = np.convolve(h, np.ones(5), mode="same")
        mode = 0.5 * (e[np.argmax(h)] + e[np.argmax(h) + 1])
        near = d[np.abs(d - mode) < 4]
        a = a0 + np.radians(float(near.mean()))
    out_dir = np.array([np.cos(a), np.sin(a), 0.0])      # cabinet -y (out of the cabinet)
    y_ax = -out_dir
    z_ax = np.array([0.0, 0.0, 1.0])
    x_ax = np.cross(y_ax, z_ax)
    R = np.stack([x_ax, y_ax, z_ax], axis=1)               # frame <- cabinet rotation
    Q = P[horiz] @ R                                       # board points in cabinet axes
    Q = Q[Q[:, 2] > 0.03]                                  # not the floor
    if len(Q) < 100:
        return None
    x0, x1 = np.percentile(Q[:, 0], [1, 99])
    y0, y1 = np.percentile(Q[:, 1], [1, 99])
    z1 = np.percentile(Q[:, 2], 99.5)
    A = (P - np.array([x0, y0, 0.0]) @ R.T) @ R            # all points, cabinet axes, origin at (x0, y0)
    strip = float(y1 - y0)

    def back_panel(z_max: float) -> Optional[float]:
        """Depth to the back panel: the deepest dense vertical surface facing the robot inside
        the footprint (object fronts are shallower). Board tops seen from just above a board
        show only its front strip, so the strip alone underestimates the depth."""
        facing = (A[:, 0] > 0.0) & (A[:, 0] < x1 - x0) & (A[:, 2] > 0.03) & (A[:, 2] < z_max) \
            & (A[:, 1] > 0.0) & (A[:, 1] < max_depth) & (N[:, :2] @ out_dir[:2] > normal_tol)
        if facing.sum() < 200:
            return None
        h, e = np.histogram(A[facing, 1], np.arange(0.0, max_depth + 0.01, 0.01))
        dense = np.nonzero(h >= max(50, 0.2 * h.max()))[0]
        return float(e[dense[-1] + 1]) if len(dense) else None

    # Pass 1: provisional depth from the back panel at any height.
    depth = max(strip, back_panel(np.inf) or 0.0)
    # Pass 2: levels above the camera show only their undersides and the side walls, so the
    # height also comes from those - inside the footprint, short of the back (a room wall
    # behind the cabinet goes up to the ceiling). Using only the board-top strip here left
    # a near-empty window from shallow views, and the height collapsed to the bottom board.
    inside = (A[:, 0] > 0.0) & (A[:, 0] < x1 - x0) & (A[:, 1] > 0.02) & (A[:, 1] < depth - 0.05) \
        & (A[:, 2] > 0.03)
    down = inside & (N[:, 2] < -normal_tol)
    side = inside & (np.abs(N[:, 2]) < 1 - normal_tol)
    if down.sum() >= 50:
        z1 = max(z1, float(np.percentile(A[down, 2], 99.5)) + 0.02)
    if side.sum() >= 200:
        z1 = max(z1, float(np.percentile(A[side, 2], 99)))
    # Pass 3: final depth from the back panel below the cabinet top only (a room wall seen
    # above the cabinet must not count); keep the provisional one if that has too few points.
    back = back_panel(z1)
    if back is not None:
        depth = max(strip, back)
    width = float(x1 - x0)
    if width_hint:
        # known width: sides = the tall wall pair that far apart around the seen boards
        H = A[(N[:, 2] > normal_tol) | (N[:, 2] < -normal_tol)]
        H = H[(H[:, 1] > -0.02) & (H[:, 1] < depth) & (H[:, 2] > 0.03) & (H[:, 2] < z1 + 0.05)]
        lo, hi = (np.percentile(H[:, 0], [5, 95]) if len(H) >= 50 else (0.0, width))
        left = _snap_to_walls(_walls_x(A, N, x_ax, depth, z1, normal_tol), width_hint, lo, hi)
        if left is not None:
            x0, width = x0 + left, float(width_hint)
    origin = R @ np.array([x0, y0, 0.0])
    return CabinetEstimate(make_transform(R, origin), width, float(depth), float(z1), a)
