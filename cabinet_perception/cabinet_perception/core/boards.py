"""Shelf boards from horizontal surface points (design note §6, steps 3-7).

A board is a z-histogram peak of upward-normal points that spans a large part
of the cabinet width; object tops are narrow, so the coverage test rejects them
(and points inside detector masks are removed before this step anyway).
Boards above the camera only show their underside: downward-normal peaks plus
the board thickness. A setup-day prior fills in boards that were not seen.
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .types import Board


@dataclass
class BoardParams:
    z_bin: float = 0.01          # histogram bin (m)
    normal_z: float = 0.9        # |n_z| above this = horizontal surface
    x_bin: float = 0.02          # coverage bin along the board (m)
    min_coverage: float = 0.5    # fraction of the cabinet width a board must span
    min_points: int = 40
    inlier: float = 0.012        # +- band around a peak used to refine z
    merge_tol: float = 0.03      # peaks / top-vs-underside closer than this are one board
    thickness: float = 0.018     # board thickness, for underside -> top
    prior_tol: float = 0.03      # detected vs prior disagreement that triggers the prior
    min_median_y: float = 0.02   # a board lies inside the cabinet, not in front of it


def _peaks(z: np.ndarray, lo: float, hi: float, p: BoardParams) -> List[float]:
    edges = np.arange(lo, hi + p.z_bin, p.z_bin)
    if len(edges) < 3 or len(z) == 0:
        return []
    h, _ = np.histogram(z, edges)
    hs = np.convolve(h, [1, 2, 1], mode="same") / 4.0
    centres = 0.5 * (edges[:-1] + edges[1:])
    idx = [i for i in range(len(hs))
           if hs[i] >= p.min_points / 4 and hs[i] >= hs[max(i - 1, 0)] and hs[i] >= hs[min(i + 1, len(hs) - 1)]]
    idx.sort(key=lambda i: -hs[i])
    return [float(centres[i]) for i in idx]


def _coverage(x: np.ndarray, x_lo: float, x_hi: float, bin_w: float) -> float:
    edges = np.arange(x_lo, x_hi + 1e-9, bin_w)
    if len(edges) < 2:
        return 0.0
    h, _ = np.histogram(x, edges)
    return float(np.mean(h >= 2))


def _boards_from(P: np.ndarray, width: float, z_lo: float, z_hi: float,
                 p: BoardParams, source: str, z_offset: float = 0.0) -> List[Board]:
    out: List[Board] = []
    for zc in _peaks(P[:, 2], z_lo, z_hi, p):
        if any(abs(zc - (b.z - z_offset)) < p.merge_tol for b in out):
            continue
        sel = P[np.abs(P[:, 2] - zc) < p.inlier]
        if len(sel) < p.min_points:
            continue
        z = float(np.median(sel[:, 2]))
        sel = sel[np.abs(sel[:, 2] - z) < p.inlier]
        if _coverage(sel[:, 0], 0.0, width, p.x_bin) < p.min_coverage:
            continue
        if np.median(sel[:, 1]) < p.min_median_y:     # floor strip in front of the cabinet
            continue
        x0, x1 = np.percentile(sel[:, 0], [1, 99])
        y0, y1 = np.percentile(sel[:, 1], [1, 99])
        out.append(Board(z=z + z_offset, x_min=float(x0), x_max=float(x1),
                         y_min=float(y0), y_max=float(y1), n_points=len(sel), source=source))
    return out


def detect_boards(P: np.ndarray, N: np.ndarray, width: float, height: float,
                  p: Optional[BoardParams] = None) -> List[Board]:
    """P, N: (M, 3) points and unit normals in the cabinet frame, already cropped
    to the cabinet box and with object (mask) points removed."""
    p = p or BoardParams()
    up = N[:, 2] > p.normal_z
    down = N[:, 2] < -p.normal_z
    z_lo, z_hi = -0.03, height + 0.05
    tops = _boards_from(P[up], width, z_lo, z_hi, p, "top")
    unders = _boards_from(P[down], width, z_lo, z_hi, p, "underside", z_offset=p.thickness)
    boards = list(tops)
    for b in unders:
        if all(abs(b.z - t.z) >= p.merge_tol for t in tops):
            boards.append(b)
    return sorted(boards, key=lambda b: b.z)


def fuse_with_prior(detected: List[Board], prior_z: Sequence[float], width: float, depth: float,
                    p: Optional[BoardParams] = None) -> Tuple[List[Board], List[str]]:
    """Prior defines WHICH boards exist (shelf geometry doesn't change); detection
    refines WHERE. Returns (boards, notes) with one note per disagreement."""
    p = p or BoardParams()
    if not prior_z:
        return detected, []
    boards, notes, used = [], [], set()
    for zp in sorted(prior_z):
        cand = [(abs(b.z - zp), i) for i, b in enumerate(detected) if i not in used]
        d, i = min(cand) if cand else (np.inf, -1)
        if d <= p.prior_tol:
            used.add(i)
            boards.append(detected[i])
        else:
            boards.append(Board(z=float(zp), x_min=0.0, x_max=width, y_min=0.0, y_max=depth, source="prior"))
            notes.append(f"board z={zp:.3f}: not seen" if not np.isfinite(d)
                         else f"board z={zp:.3f}: nearest detection {d * 100:.1f} cm off, using prior")
    for i, b in enumerate(detected):
        if i not in used:
            notes.append(f"extra board candidate z={b.z:.3f} ({b.source}) ignored (not in prior)")
    return sorted(boards, key=lambda b: b.z), notes
