"""Temporal stabilisation for live (relocalising) scans: keep the last valid cabinet pose and
structure until a change is confirmed, so single bad frames do not make the cubbies flicker.

Pose/size gate: a new localisation that disagrees with the current one (position, yaw or size
beyond tolerance) is an outlier and the last valid pose is kept. Only `confirm` consecutive
disagreeing estimates that agree with each other (the camera really moved) replace it.
Agreeing estimates are blended in (exponential smoothing), so slow drift is followed.

Structure vote: the levels (board heights) and column splits of each frame's model form a
signature. The published structure changes only when another signature wins a majority of the
last `window` frames; until then the frame is rebuilt with the last valid structure as a prior,
so its members are still placed fresh into stable cubbies.
"""
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from .types import CabinetModel


@dataclass
class StabilizerParams:
    pos_tol: float = 0.04       # m, cabinet origin
    yaw_tol: float = 4.0        # deg
    size_tol: float = 0.08      # m, any of width / depth / height
    confirm: int = 3            # consecutive consistent outliers that become the new pose
    alpha: float = 0.3          # smoothing weight of an agreeing estimate
    window: int = 5             # structure votes
    z_quant: float = 0.03       # signature resolution: board heights
    x_quant: float = 0.05       # signature resolution: split positions


def _yaw(T: np.ndarray) -> float:
    return float(np.arctan2(T[1, 0], T[0, 0]))


def _pose(x: float, y: float, z: float, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    T[:3, 3] = (x, y, z)
    return T


def _ang(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


Signature = Tuple[Tuple[int, ...], Tuple[Tuple[int, ...], ...]]


def signature(model: CabinetModel, p: StabilizerParams) -> Signature:
    """(board heights, per level: split x positions), quantised."""
    boards = tuple(int(round(b.z / p.z_quant)) for b in model.boards)
    levels: Dict[int, List[float]] = {}
    for c in model.cubbies:
        levels.setdefault(c.level, [])
        if c.column > 1:
            levels[c.level].append(c.x_min)
    splits = tuple(tuple(sorted(int(round(x / p.x_quant)) for x in levels[k])) for k in sorted(levels))
    return boards, splits


def structure_of(model: CabinetModel) -> Tuple[List[float], Dict[int, List[float]]]:
    """(board heights, {level: split x}) to rebuild a frame with this structure as its prior."""
    splits: Dict[int, List[float]] = {}
    for c in model.cubbies:
        splits.setdefault(c.level, [])
        if c.column > 1:
            splits[c.level].append(float(c.x_min))
    return [float(b.z) for b in model.boards], {k: sorted(v) for k, v in splits.items()}


class Stabilizer:
    def __init__(self, params: Optional[StabilizerParams] = None):
        self.p = params or StabilizerParams()
        self.reset()

    def reset(self):
        self.T: Optional[np.ndarray] = None
        self.size: Optional[np.ndarray] = None
        self._pending: List[Tuple[np.ndarray, np.ndarray]] = []
        self._votes: Deque[Signature] = deque(maxlen=self.p.window)
        self._structures: Dict[Signature, Tuple[List[float], Dict[int, List[float]]]] = {}
        self.adopted: Optional[Signature] = None
        self.outliers = 0

    # -- pose / size ------------------------------------------------------------------
    def _agrees(self, Ta, sa, Tb, sb) -> bool:
        p = self.p
        return (np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]) <= p.pos_tol
                and abs(np.degrees(_ang(_yaw(Ta) - _yaw(Tb)))) <= p.yaw_tol
                and np.all(np.abs(sa - sb) <= p.size_tol))

    def update_pose(self, T: np.ndarray, size) -> Tuple[np.ndarray, np.ndarray, str]:
        """New localisation -> (pose, size, 'init' | 'smoothed' | 'outlier' | 'moved')."""
        size = np.asarray(size, float)
        if self.T is None:
            self.T, self.size = T.copy(), size.copy()
            return self.T, self.size, "init"
        if self._agrees(T, size, self.T, self.size):
            a = self.p.alpha
            x, y, z = (1 - a) * self.T[:3, 3] + a * T[:3, 3]
            yaw = _yaw(self.T) + a * _ang(_yaw(T) - _yaw(self.T))
            self.T = _pose(x, y, z, yaw)
            self.size = (1 - a) * self.size + a * size
            self._pending.clear()
            return self.T, self.size, "smoothed"
        # disagrees: keep the last valid pose unless `confirm` outliers agree with each other
        if self._pending and not self._agrees(T, size, *self._pending[-1]):
            self._pending.clear()
        self._pending.append((T.copy(), size.copy()))
        if len(self._pending) >= self.p.confirm:
            Ts = [t for t, _ in self._pending]
            yaw0 = _yaw(Ts[0])
            yaw = yaw0 + float(np.mean([_ang(_yaw(t) - yaw0) for t in Ts]))
            x, y, z = np.mean([t[:3, 3] for t in Ts], axis=0)
            self.T = _pose(x, y, z, yaw)
            self.size = np.mean([s for _, s in self._pending], axis=0)
            self._pending.clear()
            self._votes.clear()             # a new view: the old structure votes no longer apply
            self.adopted = None
            return self.T, self.size, "moved"
        self.outliers += 1
        return self.T, self.size, "outlier"

    # -- structure ----------------------------------------------------------------------
    def vote(self, model: CabinetModel) -> Tuple[Signature, Optional[Tuple[List[float], Dict[int, List[float]]]]]:
        """Add this frame's structure. Returns (this frame's signature, structure to rebuild
        with) - None when the frame already matches the adopted structure."""
        sig = signature(model, self.p)
        self._structures.setdefault(sig, structure_of(model))
        self._votes.append(sig)
        best, n = Counter(self._votes).most_common(1)[0]
        if self.adopted is None or (best != self.adopted and n > len(self._votes) // 2):
            self.adopted = best
            self._structures[best] = structure_of(model) if best == sig else self._structures[best]
        if sig == self.adopted:
            self._structures[sig] = structure_of(model)   # refresh with the latest refined positions
            return sig, None
        return sig, self._structures[self.adopted]
