"""Captures (RGB-D + masks + camera pose) -> CabinetModel. No ROS imports.

    perc = CabinetPerception(prior)
    perc.add_capture(capture)      # one per head viewpoint
    model = perc.build()           # boards, cubbies, members, free space
    perc.reset()                   # new scan

Points of all captures are fused in the cabinet frame on a voxel grid, so
several head tilts fill in lower shelves hidden by the board above.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .boards import BoardParams, detect_boards, fuse_with_prior
from .cubbies import CubbyParams, build_cubbies
from .geometry import Intrinsics, normal_map, point_map, rotate_vectors, transform_points
from .members import fuse_members, member_from_mask
from .prior import CabinetPrior
from .types import Board, CabinetModel, Member


@dataclass
class Instance:
    label: str
    confidence: float
    mask: np.ndarray          # (H, W) bool / uint8
    track_id: int = -1


@dataclass
class Capture:
    depth_m: np.ndarray       # (H, W) float32 metres, NaN = invalid
    K: Intrinsics
    T_cab_cam: np.ndarray     # 4x4 cabinet <- camera optical
    rgb: Optional[np.ndarray] = None
    instances: List[Instance] = field(default_factory=list)
    stamp: float = 0.0


@dataclass
class PerceptionParams:
    voxel: float = 0.01
    crop_margin: float = 0.05
    normal_step: int = 3
    max_range: float = 3.5
    boards: BoardParams = field(default_factory=BoardParams)
    cubbies: CubbyParams = field(default_factory=CubbyParams)


def _voxel_down(P: np.ndarray, N: np.ndarray, v: float) -> Tuple[np.ndarray, np.ndarray]:
    if len(P) == 0:
        return P, N
    _, idx = np.unique(np.floor(P / v).astype(np.int64), axis=0, return_index=True)
    return P[idx], N[idx]


class CabinetPerception:
    def __init__(self, prior: CabinetPrior, params: Optional[PerceptionParams] = None):
        self.prior = prior
        self.p = params or PerceptionParams()
        self.p.boards.thickness = prior.board_thickness
        self.p.cubbies.top_is_shelf = prior.top_is_shelf
        # Bottom board top from localisation (CabinetEstimate.base_board_z) when the floor
        # was not in view; added as a board if detection missed it (seen near edge-on).
        self.base_board_z: Optional[float] = None
        self.reset()

    def reset(self):
        self._P = np.zeros((0, 3))
        self._N = np.zeros((0, 3))
        self._members: List[Member] = []
        self.n_captures = 0
        self.notes: List[str] = []

    def _crop(self, P: np.ndarray) -> np.ndarray:
        m, pr = self.p.crop_margin, self.prior
        return (np.isfinite(P).all(axis=-1) & (P[..., 0] > -m) & (P[..., 0] < pr.width + m)
                & (P[..., 1] > -m) & (P[..., 1] < pr.depth + m)
                & (P[..., 2] > 0.0) & (P[..., 2] < pr.height + m))

    def add_capture(self, cap: Capture) -> int:
        """Returns the number of shelf members found in this capture."""
        depth = cap.depth_m.copy()
        depth[depth > self.p.max_range] = np.nan
        P_cam = point_map(depth, cap.K)
        N_cam = normal_map(P_cam, step=self.p.normal_step)
        P = transform_points(cap.T_cab_cam, P_cam)
        N = rotate_vectors(cap.T_cab_cam, N_cam)

        n_before = len(self._members)
        for inst in cap.instances:
            m = member_from_mask(np.asarray(inst.mask).astype(bool), P, depth, inst.label,
                                 inst.confidence, cap.rgb, inst.track_id)
            if m is not None and self._crop(m.centroid[None])[0]:
                self._members.append(m)

        keep = self._crop(P) & np.isfinite(N).all(axis=-1)
        # Keep object points (free-space evidence) but tag them as non-board via
        # zeroed normals, so object tops never become board candidates.
        if cap.instances:
            obj = np.zeros(depth.shape, bool)
            for inst in cap.instances:
                obj |= np.asarray(inst.mask).astype(bool)
            N = N.copy()
            N[obj] = 0.0
        Pk, Nk = _voxel_down(P[keep], N[keep], self.p.voxel)
        self._P = np.vstack([self._P, Pk])
        self._N = np.vstack([self._N, Nk])
        self._P, self._N = _voxel_down(self._P, self._N, self.p.voxel)
        self.n_captures += 1
        return len(self._members) - n_before

    def build(self) -> CabinetModel:
        pr = self.prior
        detected = detect_boards(self._P, self._N, pr.width, pr.height, self.p.boards)
        boards, self.notes = fuse_with_prior(detected, pr.boards_z, pr.width, pr.depth, self.p.boards)
        zb = self.base_board_z
        if zb is not None and not pr.boards_z and all(abs(b.z - zb) > 0.05 for b in boards):
            boards = sorted(boards + [Board(z=float(zb), x_min=0.0, x_max=pr.width, y_min=0.0, y_max=pr.depth,
                                            source="base")], key=lambda b: b.z)
            self.notes.append(f"bottom board z={zb:.3f}: from localisation (not detected)")
        members = fuse_members(self._members)
        model = build_cubbies(boards, members, self._P, self._N, pr.width, pr.depth, pr.height,
                              prior_dividers=pr.dividers, thickness=pr.board_thickness, p=self.p.cubbies)
        self.notes += model.notes
        return model

    @property
    def points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Fused cloud + normals in the cabinet frame (object points have zero normals)."""
        return self._P, self._N


def summary(model: CabinetModel) -> str:
    lines = [f"cabinet {model.width:.2f} x {model.depth:.2f} x {model.height:.2f} m, "
             f"{len(model.boards)} boards: " + ", ".join(f"{b.z:.3f}({b.source[0]})" for b in model.boards)]
    for c in model.cubbies:
        names = ", ".join(m.label or "?" for m in sorted(c.members, key=lambda m: m.centroid[0])) or "(empty)"
        free = ", ".join(f"[{s.x_min:.2f}-{s.x_max:.2f}{'' if s.adjacent_to_group else ' empty'}"
                         f"{'' if s.surface_seen else ' inferred'}]" for s in c.free)
        clr = f"{c.clearance_m:.2f}" if np.isfinite(c.clearance_m) else "open"
        lines.append(f"  {c.cubby_id:<9} z={c.board_z:.3f} clr={clr} x=[{c.x_min:.2f},{c.x_max:.2f}]  "
                     f"{names}   free: {free or '-'}")
    if model.unassigned:
        lines.append("  unassigned: " + ", ".join(m.label for m in model.unassigned))
    return "\n".join(lines)
