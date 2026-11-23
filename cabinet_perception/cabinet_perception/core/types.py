"""Plain data types for the cabinet model. No ROS imports.

All geometry is in the cabinet frame: x right (facing the cabinet), y into the
cabinet, z up, origin at the bottom-front-left corner. Metres.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Board:
    z: float                    # top surface height
    x_min: float = 0.0
    x_max: float = 0.0
    y_min: float = 0.0
    y_max: float = 0.0
    n_points: int = 0
    source: str = "top"         # "top" (upward normals) | "underside" | "prior" | "base" (localisation)

    @property
    def width(self) -> float:
        return self.x_max - self.x_min


@dataclass
class Member:
    """One detected object on a shelf, with its 3D extent."""
    label: str
    confidence: float
    centroid: np.ndarray        # (3,)
    lo: np.ndarray              # (3,) robust min corner
    hi: np.ndarray              # (3,) robust max corner
    n_points: int = 0
    bbox: tuple = (0.0, 0.0, 0.0, 0.0)   # [x1, y1, x2, y2] px in the source image
    crop: Optional[np.ndarray] = None    # RGB crop (H, W, 3) uint8
    track_id: int = -1
    views: int = 1              # how many captures saw it (multi-view fusion)

    @property
    def dims(self) -> np.ndarray:
        return self.hi - self.lo


@dataclass
class FreeSegment:
    x_min: float
    x_max: float
    usable_depth_m: float
    adjacent_to_group: bool     # False = an "empty part of the shelf" (rule 10)
    # True: the board top was seen over most of the segment. False: inferred - the cubby was
    # in view there (back panel / board edge seen) and no object stands in it, but the board
    # surface itself was seen too shallow to measure (camera close to or below the board).
    surface_seen: bool = True

    @property
    def width(self) -> float:
        return self.x_max - self.x_min


@dataclass
class Cubby:
    cubby_id: str               # "s2_left": level counted from the bottom, side facing the cabinet
    level: int                  # 1 = lowest board
    side: str                   # column name, see cubbies.column_names: "left"/"right", "left"/"middle"/
                                # "right", "c1".."cN" (4+ columns), "" = whole shelf (no split)
    board_z: float
    clearance_m: float          # to the underside of the board above (inf for the top board)
    x_min: float
    x_max: float
    depth_m: float
    column: int = 1             # 1 = leftmost, facing the cabinet
    members: List[Member] = field(default_factory=list)
    free: List[FreeSegment] = field(default_factory=list)

    @property
    def group_extent(self):
        """(x_min, x_max) of the members, or None if empty."""
        if not self.members:
            return None
        return (min(m.lo[0] for m in self.members), max(m.hi[0] for m in self.members))


@dataclass
class CabinetModel:
    boards: List[Board]
    cubbies: List[Cubby]
    width: float
    depth: float
    height: float
    unassigned: List[Member] = field(default_factory=list)  # detections outside any cubby
    notes: List[str] = field(default_factory=list)          # prior vs detection disagreements

    def cubby(self, cubby_id: str) -> Optional[Cubby]:
        return next((c for c in self.cubbies if c.cubby_id == cubby_id), None)
