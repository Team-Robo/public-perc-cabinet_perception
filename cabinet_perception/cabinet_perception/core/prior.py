"""Setup-day cabinet prior (design note §2, option A), loaded from YAML."""
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import yaml


@dataclass
class CabinetPrior:
    width: float = 0.80
    depth: float = 0.35
    height: float = 1.60
    board_thickness: float = 0.018
    boards_z: List[float] = field(default_factory=list)   # top surfaces; empty = detect only
    # Physical vertical dividers (x of their centre), measured on the setup day:
    # one list for every level, or {level: [x, ...]} when levels differ. None =
    # not measured: detect them per level, else `columns` equal virtual columns.
    dividers_x: Union[None, List[float], Dict[int, List[float]]] = None
    divider_x: Optional[float] = None                      # legacy single divider (= dividers_x: [x])
    columns: int = 2                                       # virtual columns per level, 1 = no split
    top_is_shelf: bool = False
    frame_id: str = "cabinet"
    parent_frame: str = "map"
    pose_xyz: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    pose_rpy: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def dividers(self, level: int) -> Tuple[List[float], bool]:
        """(divider x positions, physical?) for a level, counted from 1 at the bottom.
        Physical = measured on the setup day: detection only refines them. Otherwise
        they are virtual split lines, used when no divider is seen on that level."""
        d = self.dividers_x
        if isinstance(d, dict):
            if level in d or str(level) in d:
                return sorted(float(x) for x in (d.get(level, d.get(str(level))) or [])), True
        elif d is not None:
            return sorted(float(x) for x in d), True
        elif self.divider_x is not None:
            return [float(self.divider_x)], True
        n = max(1, int(self.columns))
        return [self.width * k / n for k in range(1, n)], False

    @classmethod
    def load(cls, path: str) -> "CabinetPrior":
        with open(os.path.expanduser(path)) as f:
            doc = (yaml.safe_load(f) or {}).get("cabinet", {})
        pose = doc.pop("pose", {}) or {}
        prior = cls(**{k: v for k, v in doc.items() if k in cls.__dataclass_fields__})
        prior.pose_xyz = list(pose.get("xyz", prior.pose_xyz))
        prior.pose_rpy = list(pose.get("rpy", prior.pose_rpy))
        return prior
