"""Ray-cast a box-world cabinet into RGB-D + instance masks, for tests and demos.

Ground truth is exact, so tests can check board heights, cubby assignment and
free space without a robot. Not a renderer: flat colours, box objects only.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .geometry import Intrinsics
from .pipeline import Capture, Instance
from .prior import CabinetPrior


@dataclass
class SynthObject:
    label: str
    level: int                 # 1 = lowest board (index into boards_z)
    x: float                   # centre x in the cabinet frame
    size: Tuple[float, float, float] = (0.07, 0.07, 0.12)
    y: float = -1.0            # centre y; < 0 = centred in depth
    color: Tuple[int, int, int] = (200, 60, 60)


@dataclass
class SynthCabinet:
    prior: CabinetPrior
    objects: List[SynthObject] = field(default_factory=list)
    divider: bool = False      # physical vertical divider at prior.divider_x / midline
    dividers: Tuple[float, ...] = ()     # more physical dividers (x), full height
    roof: bool = True
    glass_levels: Tuple[int, ...] = ()   # levels whose board returns no depth

    def boxes(self):
        """[(lo, hi, kind, idx)], kind: 'cab' | 'obj', idx = object index."""
        p, t = self.prior, self.prior.board_thickness
        W, D, H = p.width, p.depth, p.height
        out = [((0, 0, 0), (t, D, H), "cab", -1), ((W - t, 0, 0), (W, D, H), "cab", -1),
               ((0, D - t, 0), (W, D, H), "cab", -1)]
        if self.roof:
            out.append(((0, 0, H - t), (W, D, H), "cab", -1))
        for i, z in enumerate(p.boards_z):
            if i + 1 in self.glass_levels:
                continue
            out.append(((t, 0, z - t), (W - t, D - t, z), "cab", -1))
        xs = list(self.dividers)
        if self.divider:
            xs.append(p.divider_x if p.divider_x is not None else W / 2)
        for xd in xs:
            out.append(((xd - t / 2, 0, 0), (xd + t / 2, D, H), "cab", -1))
        for i, o in enumerate(self.objects):
            z = p.boards_z[o.level - 1]
            y = o.y if o.y >= 0 else D / 2
            w, d, h = o.size
            out.append(((o.x - w / 2, y - d / 2, z), (o.x + w / 2, y + d / 2, z + h), "obj", i))
        return [(np.array(lo, float), np.array(hi, float), k, i) for lo, hi, k, i in out]

    def render(self, T_cab_cam: np.ndarray, K: Optional[Intrinsics] = None, noise: float = 0.002,
               seed: int = 0) -> Tuple[Capture, np.ndarray]:
        """Returns (capture with ground-truth instance masks, label image).
        Label image: -2 background/floor, -1 cabinet, i >= 0 object i."""
        K = K or Intrinsics(fx=390.0, fy=390.0, cx=320.0, cy=240.0, width=640, height=480)
        u, v = np.meshgrid(np.arange(K.width), np.arange(K.height))
        d_cam = np.stack([(u - K.cx) / K.fx, (v - K.cy) / K.fy, np.ones_like(u, float)], -1).reshape(-1, 3)
        d = d_cam @ T_cab_cam[:3, :3].T          # z_cam = 1 -> ray parameter t == optical depth
        o = T_cab_cam[:3, 3]
        best = np.full(len(d), np.inf)
        label = np.full(len(d), -2)
        with np.errstate(divide="ignore", invalid="ignore"):
            tf = -o[2] / d[:, 2]                  # floor z = 0
            hit = (d[:, 2] < 0) & (tf > 0)
            best[hit] = tf[hit]
            for lo, hi, kind, idx in self.boxes():
                t1, t2 = (lo - o) / d, (hi - o) / d
                tmin = np.nanmax(np.minimum(t1, t2), axis=1)
                tmax = np.nanmin(np.maximum(t1, t2), axis=1)
                h = (tmax >= np.maximum(tmin, 0)) & (tmin > 0) & (tmin < best)
                best[h] = tmin[h]
                label[h] = -1 if kind == "cab" else idx
        rng = np.random.default_rng(seed)
        depth = best.reshape(K.height, K.width).astype(np.float32)
        depth[~np.isfinite(depth)] = np.nan
        depth += rng.normal(0, 1, depth.shape).astype(np.float32) * noise * depth ** 2
        label = label.reshape(K.height, K.width)

        rgb = np.full((K.height, K.width, 3), 120, np.uint8)
        rgb[label == -1] = (170, 130, 90)
        insts = []
        for i, ob in enumerate(self.objects):
            m = label == i
            rgb[m] = ob.color
            if m.sum() > 50:
                insts.append(Instance(label=ob.label, confidence=0.9, mask=m))
        return Capture(depth_m=depth, K=K, T_cab_cam=T_cab_cam, rgb=rgb, instances=insts), label


def demo_cabinet() -> SynthCabinet:
    """4 shelves + floor board, left/right groups, one empty right side."""
    prior = CabinetPrior(width=0.80, depth=0.35, height=1.60, board_thickness=0.018,
                         boards_z=[0.08, 0.42, 0.76, 1.10, 1.44])
    objs = [
        SynthObject("apple", 1, 0.10, (0.08, 0.08, 0.08), color=(220, 40, 40)),
        SynthObject("banana", 1, 0.22, (0.15, 0.05, 0.05), color=(240, 220, 60)),
        SynthObject("cola_can", 1, 0.55, (0.07, 0.07, 0.12), color=(150, 0, 0)),
        SynthObject("water_bottle", 1, 0.68, (0.07, 0.07, 0.22), color=(160, 200, 255)),
        SynthObject("pringles_can", 2, 0.12, (0.08, 0.08, 0.24), color=(220, 120, 0)),
        SynthObject("crackers", 2, 0.26, (0.14, 0.05, 0.20), color=(200, 170, 100)),
        SynthObject("cornflakes_box", 2, 0.60, (0.19, 0.07, 0.28), color=(250, 200, 0)),
        SynthObject("sponge", 3, 0.15, (0.10, 0.07, 0.04), color=(80, 200, 80)),
        SynthObject("dish_soap", 3, 0.28, (0.07, 0.05, 0.20), color=(60, 160, 220)),
        SynthObject("dice", 4, 0.12, (0.05, 0.05, 0.05), color=(250, 250, 250)),
        SynthObject("tennis_ball", 4, 0.24, (0.07, 0.07, 0.07), color=(200, 255, 60)),
    ]
    return SynthCabinet(prior=prior, objects=objs)


def grid_cabinet() -> SynthCabinet:
    """3 levels x 3 columns, two physical dividers (x 0.40, 0.80); s3_middle is empty.
    The returned prior only says `columns: 3` - the dividers must be detected."""
    prior = CabinetPrior(width=1.20, depth=0.35, height=1.23, board_thickness=0.018,
                         boards_z=[0.06, 0.45, 0.84], columns=3)
    o = SynthObject
    objs = [
        o("apple", 1, 0.12, (0.08, 0.08, 0.08), color=(220, 40, 40)),
        o("orange", 1, 0.26, (0.08, 0.08, 0.08), color=(250, 140, 0)),
        o("cola_can", 1, 0.54, (0.07, 0.07, 0.12), color=(150, 0, 0)),
        o("water_bottle", 1, 0.66, (0.07, 0.07, 0.22), color=(160, 200, 255)),
        o("pringles_can", 1, 0.94, (0.08, 0.08, 0.24), color=(220, 120, 0)),
        o("crackers", 1, 1.08, (0.12, 0.05, 0.18), color=(200, 170, 100)),
        o("sponge", 2, 0.14, (0.10, 0.07, 0.04), color=(80, 200, 80)),
        o("dish_soap", 2, 0.28, (0.07, 0.05, 0.20), color=(60, 160, 220)),
        o("dice", 2, 0.54, (0.05, 0.05, 0.05), color=(250, 250, 250)),
        o("tennis_ball", 2, 0.66, (0.07, 0.07, 0.07), color=(200, 255, 60)),
        o("cornflakes_box", 2, 1.00, (0.19, 0.07, 0.28), color=(250, 200, 0)),
        o("shampoo", 3, 0.14, (0.06, 0.05, 0.20), color=(200, 120, 220)),
        o("soap_bar", 3, 0.28, (0.09, 0.06, 0.04), color=(240, 240, 200)),
        o("red_marker", 3, 0.94, (0.14, 0.02, 0.02), color=(230, 20, 20)),
        o("notebook", 3, 1.08, (0.15, 0.10, 0.02), color=(230, 60, 60)),
    ]
    return SynthCabinet(prior=prior, objects=objs, dividers=(0.40, 0.80))


SCENES = {"demo": demo_cabinet, "grid": grid_cabinet}
