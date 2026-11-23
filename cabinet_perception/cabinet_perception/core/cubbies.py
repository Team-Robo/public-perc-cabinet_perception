"""Boards + members + raw points -> cubbies (level x column) with free segments.

Rows are the levels between boards. Columns per level, like boards (the prior
says which, detection says where):
  - setup-day dividers (prior.dividers_x): physical walls; a detected divider
    within `divider_match_tol` refines each, an unseen one keeps the prior x.
  - otherwise every physical divider seen on that level (peaks of the x-histogram
    of +-x-normal points that span the level height and the cabinet depth).
  - fewer seen than `prior.columns` implies (none, or one hidden): the virtual
    split lines of `columns` equal columns (2 = the midline) that are not near a
    seen divider, each moved to the nearest gap between members when one
    straddles it (design note §2). No gap -> that split is dropped; one column
    = whole-shelf cubby (side "").
Cubby ids: s<level>_<side>, sides from column_names(): left/right for two
columns (the rulebook's "each side of the shelves"), left/middle/right for
three, c1..cN beyond that, "" for a whole shelf.

Free space: every occupied point above a board (not just detections, so
unknown objects and decorations count) is projected onto x. A bin is free only
if the board surface itself was observed there and nothing stands on it;
unobserved bins are unknown, never free.
"""
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from .types import Board, CabinetModel, Cubby, FreeSegment, Member


@dataclass
class CubbyParams:
    split_sides: bool = True
    min_clearance: float = 0.08     # boards with less space above are not shelves
    level_tol: float = 0.03         # member bottom may sit this far below the board top
    divider_normal_x: float = 0.9
    divider_min_points: int = 60
    divider_z_coverage: float = 0.5
    divider_y_coverage: float = 0.5   # a divider spans the cabinet depth; a box side does not
    divider_match_tol: float = 0.06   # detected vs setup-day divider that count as the same one
    min_column_width: float = 0.12    # narrowest column (also: divider distance from the side walls)
    min_gap: float = 0.03           # gap between member groups that justifies a split
    max_split_shift: float = 0.25   # fraction of width the split may move from the prior
    x_bin: float = 0.01
    occ_margin: float = 0.02        # clearance kept around occupied bins
    obj_min_height: float = 0.015   # points this far above the board are "something standing"
    max_obj_height: float = 0.45    # only look this far up for occupancy (tall shelves)
    wall_margin: float = 0.03       # side walls / divider faces
    back_margin: float = 0.05       # back panel + its depth noise
    min_free: float = 0.06          # narrowest free segment worth reporting
    min_view_points: int = 3        # points of the cubby volume in a column for it to count as in view
    back_clear: float = 0.03        # occupancy ignores points this close in front of the back panel
    back_min_points: int = 40       # back-panel points needed to locate it (else depth - back_margin)
    adjacent_tol: float = 0.08      # free segment within this of a group = "next to it"
    top_is_shelf: bool = False      # treat the cabinet roof as a shelf
    roof_tol: float = 0.04


def _intervals(mask: np.ndarray, x0: float, bin_w: float) -> List[Tuple[float, float]]:
    out, start = [], None
    for i, v in enumerate(np.append(mask, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((x0 + start * bin_w, x0 + i * bin_w))
            start = None
    return out


def column_names(n: int) -> List[str]:
    if n <= 1:
        return [""]
    if n == 2:
        return ["left", "right"]
    if n == 3:
        return ["left", "middle", "right"]
    return [f"c{i}" for i in range(1, n + 1)]


def find_dividers(P: np.ndarray, N: Optional[np.ndarray], z0: float, z1: float, width: float,
                  depth: float, p: CubbyParams) -> List[float]:
    """x of every vertical divider seen between z0 and z1, left to right."""
    if N is None or len(P) == 0:
        return []
    lo, hi = p.min_column_width, width - p.min_column_width
    if hi <= lo:
        return []
    sel = (np.abs(N[:, 0]) > p.divider_normal_x) & (P[:, 2] > z0 + 0.02) & (P[:, 2] < z1 - 0.02) \
        & (P[:, 0] > lo) & (P[:, 0] < hi)
    x, y, z = P[sel, 0], P[sel, 1], P[sel, 2]
    if len(x) < p.divider_min_points:
        return []
    edges = np.arange(lo, hi + 0.02, 0.02)
    h, _ = np.histogram(x, edges)
    z_edges = np.arange(z0, z1 + 1e-9, 0.02)
    if len(z_edges) < 2:
        return []
    out: List[float] = []
    for i in sorted(range(len(h)), key=lambda i: -h[i]):     # strongest first, then non-max suppression
        if h[i] < p.divider_min_points:
            break
        xc = 0.5 * (edges[i] + edges[i + 1])
        if any(abs(xc - d) < p.min_column_width for d in out):
            continue
        band = np.abs(x - xc) < 0.02
        zh, _ = np.histogram(z[band], z_edges)
        if np.mean(zh > 0) < p.divider_z_coverage:
            continue
        y0, y1 = np.percentile(y[band], [5, 95])
        if y1 - y0 < p.divider_y_coverage * depth:
            continue
        out.append(float(np.median(x[band])))
    return sorted(out)


def _level_dividers(level: int, seen: List[float], prior: List[float], physical: bool,
                    members: List[Member], width: float, p: CubbyParams, notes: List[str]) -> List[float]:
    if physical:
        out = []
        for xp in prior:
            near = [d for d in seen if abs(d - xp) <= p.divider_match_tol]
            if near:
                out.append(min(near, key=lambda d: abs(d - xp)))
            else:
                out.append(xp)
                notes.append(f"level {level}: divider x={xp:.3f} not seen, using prior")
        for d in seen:
            if all(abs(d - xp) > p.divider_match_tol for xp in prior):
                notes.append(f"level {level}: extra divider candidate x={d:.3f} ignored (not in prior)")
        return sorted(out)
    # Seen dividers are real. If fewer are seen than the column count implies (one
    # hidden behind an object or seen edge-on), fill in the virtual lines that are
    # not near a seen one, each moved into a member gap.
    out = list(seen)
    if len(seen) < len(prior):
        half = 0.5 * width / (len(prior) + 1)
        for xp in prior:
            if any(abs(xp - d) < half for d in seen):
                continue
            x = _split_x(members, xp, width, p)
            if x is not None and all(abs(x - o) >= max(p.min_gap, p.min_column_width if seen else 0) for o in out):
                out.append(x)
    return sorted(out)


def _split_x(members: List[Member], prior_x: float, width: float, p: CubbyParams) -> Optional[float]:
    """Prior split, moved into a member gap if a member straddles it; None = no split."""
    straddle = [m for m in members if m.lo[0] < prior_x < m.hi[0]]
    if not straddle:
        return prior_x
    ivs = sorted((m.lo[0], m.hi[0]) for m in members)
    merged = [list(ivs[0])]
    for a, b in ivs[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    gaps = [(merged[i][1], merged[i + 1][0]) for i in range(len(merged) - 1)
            if merged[i + 1][0] - merged[i][1] >= p.min_gap]
    gaps = [g for g in gaps if abs(0.5 * (g[0] + g[1]) - prior_x) <= p.max_split_shift * width]
    if not gaps:
        return None
    g = min(gaps, key=lambda g: abs(0.5 * (g[0] + g[1]) - prior_x))
    return 0.5 * (g[0] + g[1])


def _back_panel_y(P: np.ndarray, N: Optional[np.ndarray], x_lo: float, x_hi: float, z_lo: float, z_hi: float,
                  depth: float, p: CubbyParams) -> Optional[float]:
    """y of the back panel behind a cubby: front of the deepest dense band of surfaces facing the
    camera (normal along -y). None without normals or when too little of it is seen."""
    if N is None or z_hi <= z_lo:
        return None
    sel = (P[:, 0] >= x_lo) & (P[:, 0] < x_hi) & (P[:, 2] > z_lo) & (P[:, 2] < z_hi) \
        & (P[:, 1] > 0.05) & (P[:, 1] < depth + 0.10) & (np.abs(N[:, 1]) > 0.8)
    if sel.sum() < p.back_min_points:
        return None
    h, e = np.histogram(P[sel, 1], np.arange(0.05, depth + 0.11, 0.01))
    dense = np.nonzero(h >= max(p.back_min_points // 4, 0.2 * h.max()))[0]
    if not len(dense):
        return None
    i = dense[-1]
    while i > 0 and h[i - 1] >= 0.2 * h.max():         # back to the front of that band
        i -= 1
    return float(e[i])


def _free_segments(P: np.ndarray, N: Optional[np.ndarray], board: Board, clearance: float,
                   x_lo: float, x_hi: float, depth: float, members: List[Member],
                   p: CubbyParams) -> List[FreeSegment]:
    x_lo, x_hi = x_lo + p.wall_margin, x_hi - p.wall_margin
    n = int(np.ceil((x_hi - x_lo) / p.x_bin))
    if n <= 0 or len(P) == 0:
        return []
    inside = (P[:, 0] >= x_lo) & (P[:, 0] < x_hi) & (P[:, 1] > -0.02) & (P[:, 1] < depth - p.back_margin)
    top = board.z + min(clearance - 0.02, p.max_obj_height)
    # Occupancy stops just in front of this cubby's back panel, found in the data (the deepest
    # dense surface facing the camera), not at `depth - back_margin`: the estimated depth varies by a
    # few cm between frames, and when it was too deep the panel itself counted as an object filling
    # the whole cubby width, so free space came and went.
    back_y = _back_panel_y(P, N, x_lo, x_hi, board.z + 0.02, board.z + clearance - 0.02, depth, p)
    y_cut = back_y - p.back_clear if back_y is not None else depth - p.back_margin
    in_front = (P[:, 0] >= x_lo) & (P[:, 0] < x_hi) & (P[:, 1] > -0.02) & (P[:, 1] < y_cut)
    occ_pts = P[in_front & (P[:, 2] > board.z + p.obj_min_height) & (P[:, 2] < top)]
    surf = inside & (np.abs(P[:, 2] - board.z) < 0.01)
    if N is not None:
        surf &= N[:, 2] > 0.8
    surf_pts = P[surf]

    def bins(x):
        return np.clip(((x - x_lo) / p.x_bin).astype(int), 0, n - 1)

    occ = np.zeros(n, bool)
    np.add.at(occ, bins(occ_pts[:, 0]), True)
    for m in members:   # detections occupy their full extent even where unseen
        occ[bins(np.array([m.lo[0]]))[0]:bins(np.array([m.hi[0]]))[0] + 1] = True
    seen = np.zeros(n, int)
    np.add.at(seen, bins(surf_pts[:, 0]), 1)
    # In view: any point of the cubby volume in that column (board top or edge, back panel,
    # objects). Free = in view and nothing standing there, at any depth - the board surface
    # need not be seen (from close to a board it is seen too shallow to give points).
    vol = (P[:, 0] >= x_lo) & (P[:, 0] < x_hi) & (P[:, 1] > -0.02) & (P[:, 1] < depth + 0.03) \
        & (P[:, 2] > board.z - 0.03) & (P[:, 2] < board.z + clearance)
    in_view = np.zeros(n, int)
    np.add.at(in_view, bins(P[vol, 0]), 1)
    k = int(round(p.occ_margin / p.x_bin))
    if k > 0:
        occ = np.convolve(occ.astype(int), np.ones(2 * k + 1), mode="same") > 0
    free = (in_view >= p.min_view_points) & ~occ

    extent = None
    if members:
        extent = (min(m.lo[0] for m in members), max(m.hi[0] for m in members))
    out = []
    for a, b in _intervals(free, x_lo, p.x_bin):
        if b - a < p.min_free:
            continue
        i0, i1 = bins(np.array([a]))[0], bins(np.array([b - 1e-6]))[0] + 1
        surface = float(np.mean(seen[i0:i1] >= 2)) >= 0.5
        # nothing stands in these columns at any depth: the whole cubby depth is usable
        usable = float(depth - p.back_margin)
        adj = extent is not None and (abs(a - extent[1]) <= p.adjacent_tol or abs(extent[0] - b) <= p.adjacent_tol)
        out.append(FreeSegment(float(a), float(b), usable, bool(adj), surface))
    return out


def free_box_height(cubby: Cubby, cap: float = 0.45) -> float:
    """Height of a free-space box on this cubby's board: the clearance to the board above
    (the space an item may fill), capped; 0.3 m when there is no board above."""
    h = cubby.clearance_m if np.isfinite(cubby.clearance_m) else 0.3
    return float(max(0.0, min(h, cap)))


def build_cubbies(boards: List[Board], members: List[Member], P: np.ndarray, N: Optional[np.ndarray],
                  width: float, depth: float, height: float,
                  prior_dividers: Optional[Callable[[int], Tuple[List[float], bool]]] = None,
                  thickness: float = 0.018, p: Optional[CubbyParams] = None) -> CabinetModel:
    """prior_dividers(level) -> (x list, physical?), e.g. CabinetPrior.dividers; None = midline."""
    p = p or CubbyParams()
    prior_dividers = prior_dividers or (lambda level: ([0.5 * width], False))
    notes: List[str] = []
    boards = sorted(boards, key=lambda b: b.z)
    levels = []   # (board, clearance)
    for i, b in enumerate(boards):
        if i + 1 < len(boards):
            clr = boards[i + 1].z - thickness - b.z
        elif b.z >= height - p.roof_tol:
            if not p.top_is_shelf:
                continue
            clr = float("inf")
        else:
            clr = height - thickness - b.z
        if clr >= p.min_clearance:
            levels.append((b, clr))

    by_level = {i: [] for i in range(len(levels))}
    unassigned = []
    for m in members:
        if not (-0.05 < m.centroid[0] < width + 0.05 and -0.05 < m.centroid[1] < depth + 0.05):
            unassigned.append(m)
            continue
        cand = [i for i, (b, clr) in enumerate(levels) if b.z - p.level_tol <= m.lo[2] < b.z + clr]
        if cand:
            by_level[cand[-1]].append(m)
        else:
            unassigned.append(m)

    cubbies = []
    for li, (b, clr) in enumerate(levels):
        level = li + 1
        mem = by_level[li]
        z_top = b.z + (clr if np.isfinite(clr) else 0.4)
        divs: List[float] = []
        if p.split_sides:
            prior_x, physical = prior_dividers(level)
            seen = find_dividers(P, N, b.z, z_top, width, depth, p)
            divs = _level_dividers(level, seen, prior_x, physical, mem, width, p, notes)
        bounds = [0.0] + divs + [width]
        names = column_names(len(bounds) - 1)
        for k, side in enumerate(names):
            x0, x1 = bounds[k], bounds[k + 1]
            # outermost columns also take members whose centroid pokes past the side walls
            xa = -np.inf if k == 0 else x0
            xb = np.inf if k == len(names) - 1 else x1
            cm = [m for m in mem if xa <= m.centroid[0] < xb]
            cid = f"s{level}_{side}" if side else f"s{level}"
            c = Cubby(cubby_id=cid, level=level, side=side, board_z=b.z, clearance_m=float(clr),
                      x_min=x0, x_max=x1, depth_m=depth, column=k + 1, members=cm)
            c.free = _free_segments(P, N, b, clr, x0, x1, depth, cm, p)
            cubbies.append(c)
    return CabinetModel(boards=boards, cubbies=cubbies, width=width, depth=depth, height=height,
                        unassigned=unassigned, notes=notes)
