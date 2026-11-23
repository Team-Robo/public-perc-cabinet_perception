"""Localisation must not depend on knowing the camera height (handheld bench tests)."""
import numpy as np
import pytest

from cabinet_perception.core.geometry import look_at, normal_map, point_map, rotate_vectors, transform_points
from cabinet_perception.core.localize import estimate_cabinet
from cabinet_perception.core.pipeline import CabinetPerception, Capture
from cabinet_perception.core.prior import CabinetPrior
from cabinet_perception.core.synthetic import demo_cabinet

EYE = (0.4, -1.0, 1.35)


def cloud(sc, target=(0.4, 0.2, 0.7), dz=0.0):
    """Cloud in a frame whose z is off by dz (a wrong camera_height), plus the capture."""
    T = look_at(EYE, target)
    cap, _ = sc.render(T)
    shift = np.eye(4)
    shift[2, 3] = dz
    Pm = point_map(cap.depth_m, cap.K)
    P = transform_points(shift @ T, Pm).reshape(-1, 3)
    N = rotate_vectors(T, normal_map(Pm)).reshape(-1, 3)
    return P, N, cap, shift @ T


@pytest.mark.parametrize("dz", [-0.4, 0.0, 0.35])
def test_floor_found_whatever_the_camera_height(dz):
    P, N, _, T = cloud(demo_cabinet(), dz=dz)
    est = estimate_cabinet(P, N, robot_xy=T[:2, 3])
    assert est is not None and est.floor_seen
    assert abs(est.T_frame_cab[2, 3] - dz) < 0.02          # cabinet base = the (shifted) floor
    assert abs(est.width - 0.80) < 0.06


def test_no_floor_in_view_uses_bottom_board():
    P, N, _, T = cloud(demo_cabinet(), dz=0.2)
    keep = P[:, 2] > 0.2 + 0.03                              # floor cropped away
    est = estimate_cabinet(P[keep], N[keep], robot_xy=T[:2, 3])
    assert est is not None and not est.floor_seen
    # lowest board top is 0.08 above the floor; the base is put base_gap (0.08) below it
    assert abs(est.T_frame_cab[2, 3] - 0.2) < 0.02


def test_pipeline_on_localised_frame_finds_the_boards():
    sc = demo_cabinet()
    P, N, cap, T = cloud(sc, dz=0.5)                          # camera_height wrong by 0.5 m
    est = estimate_cabinet(P, N, robot_xy=T[:2, 3])
    perc = CabinetPerception(CabinetPrior(width=est.width, depth=est.depth, height=est.height + 0.05))
    perc.add_capture(Capture(cap.depth_m, cap.K, np.linalg.inv(est.T_frame_cab) @ T, cap.rgb, cap.instances))
    model = perc.build()
    z = [round(b.z, 2) for b in model.boards if b.z < 1.3]
    assert np.allclose(z, [0.08, 0.42, 0.76, 1.10], atol=0.02)


def test_no_floor_reports_bottom_board_and_full_depth():
    P, N, _, T = cloud(demo_cabinet(), dz=0.2)
    keep = P[:, 2] > 0.2 + 0.03
    est = estimate_cabinet(P[keep], N[keep], robot_xy=T[:2, 3])
    assert est.base_board_z == pytest.approx(0.08)
    assert abs(est.depth - 0.35) < 0.04                      # back panel, not just the board fronts


def _localised(sc):
    P, N, cap, T = cloud(sc, dz=0.5)
    est = estimate_cabinet(P, N, robot_xy=T[:2, 3])
    perc = CabinetPerception(CabinetPrior(width=est.width, depth=est.depth, height=est.height + 0.05))
    perc.add_capture(Capture(cap.depth_m, cap.K, np.linalg.inv(est.T_frame_cab) @ T, cap.rgb, cap.instances))
    return perc


def test_bottom_board_from_localisation_when_detection_misses_it():
    perc = _localised(demo_cabinet())
    near = np.abs(perc._P[:, 2] - 0.08) < 0.04                # bottom board seen edge-on: no points
    perc._P, perc._N = perc._P[~near], perc._N[~near]
    assert all(abs(b.z - 0.08) > 0.03 for b in perc.build().boards)
    perc.base_board_z = 0.08
    base = [b for b in perc.build().boards if b.source == "base"]
    assert len(base) == 1 and base[0].z == pytest.approx(0.08)


def test_detected_bottom_board_is_not_duplicated():
    perc = _localised(demo_cabinet())
    perc.base_board_z = 0.08
    boards = perc.build().boards
    assert sum(abs(b.z - 0.08) < 0.05 for b in boards) == 1
    assert all(b.source != "base" for b in boards)


def test_shallow_view_just_above_a_board_keeps_full_height_and_depth():
    """Camera a little above a board (its top seen as a thin front strip), floor out of
    view: height and depth must come from undersides / back panel, not collapse to the strip."""
    sc = demo_cabinet()
    T = look_at((0.4, -0.9, 0.44), (0.4, 0.2, 0.69))          # 2 cm above the 0.42 board, looking up
    cap, _ = sc.render(T)
    Pm = point_map(cap.depth_m, cap.K)
    P = transform_points(T, Pm).reshape(-1, 3)
    N = rotate_vectors(T, normal_map(Pm)).reshape(-1, 3)
    keep = np.isfinite(P).all(1) & (P[:, 2] > 0.3) & (np.linalg.norm(P[:, :2] - T[:2, 3], axis=1) < 2.5)
    est = estimate_cabinet(P[keep], N[keep], robot_xy=T[:2, 3])
    assert est is not None and not est.floor_seen
    top = est.T_frame_cab[2, 3] + est.height                   # cabinet top in the world frame
    assert top > 1.3                                            # not the 0.42 board strip
    assert abs(est.depth - 0.35) < 0.05


def test_known_width_snaps_sides_to_walls_when_tops_are_partly_seen():
    """Only the right half of the board tops is seen (angled view): without a width the cabinet
    shrinks to that half; with the setup-day width the sides snap to the side walls."""
    P, N, _, T = cloud(demo_cabinet())
    near = np.isfinite(P).all(1) & (np.linalg.norm(P[:, :2] - T[:2, 3], axis=1) < 2.0)   # node's range crop
    keep = near & ~((N[:, 2] > 0.9) & (P[:, 0] < 0.40) & (P[:, 2] > 0.05))
    est = estimate_cabinet(P[keep], N[keep], robot_xy=T[:2, 3])
    assert est.width < 0.6
    est = estimate_cabinet(P[keep], N[keep], robot_xy=T[:2, 3], width_hint=0.80)
    assert est.width == pytest.approx(0.80)
    left = est.T_frame_cab[:3, 3] @ est.T_frame_cab[:3, 0]                   # cabinet left side, world x
    assert abs(left - 0.0) < 0.03
