"""ROS-free tests on the synthetic cabinet (exact ground truth)."""
import numpy as np
import pytest

from cabinet_perception.core.boards import detect_boards, fuse_with_prior
from cabinet_perception.core.cubbies import CubbyParams, build_cubbies
from cabinet_perception.core.geometry import look_at, normal_map, point_map, rotate_vectors
from cabinet_perception.core.members import fuse_members
from cabinet_perception.core.pipeline import CabinetPerception
from cabinet_perception.core.prior import CabinetPrior
from cabinet_perception.core.synthetic import SynthCabinet, SynthObject, demo_cabinet
from cabinet_perception.core.types import Board, Member

EYE = (0.4, -1.0, 1.35)                  # head camera ~1 m in front of the cabinet
TILTS = [(0.4, 0.2, 1.1), (0.4, 0.2, 0.4)]


def run(scene, prior=None, tilts=TILTS, eye=EYE):
    perc = CabinetPerception(prior or scene.prior)
    for tgt in tilts:
        cap, _ = scene.render(look_at(eye, tgt))
        perc.add_capture(cap)
    return perc, perc.build()


def detect_only(prior):
    return CabinetPrior(**{**prior.__dict__, "boards_z": []})


def labels(cubby):
    return sorted(m.label for m in cubby.members)


def test_normals_of_a_floor_point_up():
    sc = SynthCabinet(prior=CabinetPrior(boards_z=[]), roof=False)
    sc.prior.width = 0.0                      # no cabinet in view, just the floor
    T = look_at((0, -1, 1.2), (0, 1, 0))
    cap, _ = sc.render(T, noise=0.0)
    N = rotate_vectors(T, normal_map(point_map(cap.depth_m, cap.K)))
    nz = N[..., 2][np.isfinite(N[..., 2])]
    assert np.median(nz) > 0.99


def test_boards_detected_without_prior():
    sc = demo_cabinet()
    perc, model = run(sc, prior=detect_only(sc.prior))
    z = [b.z for b in model.boards if b.z < 1.3]
    truth = [0.08, 0.42, 0.76, 1.10]
    assert len(z) == len(truth)
    assert np.allclose(z, truth, atol=0.01)


def test_board_above_camera_from_underside():
    # Camera low (0.9 m) looking up: boards above it only show their underside.
    sc = demo_cabinet()
    perc, model = run(sc, prior=detect_only(sc.prior), eye=(0.4, -1.2, 0.9), tilts=[(0.4, 0.2, 1.3)])
    b = [b for b in model.boards if abs(b.z - 1.10) < 0.02]
    assert b and b[0].source == "underside"


def test_prior_fills_unseen_board_and_logs_it():
    sc = demo_cabinet()
    sc.glass_levels = (3,)                        # glass board: no depth return
    perc, model = run(sc)
    assert [round(b.z, 2) for b in model.boards] == [0.08, 0.42, 0.76, 1.10, 1.44]
    assert next(b for b in model.boards if abs(b.z - 0.76) < 0.01).source == "prior"
    assert any("0.760" in n for n in perc.notes)


def test_members_land_in_the_right_cubby():
    sc = demo_cabinet()
    _, model = run(sc)
    c = {cb.cubby_id: labels(cb) for cb in model.cubbies}
    assert c["s1_left"] == ["apple", "banana"]
    assert c["s1_right"] == ["cola_can", "water_bottle"]
    assert c["s2_left"] == ["crackers", "pringles_can"]
    assert c["s2_right"] == ["cornflakes_box"]
    assert c["s3_left"] == ["dish_soap", "sponge"]
    assert c["s3_right"] == []
    assert c["s4_left"] == ["dice", "tennis_ball"]
    assert not model.unassigned


def test_empty_side_is_reported_as_free_and_not_adjacent():
    sc = demo_cabinet()
    _, model = run(sc)
    s3r = model.cubby("s3_right")
    assert s3r.free and max(s.width for s in s3r.free) > 0.25
    assert not any(s.adjacent_to_group for s in s3r.free)


def test_free_segment_next_to_group_is_adjacent():
    prior = CabinetPrior(width=0.80, depth=0.35, height=0.60, boards_z=[0.05])
    sc = SynthCabinet(prior=prior, objects=[SynthObject("apple", 1, 0.10, (0.08, 0.08, 0.08)),
                                            SynthObject("pear", 1, 0.20, (0.08, 0.08, 0.10))])
    _, model = run(sc, tilts=[(0.4, 0.2, 0.1)], eye=(0.4, -0.8, 0.9))
    left = model.cubby("s1_left")
    seg = max(left.free, key=lambda s: s.width)
    assert seg.adjacent_to_group and seg.x_min > 0.24 - 0.01
    # occupied extent of the group is never reported free
    assert all(s.x_min >= 0.24 or s.x_max <= 0.06 for s in left.free)


def test_split_moves_into_gap_when_group_straddles_midline():
    prior = CabinetPrior(width=0.80, depth=0.35, height=0.60, boards_z=[0.05])
    objs = [SynthObject("a", 1, 0.30, (0.08, 0.08, 0.1)), SynthObject("b", 1, 0.40, (0.08, 0.08, 0.1)),
            SynthObject("c", 1, 0.62, (0.08, 0.08, 0.1))]
    sc = SynthCabinet(prior=prior, objects=objs)
    _, model = run(sc, tilts=[(0.4, 0.2, 0.1)], eye=(0.4, -0.8, 0.9))
    assert labels(model.cubby("s1_left")) == ["a", "b"]
    assert labels(model.cubby("s1_right")) == ["c"]
    assert 0.44 < model.cubby("s1_left").x_max < 0.58


def test_no_gap_gives_whole_shelf_cubby():
    m = [Member("a", 1, np.array([0.35, .1, .1]), np.array([0.30, .05, .05]), np.array([0.45, .15, .2])),
         Member("b", 1, np.array([0.50, .1, .1]), np.array([0.45, .05, .05]), np.array([0.55, .15, .2]))]
    model = build_cubbies([Board(0.05, 0, .8, 0, .35)], m, np.zeros((0, 3)), None, 0.8, 0.35, 0.6)
    assert [c.cubby_id for c in model.cubbies] == ["s1"]
    assert model.cubbies[0].side == ""


def test_physical_divider_is_used():
    prior = CabinetPrior(width=0.80, depth=0.35, height=0.60, boards_z=[0.05], divider_x=0.30)
    sc = SynthCabinet(prior=prior, objects=[SynthObject("a", 1, 0.20), SynthObject("b", 1, 0.45)], divider=True)
    detect = CabinetPrior(**{**prior.__dict__, "divider_x": None})     # don't hand it the answer
    _, model = run(sc, prior=detect, tilts=[(0.4, 0.2, 0.1)], eye=(0.4, -0.8, 0.9))
    assert abs(model.cubby("s1_left").x_max - 0.30) < 0.02


def test_multiview_dedupes_members_but_not_stacked_ones():
    def m(x, z):
        c = np.array([x, 0.15, z])
        return Member("o", 0.9, c, c - 0.04, c + 0.04)
    fused = fuse_members([m(0.10, 0.12), m(0.11, 0.13), m(0.10, 0.50)])
    assert len(fused) == 2
    assert max(f.views for f in fused) == 2


def test_fuse_with_prior_reports_disagreement():
    boards, notes = fuse_with_prior([Board(0.10), Board(0.47)], [0.10, 0.42], 0.8, 0.35)
    assert [b.source for b in boards] == ["top", "prior"]
    assert any("5.0 cm off" in n for n in notes)


@pytest.mark.parametrize("bad", [CubbyParams(split_sides=False)])
def test_split_can_be_disabled(bad):
    m = [Member("a", 1, np.array([0.1, .1, .1]), np.array([0.05, .05, .05]), np.array([0.15, .15, .2]))]
    model = build_cubbies([Board(0.05, 0, .8, 0, .35)], m, np.zeros((0, 3)), None, 0.8, 0.35, 0.6, p=bad)
    assert [c.cubby_id for c in model.cubbies] == ["s1"]


def test_localize_recovers_cabinet_pose_without_prior():
    from cabinet_perception.core.geometry import make_transform, transform_points
    from cabinet_perception.core.localize import estimate_cabinet
    sc = demo_cabinet()
    yaw = np.radians(25)
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    T_base_cab = make_transform(R, (1.2, 0.3, 0.0))          # cabinet pose in a robot base frame
    T_cab_cam = look_at(EYE, (0.4, 0.2, 0.7))
    cap, _ = sc.render(T_cab_cam)
    T_base_cam = T_base_cab @ T_cab_cam
    P = transform_points(T_base_cam, point_map(cap.depth_m, cap.K)).reshape(-1, 3)
    N = rotate_vectors(T_base_cam, normal_map(point_map(cap.depth_m, cap.K))).reshape(-1, 3)
    robot = transform_points(T_base_cam, np.zeros(3))[:2]
    est = estimate_cabinet(P, N, robot_xy=robot)
    assert est is not None
    assert abs(np.degrees(np.angle(np.exp(1j * (est.yaw - (yaw - np.pi / 2)))))) < 2
    assert abs(est.width - 0.80) < 0.06 and abs(est.depth - 0.35) < 0.06
    assert np.linalg.norm(est.T_frame_cab[:2, 3] - T_base_cab[:2, 3]) < 0.05


def _grazing_scene():
    prior = CabinetPrior(width=0.80, depth=0.35, height=0.60, boards_z=[0.05])
    objs = [SynthObject("apple", 1, 0.10, (0.08, 0.08, 0.08)), SynthObject("box", 1, 0.62, (0.10, 0.10, 0.12))]
    return SynthCabinet(prior=prior, objects=objs)


GRAZING_EYE, GRAZING_TGT = (0.4, -0.8, 0.08), (0.4, 0.2, 0.12)   # 3 cm above the board


def test_grazing_view_free_space_from_occupancy_not_board_surface():
    """Camera just above the board: the board top gives (almost) no points, but the cubby is in
    view and nothing stands next to the objects, so that floor is free (inferred)."""
    _, model = run(_grazing_scene(), tilts=[GRAZING_TGT], eye=GRAZING_EYE)
    left, right = model.cubby("s1_left"), model.cubby("s1_right")
    seg = max(left.free, key=lambda s: s.width)
    assert seg.width > 0.15 and seg.x_min > 0.14                  # right of the apple (0.06-0.14)
    assert all(s.x_max <= 0.06 or s.x_min >= 0.14 for s in left.free)
    assert all(s.x_max <= 0.57 or s.x_min >= 0.67 for s in right.free)   # never on/under the box
    assert all(s.usable_depth_m > 0.25 for s in left.free + right.free)


def test_cubby_out_of_view_is_not_free():
    sc = _grazing_scene()
    cap, _ = sc.render(look_at(GRAZING_EYE, GRAZING_TGT))
    cap.depth_m[:, cap.depth_m.shape[1] // 2:] = np.nan               # right half of the image unseen
    perc = CabinetPerception(sc.prior)
    perc.add_capture(cap)
    model = perc.build()
    assert not model.cubby("s1_right").free or max(s.x_max for s in model.cubby("s1_right").free) < 0.55


def test_back_panel_is_not_occupancy_when_depth_is_overestimated():
    """Estimated cabinet depth a few cm too deep: the back panel must still not count as an
    object (it filled the whole cubby width and free space flickered with the depth estimate)."""
    sc = _grazing_scene()
    deep = CabinetPrior(**{**sc.prior.__dict__, "depth": sc.prior.depth + 0.08})   # old cut (depth - 5 cm) behind the panel
    _, model = run(sc, prior=deep, tilts=[(0.4, 0.2, 0.1)], eye=(0.4, -0.8, 0.45))
    left = model.cubby("s1_left")
    assert left.free and max(s.width for s in left.free) > 0.15
