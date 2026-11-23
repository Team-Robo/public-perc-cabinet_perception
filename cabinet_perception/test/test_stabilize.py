"""Live stabilisation: single bad frames keep the last valid pose and cubby structure."""
import numpy as np

from cabinet_perception.core.cubbies import free_box_height
from cabinet_perception.core.stabilize import Stabilizer, _pose, signature
from cabinet_perception.core.types import Board, CabinetModel, Cubby

SIZE = (0.96, 0.32, 0.85)


def test_outlier_keeps_last_valid_pose_and_size():
    st = Stabilizer()
    st.update_pose(_pose(0.70, 0.40, 0.84, 0.0), SIZE)
    T, size, state = st.update_pose(_pose(0.50, 0.40, 0.84, 0.0), (1.16, 0.32, 1.25))   # neighbour column + ceiling
    assert state == "outlier" and T[0, 3] == 0.70 and size[0] == 0.96 and size[2] == 0.85
    T, _, state = st.update_pose(_pose(0.71, 0.40, 0.84, 0.01), SIZE)
    assert state == "smoothed" and 0.70 < T[0, 3] < 0.71


def test_confirmed_change_is_adopted():
    st = Stabilizer()
    st.update_pose(_pose(0.70, 0.40, 0.84, 0.0), SIZE)
    states = [st.update_pose(_pose(0.90, 0.30, 0.84, 0.2), SIZE)[2] for _ in range(3)]
    assert states == ["outlier", "outlier", "moved"]
    assert abs(st.T[0, 3] - 0.90) < 1e-6


def test_alternating_outliers_never_confirm():
    st = Stabilizer()
    st.update_pose(_pose(0.70, 0.40, 0.84, 0.0), SIZE)
    for i in range(6):                                      # two different bad estimates, alternating
        x = 0.50 if i % 2 else 0.95
        assert st.update_pose(_pose(x, 0.40, 0.84, 0.0), SIZE)[2] == "outlier"


def _model(levels):
    """levels: [(board z, [split x...])], roof at 0.85."""
    boards = [Board(z=z, x_max=0.96, y_max=0.32) for z, _ in levels] + [Board(z=0.85, x_max=0.96, y_max=0.32)]
    cubbies = []
    for li, (z, splits) in enumerate(levels, start=1):
        edges = [0.0] + list(splits) + [0.96]
        for ci in range(len(edges) - 1):
            cubbies.append(Cubby(cubby_id=f"s{li}_{ci}", level=li, side=str(ci), board_z=z, x_min=edges[ci],
                                 x_max=edges[ci + 1], clearance_m=0.3, depth_m=0.32, column=ci + 1))
    return CabinetModel(boards=boards, cubbies=cubbies, width=0.96, depth=0.32, height=0.9)


def test_structure_vote_keeps_last_valid_until_majority():
    st = Stabilizer()
    four = _model([(0.08, [0.48]), (0.46, [0.48])])
    two = _model([(0.08, [0.48])])                          # upper level missed in this frame
    assert st.vote(four)[1] is None                          # first frame adopted
    assert st.vote(four)[1] is None
    sig, structure = st.vote(two)
    assert structure is not None                             # rebuild with the 4-cubby structure
    boards_z, splits = structure
    assert sorted(splits) == [1, 2] and 0.46 in boards_z
    for _ in range(3):
        st.vote(two)
    assert st.adopted == signature(two, st.p)                # a persistent change wins the majority


def test_free_box_height_is_clearance_capped():
    c = Cubby(cubby_id="s1_left", level=1, side="left", board_z=0.08, x_min=0, x_max=0.48, clearance_m=0.36, depth_m=0.32)
    assert free_box_height(c) == 0.36
    c.clearance_m = 1.2
    assert free_box_height(c) == 0.45
    c.clearance_m = float("inf")
    assert free_box_height(c) == 0.3
