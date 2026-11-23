"""Multiple rows x columns: divider detection, setup-day dividers, virtual columns."""
import numpy as np

from cabinet_perception.core.cubbies import CubbyParams, _level_dividers, column_names, find_dividers
from cabinet_perception.core.geometry import look_at
from cabinet_perception.core.pipeline import CabinetPerception
from cabinet_perception.core.prior import CabinetPrior
from cabinet_perception.core.synthetic import SynthCabinet, SynthObject, grid_cabinet
from cabinet_perception.core.types import Member

GRID_EYE = (0.6, -1.0, 1.30)
GRID_TILTS = [(0.6, 0.2, 0.95), (0.6, 0.2, 0.30)]


def run(scene, prior=None, tilts=GRID_TILTS, eye=GRID_EYE, drop_masks=()):
    perc = CabinetPerception(prior or scene.prior)
    for tgt in tilts:
        cap, _ = scene.render(look_at(eye, tgt))
        cap.instances = [i for i in cap.instances if i.label not in drop_masks]
        perc.add_capture(cap)
    return perc, perc.build()


def table(model):
    return {c.cubby_id: sorted(m.label for m in c.members) for c in model.cubbies}


def test_column_names():
    assert column_names(1) == [""]
    assert column_names(2) == ["left", "right"]
    assert column_names(3) == ["left", "middle", "right"]
    assert column_names(5) == ["c1", "c2", "c3", "c4", "c5"]


def test_grid_3x3_dividers_detected():
    sc = grid_cabinet()
    perc, model = run(sc)
    assert table(model) == {
        "s1_left": ["apple", "orange"], "s1_middle": ["cola_can", "water_bottle"],
        "s1_right": ["crackers", "pringles_can"],
        "s2_left": ["dish_soap", "sponge"], "s2_middle": ["dice", "tennis_ball"],
        "s2_right": ["cornflakes_box"],
        "s3_left": ["shampoo", "soap_bar"], "s3_middle": [], "s3_right": ["notebook", "red_marker"],
    }
    for c in model.cubbies:
        assert abs(c.x_min - {1: 0.0, 2: 0.40, 3: 0.80}[c.column]) < 0.02, c.cubby_id
    assert not any("divider" in n for n in perc.notes)


def test_detection_wins_over_virtual_column_count():
    # prior still says the default 2 columns; the two physical dividers make it 3
    sc = grid_cabinet()
    prior = CabinetPrior(**{**sc.prior.__dict__, "columns": 2})
    _, model = run(sc, prior=prior)
    assert sorted({c.side for c in model.cubbies}) == ["left", "middle", "right"]


def test_hidden_divider_filled_from_column_count():
    # columns: 3, but only the divider at 0.80 is seen: the virtual line at 0.40 fills in
    m = [Member("a", 1, np.array([0.20, .1, .1]), np.array([0.15, .05, .05]), np.array([0.25, .15, .2])),
         Member("b", 1, np.array([0.55, .1, .1]), np.array([0.50, .05, .05]), np.array([0.60, .15, .2]))]
    divs = _level_dividers(1, [0.80], [0.40, 0.80], False, m, 1.20, CubbyParams(), [])
    assert np.allclose(divs, [0.40, 0.80])
    # columns: 2 (midline 0.60) but two dividers seen: they win, no extra split
    assert _level_dividers(1, [0.40, 0.80], [0.60], False, m, 1.20, CubbyParams(), []) == [0.40, 0.80]


def test_empty_middle_cubby_is_free_and_not_adjacent():
    _, model = run(grid_cabinet())
    s3m = model.cubby("s3_middle")
    assert s3m.free and max(s.width for s in s3m.free) > 0.25
    assert not any(s.adjacent_to_group for s in s3m.free)


def test_setup_day_dividers_used_when_not_seen():
    # no physical walls in the scene, but the setup-day prior lists two: they are
    # kept (and logged) even though a group straddles one of them
    prior = CabinetPrior(width=1.20, depth=0.35, height=0.60, boards_z=[0.05], dividers_x=[0.40, 0.80])
    sc = SynthCabinet(prior=prior, objects=[SynthObject("a", 1, 0.20), SynthObject("b", 1, 0.60),
                                            SynthObject("c", 1, 1.00)])
    perc, model = run(sc, tilts=[(0.6, 0.2, 0.1)], eye=(0.6, -0.9, 0.9))
    assert table(model) == {"s1_left": ["a"], "s1_middle": ["b"], "s1_right": ["c"]}
    assert sum("not seen, using prior" in n for n in perc.notes) == 2


def test_setup_day_divider_refined_by_detection():
    prior = CabinetPrior(width=1.20, depth=0.35, height=0.60, boards_z=[0.05], dividers_x=[0.43, 0.78])
    sc = SynthCabinet(prior=CabinetPrior(**{**prior.__dict__, "dividers_x": None}), dividers=(0.40, 0.80))
    _, model = run(sc, prior=prior, tilts=[(0.6, 0.2, 0.1)], eye=(0.6, -0.9, 0.9))
    xs = sorted(c.x_min for c in model.cubbies)[1:]
    assert np.allclose(xs, [0.40, 0.80], atol=0.02)


def test_per_level_dividers_from_prior():
    prior = CabinetPrior(width=1.20, depth=0.35, height=1.00, boards_z=[0.05, 0.50],
                         dividers_x={1: [0.60], 2: [0.40, 0.80]})
    assert prior.dividers(1) == ([0.60], True)
    assert prior.dividers(2) == ([0.40, 0.80], True)
    assert prior.dividers(3) == ([0.60], False)       # level not listed -> virtual columns (default 2)


def test_virtual_columns_without_dividers():
    prior = CabinetPrior(width=1.20, depth=0.35, height=0.60, boards_z=[0.05], columns=3)
    sc = SynthCabinet(prior=prior, objects=[SynthObject("a", 1, 0.15), SynthObject("b", 1, 0.25),
                                            SynthObject("c", 1, 0.60), SynthObject("d", 1, 1.05)])
    _, model = run(sc, tilts=[(0.6, 0.2, 0.1)], eye=(0.6, -0.9, 0.9))
    assert table(model) == {"s1_left": ["a", "b"], "s1_middle": ["c"], "s1_right": ["d"]}


def test_unknown_tall_box_is_not_a_divider():
    # an unmasked object (unknown to the detector) has vertical side faces too,
    # but they span its own depth, not the cabinet's
    prior = CabinetPrior(width=0.80, depth=0.35, height=0.60, boards_z=[0.05])
    sc = SynthCabinet(prior=prior, objects=[SynthObject("box", 1, 0.25, (0.10, 0.08, 0.40))])
    perc, _ = run(sc, tilts=[(0.4, 0.2, 0.2)], eye=(0.4, -0.9, 0.9), drop_masks=("box",))
    P, N = perc.points
    assert find_dividers(P, N, 0.05, 0.55, 0.80, 0.35, CubbyParams()) == []
