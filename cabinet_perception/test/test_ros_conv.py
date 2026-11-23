"""v0 CabinetScene conversion: the grouping server must be able to read what we publish."""
import pytest

pytest.importorskip("object_grouping_interfaces")

from std_msgs.msg import Header  # noqa: E402

from cabinet_perception.core.geometry import look_at  # noqa: E402
from cabinet_perception.core.pipeline import CabinetPerception  # noqa: E402
from cabinet_perception.core.synthetic import demo_cabinet  # noqa: E402
from cabinet_perception.ros_conv import model_to_markers, model_to_scene  # noqa: E402


def test_scene_v0_contract():
    sc = demo_cabinet()
    perc = CabinetPerception(sc.prior)
    for tgt in [(0.4, 0.2, 1.1), (0.4, 0.2, 0.4)]:
        perc.add_capture(sc.render(look_at((0.4, -1.0, 1.35), tgt))[0])
    model = perc.build()
    msg = model_to_scene(model, Header(frame_id="cabinet"), {"apple": "fruits", "banana": "fruits"})

    assert [s.shelf_id for s in msg.shelves] == [1, 2, 3, 4, 5]
    assert abs(msg.shelves[1].height_m - 0.42) < 0.01
    ids = [g.group_id for g in msg.groups]
    assert "s3_right" not in ids                         # empty cubbies are not groups
    g = next(g for g in msg.groups if g.group_id == "s1_left")
    assert (g.shelf_id, g.side, g.grouping, g.category) == (1, "left", "category", "fruits")   # shared category
    assert [m.class_name for m in g.members] == ["apple", "banana"]       # sorted left -> right
    assert g.members[0].category == "fruits"
    assert len(g.member_crops) == len(g.members) and g.member_crops[0].width > 0
    assert g.member_crops[0].encoding == "rgb8"
    assert msg.shelves[2].free_width_m > 0.25            # s3_right empty

    markers = model_to_markers(model, Header(frame_id="cabinet"))
    assert len(markers.markers) > 20

    mixed = model_to_scene(model, Header(frame_id="cabinet"), {"apple": "fruits", "banana": "toys"})
    g = next(g for g in mixed.groups if g.group_id == "s1_left")
    assert (g.grouping, g.category) == ("", "")          # mixed group stays open


def test_grouping_side_can_read_it():
    """Round-trip through object_grouping's own msg -> core converter, if installed."""
    ros_conv = pytest.importorskip("object_grouping.ros_conv")
    sc = demo_cabinet()
    perc = CabinetPerception(sc.prior)
    perc.add_capture(sc.render(look_at((0.4, -1.0, 1.35), (0.4, 0.2, 0.7)))[0])
    scene = ros_conv.msg_to_scene(model_to_scene(perc.build(), Header(frame_id="cabinet")))
    assert scene.groups and all(g.members for g in scene.groups)
    assert any(m.image is not None for g in scene.groups for m in g.members)
