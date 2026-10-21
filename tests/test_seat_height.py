import importlib.util

import numpy as np
import pytest

from arena_assets.impl import Face
from arena_assets.impl.build.ObjectDatabaseBuilder import carry_slots, place_slot
from arena_assets.utils.affordance import largest_seat, surface_z
from arena_assets.utils.geom import BoundingBox

needs_geometry = pytest.mark.skipif(importlib.util.find_spec("arena_simulation_setup") is None, reason="arena_simulation_setup comes with the Arena workspace")

BOX_TRIANGLES = np.array([[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4], [2, 6, 3], [3, 6, 7], [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]])


def source_slot(value: dict):
    from arena_simulation_setup.utils.affordance import Slot

    return Slot.parse(value)


def mesh(*boxes: tuple[float, float, float, float, float, float], flip: bool = False) -> tuple[np.ndarray, np.ndarray]:
    points, triangles = [], []
    for index, (x0, x1, y0, y1, z0, z1) in enumerate(boxes):
        points += [(x, y, z) for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)]
        triangles.append(BOX_TRIANGLES + 8 * index)
    faces = np.concatenate(triangles)
    return np.array(points, np.float64), faces[:, ::-1] if flip else faces


def legs(x0: float, x1: float, y0: float, y1: float, top: float) -> list[tuple[float, float, float, float, float, float]]:
    return [(x, x + 0.04, y, y + 0.04, 0.0, top) for x in (x0, x1 - 0.04) for y in (y0, y1 - 0.04)]


CHAIR = mesh(
    *legs(-0.25, 0.25, -0.25, 0.25, 0.40),
    (-0.25, 0.25, -0.25, 0.25, 0.40, 0.45),
    (-0.25, 0.25, 0.20, 0.25, 0.45, 0.90),
    (-0.30, -0.25, -0.25, 0.20, 0.45, 0.65),
    (0.25, 0.30, -0.25, 0.20, 0.45, 0.65),
)
SOFA = mesh(
    (-1.0, 1.0, -0.45, 0.45, 0.0, 0.30),
    (-0.95, -0.02, -0.40, 0.25, 0.30, 0.43),
    (0.02, 0.95, -0.40, 0.25, 0.30, 0.46),
    (-1.0, 1.0, 0.25, 0.45, 0.30, 0.85),
    (-1.0, -0.95, -0.45, 0.25, 0.30, 0.62),
    (0.95, 1.0, -0.45, 0.25, 0.30, 0.62),
)
CHAIR_BBOX = BoundingBox(((-0.30, 0.30), (-0.25, 0.25), (0.0, 0.90)))


def test_chair_seat_ignores_armrests_and_backrest():
    assert surface_z(*CHAIR, ((-0.30, 0.30), (-0.25, 0.25))) == pytest.approx(0.45)


def test_bar_stool_seat_high_in_the_band():
    stool = mesh(*legs(-0.18, 0.18, -0.18, 0.18, 0.72), (-0.18, 0.18, -0.18, 0.18, 0.72, 0.76))
    assert surface_z(*stool, ((-0.2, 0.2), (-0.2, 0.2))) == pytest.approx(0.76)


def test_sofa_gives_each_seat_its_own_height():
    assert surface_z(*SOFA, ((-0.9, -0.1), (-0.4, 0.2))) == pytest.approx(0.43)
    assert surface_z(*SOFA, ((0.1, 0.9), (-0.4, 0.2))) == pytest.approx(0.46)


def test_flipped_normals_give_the_same_height():
    assert surface_z(*mesh(*legs(-0.25, 0.25, -0.25, 0.25, 0.40), (-0.25, 0.25, -0.25, 0.25, 0.40, 0.45), flip=True), ((-0.3, 0.3), (-0.3, 0.3))) == pytest.approx(0.45)


def test_no_seat_surface_under_the_area():
    table = mesh(*legs(-0.5, 0.5, -0.5, 0.5, 0.96), (-0.5, 0.5, -0.5, 0.5, 0.96, 1.0))
    assert surface_z(*table, ((-0.5, 0.5), (-0.5, 0.5))) is None
    assert surface_z(*CHAIR, ((2.0, 3.0), (2.0, 3.0))) is None


def test_largest_seat_is_the_chair_seat():
    x, y, z = largest_seat(*CHAIR)
    assert (x, y, z) == pytest.approx((0.0, 0.0, 0.45), abs=0.03)


@needs_geometry
def test_place_slot_fills_the_z_range():
    slot = place_slot(source_slot({"bbox": [[-0.25, 0.25], [-0.25, 0.2]], "yaw": "-y"}), *CHAIR, CHAIR_BBOX, Face.NEG_Y, recompute=False, label="chair")
    assert list(slot.bbox[2]) == pytest.approx([0.0, 0.45])
    assert slot.facing.serialize() == "-y"


@needs_geometry
def test_place_slot_keeps_a_hand_z_unless_recomputing():
    hand = source_slot({"bbox": [[-0.25, 0.25], [-0.25, 0.2], [0.0, 0.5]], "yaw": "-y"})
    assert place_slot(hand, *CHAIR, CHAIR_BBOX, Face.NEG_Y, recompute=False, label="chair") is hand
    assert place_slot(hand, *CHAIR, CHAIR_BBOX, Face.NEG_Y, recompute=True, label="chair").bbox[2][1] == pytest.approx(0.45)


@needs_geometry
def test_place_slot_places_an_empty_slot_facing_the_face_axis():
    slot = place_slot(source_slot({}), *CHAIR, CHAIR_BBOX, Face.POS_X, recompute=False, label="chair")
    assert slot.bbox[2] == pytest.approx((0.0, 0.45))
    assert slot.bbox[0][1] - slot.bbox[0][0] == pytest.approx(0.45, abs=0.06)
    assert slot.facing.serialize() == "+x"


@needs_geometry
def test_place_slot_falls_back_without_a_seat_surface():
    flat = mesh((-0.5, 0.5, -0.5, 0.5, 0.0, 0.1))
    slot = place_slot(source_slot({"bbox": [[-0.2, 0.2], [-0.2, 0.2]], "yaw": 0}), *flat, BoundingBox(((-0.5, 0.5), (-0.5, 0.5), (0.0, 0.1))), Face.NEG_Y, recompute=False, label="mat")
    assert slot.bbox[2][1] == pytest.approx(0.1)


@needs_geometry
def test_place_slot_puts_an_empty_slot_at_the_bbox_center_without_a_seat_surface():
    flat = mesh((-0.5, 0.5, -0.5, 0.5, 0.0, 0.1))
    slot = place_slot(source_slot({"yaw": "+y"}), *flat, BoundingBox(((-0.5, 0.5), (-0.5, 0.5), (0.0, 0.1))), Face.NEG_Y, recompute=False, label="mat")
    assert list(slot.bbox) == pytest.approx([(-0.225, 0.225), (-0.225, 0.225), (0.0, 0.1)])
    assert slot.facing.serialize() == "+y"


@needs_geometry
def test_carry_slots_takes_z_from_the_earlier_build():
    built = [source_slot({"bbox": [[0.0, 0.5], [0.0, 0.5], [0.0, 0.43]], "yaw": "-y"}), source_slot({"bbox": [[0.6, 1.0], [0.0, 0.4], [0.0, 0.45]], "yaw": "+x"})]
    source = [source_slot({"bbox": [[0.0, 0.5], [0.0, 0.5]], "yaw": "+y"}), source_slot({}), source_slot({"bbox": [[2.0, 2.5], [0.0, 0.5]], "yaw": "-y"})]
    carried = carry_slots(source, built)
    assert list(carried[0].bbox[2]) == [0.0, 0.43]
    assert carried[0].facing.serialize() == "+y"
    assert carried[1] == built[1]
    assert carried[2] is source[2]


@needs_geometry
def test_carry_slots_leaves_the_empty_slot_when_the_built_one_is_ambiguous_or_faces_elsewhere():
    built = [source_slot({"bbox": [[0.0, 0.5], [0.0, 0.5], [0.0, 0.43]], "yaw": "-y"}), source_slot({"bbox": [[0.6, 1.0], [0.0, 0.4], [0.0, 0.45]], "yaw": "+x"})]
    empty = source_slot({})
    assert carry_slots([empty], built)[0] is empty
    turned = source_slot({"yaw": "-x"})
    assert carry_slots([source_slot({"bbox": [[0.0, 0.5], [0.0, 0.5]], "yaw": "-y"}), turned], built)[1] is turned
