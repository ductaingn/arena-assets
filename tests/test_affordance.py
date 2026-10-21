import importlib.util
import math

import pytest
import yaml

from arena_assets.impl import converter
from arena_assets.impl.build.ObjectDatabaseBuilder import ObjectAnnotation

pytestmark = pytest.mark.skipif(importlib.util.find_spec("arena_simulation_setup") is None, reason="arena_simulation_setup comes with the Arena workspace")

SOFA_B = """
name: Sofa_B
path: Residential/Object/Sofa_B
bounding_box: [[-1.4647, 1.4644], [-1.2297, 1.2267], [0.0, 0.9]]
face: -y
affordances:
  sit:
    - {bbox: [[-0.46, 0.48], [0.03, 0.72]], yaw: -y}
    - {bbox: [[-0.98, -0.48], [-0.50, 0.03]], yaw: +x}
    - {bbox: [[-1.30, -0.56], [-1.13, -0.26]], yaw: [+x, -y]}
    - {bbox: [[0.48, 1.29], [-0.51, 0.08]], yaw: -x}
    - {bbox: [[0.59, 1.36], [-1.06, -0.38]], yaw: [-x, +x, -y]}
"""


def sofa() -> ObjectAnnotation:
    return converter.structure(yaml.safe_load(SOFA_B), ObjectAnnotation)


def test_structures_slots_with_axes_and_lists():
    slots = sofa().affordances["sit"]
    assert len(slots) == 5
    assert slots[0].facing.yaws == pytest.approx([-math.pi / 2])
    assert slots[4].facing.serialize() == ["-x", "+x", "-y"]


def test_unbuilt_slot_bbox_has_no_z_range():
    assert sofa().affordances["sit"][0].bbox[2] == (0.0, 0.0)


def test_unstructure_keeps_source_form():
    data = converter.unstructure(sofa())
    assert data["affordances"]["sit"][0] == {"bbox": [[-0.46, 0.48], [0.03, 0.72]], "yaw": "-y"}
    assert data["affordances"]["sit"][2]["yaw"] == ["+x", "-y"]


def test_empty_slot_is_left_for_the_build():
    annotation = converter.structure({"name": "Stool", "path": "Object/Stool", "affordances": {"sit": [{}]}}, ObjectAnnotation)
    assert converter.unstructure(annotation)["affordances"] == {"sit": [{}]}


def test_annotation_without_affordances_omits_them():
    annotation = converter.structure({"name": "Box", "path": "Object/Box"}, ObjectAnnotation)
    assert "affordances" not in converter.unstructure(annotation)
    assert "affordances" not in annotation.as_metadata


def test_rejects_unknown_affordance():
    with pytest.raises(ValueError, match="unknown affordance 'fly', known: sit"):
        converter.structure({"name": "Box", "path": "Object/Box", "affordances": {"fly": []}}, ObjectAnnotation)


def test_rejects_two_slots_left_to_the_build():
    with pytest.raises(ValueError, match="at most one sit slot"):
        converter.structure({"name": "Sofa", "path": "Object/Sofa", "affordances": {"sit": [{}, {"yaw": "-y"}]}}, ObjectAnnotation)


def test_metadata_roundtrip_keeps_affordances():
    annotation = sofa()
    restored = ObjectAnnotation.from_metadata(annotation.as_metadata)
    assert restored.affordances == annotation.affordances


def test_query_affordance_by_enum_and_string():
    from arena_simulation_setup.utils.affordance import AffordanceKind

    annotation = sofa()
    assert annotation.query_affordance(AffordanceKind.SIT, 3) == annotation.query_affordance("sit", 3)
    slot = annotation.query_affordance("sit")
    assert (slot.kind, slot.index, slot.facing.serialize()) == (AffordanceKind.SIT, 0, "-y")


def test_query_affordance_names_what_exists():
    annotation = sofa()
    with pytest.raises(KeyError, match="known: sit"):
        annotation.query_affordance("lie")
    with pytest.raises(KeyError, match="has 5 sit slots, got slot 5"):
        annotation.query_affordance("sit", 5)
    with pytest.raises(KeyError, match="has no sit affordance, it has: none"):
        converter.structure({"name": "Box", "path": "Object/Box"}, ObjectAnnotation).query_affordance("sit")


def test_query_affordance_refuses_unplaced_slot():
    annotation = converter.structure({"name": "Stool", "path": "Object/Stool", "affordances": {"sit": [{}]}}, ObjectAnnotation)
    with pytest.raises(ValueError, match="not placed yet"):
        annotation.query_affordance("sit")


def test_affordance_slots_feed_claims_and_skip_unplaced_ones():
    from arena_simulation_setup.utils.affordance import AffordanceClaims

    raw = yaml.safe_load(SOFA_B)
    raw["affordances"]["sit"].append({})
    annotation = converter.structure(raw, ObjectAnnotation)
    claims = AffordanceClaims(annotation.affordance_slots())
    assert [slot.index for slot in claims.slots] == [0, 1, 2, 3, 4]
    claims.claim(annotation.query_affordance("sit", 1))
    assert [slot.index for slot in claims.available()] == [0, 3, 4]
