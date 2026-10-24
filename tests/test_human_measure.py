import numpy as np
import pytest

from arena_assets.impl import Face
from arena_assets.impl.build.HumanDatabaseBuilder import HumanAnnotation, measure_pelvis_clearance
from arena_assets.utils.human import Skin, pelvis_clearance, read_skin


def body(coat: bool = False) -> Skin:
    """Pelvis 0.9 m up, hip joints 0.02 m lower and 0.01 m behind it, thighs reaching 0.1 m behind the hip joints."""
    joints = {"Hips": np.array([0.0, 0.0, 0.9]), "LeftUpLeg": np.array([-0.01, 0.1, 0.88]), "RightUpLeg": np.array([-0.01, -0.1, 0.88])}
    points = [(-0.11, 0.1, 0.8), (0.05, 0.1, 0.6), (-0.11, -0.1, 0.8), (0.0, 0.0, 1.0)]
    dominant = ["LeftUpLeg", "LeftUpLeg", "RightUpLeg", "Hips"]
    if coat:  # a hem hanging from the pelvis, far behind and below the body
        points.append((-0.4, 0.0, 0.2))
        dominant.append("Hips")
    return Skin(joints=joints, points=np.array(points), dominant=dominant)


def test_clearance_is_the_thigh_behind_the_hip_joint_seen_from_the_pelvis():
    clearance = pelvis_clearance(body(), np.array([1.0, 0.0]))
    assert (clearance.sit, clearance.lie) == pytest.approx((0.02 + 0.1, 0.01 + 0.1))


def test_clearance_holds_plain_floats_for_yaml():
    clearance = pelvis_clearance(body(), np.array([1.0, 0.0]))
    assert type(clearance.sit) is float and type(clearance.lie) is float


def test_garments_hanging_from_the_pelvis_do_not_count():
    assert pelvis_clearance(body(coat=True), np.array([1.0, 0.0])) == pelvis_clearance(body(), np.array([1.0, 0.0]))


def test_clearance_follows_the_facing():
    turned = body()
    rotate = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # a quarter turn: facing +x becomes +y
    turned = Skin(joints={k: rotate @ v for k, v in turned.joints.items()}, points=turned.points @ rotate.T, dominant=turned.dominant)
    assert pelvis_clearance(turned, np.array([0.0, 1.0])) == pelvis_clearance(body(), np.array([1.0, 0.0]))


def test_clearance_needs_both_thighs():
    skin = body()
    with pytest.raises(ValueError, match="RightUpLeg"):
        pelvis_clearance(Skin(joints=skin.joints, points=skin.points[:2], dominant=skin.dominant[:2]), np.array([1.0, 0.0]))


DAE = """<?xml version="1.0"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <library_geometries><geometry id="body-mesh"><mesh>
    <source id="body-positions"><float_array id="pa" count="9">0 0 1  0.1 0 0  0 0.1 0</float_array>
      <technique_common><accessor source="#pa" count="3" stride="3"/></technique_common></source>
    <vertices id="body-vertices"><input semantic="POSITION" source="#body-positions"/></vertices>
  </mesh></geometry></library_geometries>
  <library_controllers><controller id="body-skin"><skin source="#body-mesh">
    <bind_shape_matrix>1 0 0 0  0 1 0 0  0 0 1 0.5  0 0 0 1</bind_shape_matrix>
    <source id="joints"><Name_array id="ja" count="2">Hips LeftUpLeg</Name_array></source>
    <source id="binds"><float_array id="ba" count="32">1 0 0 0  0 1 0 0  0 0 1 -0.9  0 0 0 1   1 0 0 0  0 1 0 -0.1  0 0 1 -0.8  0 0 0 1</float_array>
      <technique_common><accessor source="#ba" count="2" stride="16"/></technique_common></source>
    <source id="weights"><float_array id="wa" count="3">1 0.7 0.3</float_array>
      <technique_common><accessor source="#wa" count="3" stride="1"/></technique_common></source>
    <joints><input semantic="JOINT" source="#joints"/><input semantic="INV_BIND_MATRIX" source="#binds"/></joints>
    <vertex_weights count="3"><input semantic="JOINT" source="#joints" offset="0"/><input semantic="WEIGHT" source="#weights" offset="1"/>
      <vcount>1 2 1</vcount><v>0 0  0 1 1 2  1 0</v></vertex_weights>
  </skin></controller></library_controllers>
</COLLADA>
"""


def test_read_skin_takes_joints_from_inverse_binds_and_points_from_the_bind_shape(tmp_path):
    (tmp_path / "body.dae").write_text(DAE)
    skin = read_skin(tmp_path / "body.dae")
    assert skin.joints["Hips"] == pytest.approx([0.0, 0.0, 0.9])
    assert skin.joints["LeftUpLeg"] == pytest.approx([0.0, 0.1, 0.8])
    assert skin.points == pytest.approx(np.array([[0.0, 0.0, 1.5], [0.1, 0.0, 0.5], [0.0, 0.1, 0.5]]))
    assert skin.dominant == ["Hips", "Hips", "LeftUpLeg"]


def test_build_leaves_an_unmeasurable_body_unannotated(tmp_path):
    annotation = HumanAnnotation(name="Nurse01", path="Human/Nurse01", face=Face.POS_X)
    assert measure_pelvis_clearance(annotation, tmp_path / "missing.dae") is None
    assert measure_pelvis_clearance(HumanAnnotation(name="Odd", path="Human/Odd", face=Face.XY), tmp_path / "missing.dae") is None
