import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from arena_assets.impl import ANNOTATION_NAME
from arena_assets.utils.geom import BoundingBox
from arena_assets.utils.ModelConverter import BlenderWorker

BOX_FACES = [0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3]
UPRIGHT_BOX = BoundingBox(((-0.5, 0.5), (-1.0, 1.0), (0.0, 3.0)))
BOX_EXTENTS = {
    "Z": ((-0.5, 0.5), (-1.0, 1.0), (0.0, 3.0)),
    "Y": ((-0.5, 0.5), (0.0, 3.0), (-1.0, 1.0)),
}
SUNKEN_BOX = ((-0.5, 0.5), (-1.0, 1.0), (-1.0, 2.0))
HULL = np.array([[-0.4, -0.9, -1.0], [0.4, -0.9, -1.0], [0.0, 0.9, -1.0], [0.0, 0.0, 1.9]])
HULL_FACES = [[0, 1, 2], [0, 1, 3], [1, 2, 3], [2, 0, 3]]
OFF_CENTER = "double3 xformOp:translate = (4, 6, 2)\n    uniform token[] xformOpOrder = [\"xformOp:translate\"]"
EXTRAS = {
    "LIGHT": f"""
def SphereLight "Lamp"
{{
    float inputs:intensity = 1000
    float inputs:radius = 0.1
    {OFF_CENTER}
}}
""",
    "EMPTY": f"""
def Xform "Marker"
{{
    {OFF_CENTER}
}}
""",
}


def _usda_box(up_axis, extra="", extents=None):
    xs, ys, zs = extents or BOX_EXTENTS[up_axis]
    points = ", ".join(f"({x}, {y}, {z})" for x in xs for y in ys for z in zs)
    return f"""#usda 1.0
(
    defaultPrim = "Root"
    metersPerUnit = 1
    upAxis = "{up_axis}"
)

def Xform "Root"
{{
    def Mesh "Box"
    {{
        int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
        int[] faceVertexIndices = {BOX_FACES}
        point3f[] points = [{points}]
    }}
}}
"""


def _write_box_source(directory, up_axis="Z"):
    directory.mkdir(parents=True)
    (directory / "box.usda").write_text(_usda_box(up_axis))
    (directory / ANNOTATION_NAME).write_text(yaml.safe_dump({"desc": "box"}))
    return directory / "box.usda"


def _flat(bounding_box):
    return [bound for side in bounding_box for bound in side]


def _loaded_box(path):
    worker = BlenderWorker()
    try:
        return BoundingBox(worker.run([("load", str(path)), ("bounding_box",)]).values[1]).round(4)
    finally:
        worker.close()


@pytest.mark.parametrize("up_axis", ["Z", "Y"])
def test_build_keeps_usd_source_upright(blender, tmp_path, up_axis):
    from arena_assets.impl.build.ObjectDatabaseBuilder import ObjectDatabaseBuilder

    input_path = tmp_path / "input"
    output_path = tmp_path / "output"
    _write_box_source(input_path / "Object" / "box", up_axis)

    builder = ObjectDatabaseBuilder(input_path, output_path)
    builder.enable("formats", "fbx")
    dest = output_path / "Object" / "box"
    dest.mkdir(parents=True)
    built = builder.process_entity(builder.read_annotation_file(input_path / "Object" / "box"), dest)

    assert built is not None
    assert built.bounding_box == UPRIGHT_BOX
    assert _flat(_loaded_box(dest / "box.fbx")) == pytest.approx(_flat(UPRIGHT_BOX), abs=1e-3)


GLB_BUILD_SCRIPT = """
import sys
from pathlib import Path

from arena_assets.impl.build.ObjectDatabaseBuilder import ObjectDatabaseBuilder

input_path, output_path = Path(sys.argv[1]), Path(sys.argv[2])
builder = ObjectDatabaseBuilder(input_path, output_path)
builder.enable("formats", "glb")
dest = output_path / "Object" / "box"
dest.mkdir(parents=True)
built = builder.process_entity(builder.read_annotation_file(input_path / "Object" / "box"), dest)
assert built is not None
assert (dest / "box.glb").stat().st_size > 0
"""


def test_process_exits_after_gltf_build(blender, tmp_path):

    _write_box_source(tmp_path / "input" / "Object" / "box")
    result = subprocess.run([sys.executable, "-c", GLB_BUILD_SCRIPT, str(tmp_path / "input"), str(tmp_path / "output")], capture_output=True, text=True, timeout=120)

    assert result.returncode == 0, result.stderr


def test_build_bakes_the_declared_collider_rectified_with_the_model(blender, tmp_path):
    import xml.etree.ElementTree as ET

    from pxr import Usd, UsdGeom, UsdPhysics

    from arena_assets.impl.build.ObjectDatabaseBuilder import ObjectDatabaseBuilder
    from arena_assets.utils.mesh_io import read_mesh

    source = tmp_path / "input" / "Object" / "box"
    source.mkdir(parents=True)
    (source / "box.usda").write_text(_usda_box("Z", extents=SUNKEN_BOX))
    (source / "hull.obj").write_text("\n".join([*(f"v {x} {y} {z}" for x, y, z in HULL), *(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in HULL_FACES)]) + "\n")
    (source / ANNOTATION_NAME).write_text(yaml.safe_dump({"desc": "box", "collider": "hull.obj"}))

    builder = ObjectDatabaseBuilder(tmp_path / "input", tmp_path / "output")
    builder.enable("formats", "usdz,sdf")
    dest = tmp_path / "output" / "Object" / "box"
    dest.mkdir(parents=True)
    annotation = builder.read_annotation_file(source)
    assert Path(ObjectDatabaseBuilder.find_mainfile(annotation)) == Path("box.usda")
    built = builder.process_entity(annotation, dest)

    assert built is not None
    assert built.bounding_box == UPRIGHT_BOX
    assert built.collider == "box.collider.obj"
    lifted = HULL + (0.0, 0.0, 1.0)
    np.testing.assert_allclose(read_mesh(dest / "box.collider.obj")[0], lifted, atol=1e-5)

    stage = Usd.Stage.Open(str(dest / "box.usdz"))
    visuals = [prim for prim in Usd.PrimRange(stage.GetDefaultPrim()) if prim.IsA(UsdGeom.Mesh) and prim.GetName() != "mesh"]
    assert visuals
    assert all(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False for prim in visuals)
    mesh = UsdGeom.Mesh(stage.GetDefaultPrim().GetChild("collider").GetChild("mesh"))
    world = np.asarray(mesh.ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    points = np.array(mesh.GetPointsAttr().Get(), np.float64) @ world[:3, :3].T + world[:3, 3]
    assert (UsdGeom.GetStageUpAxis(stage), UsdGeom.GetStageMetersPerUnit(stage)) == (UsdGeom.Tokens.z, 1.0)
    np.testing.assert_allclose(points, lifted, atol=1e-4)

    link = ET.parse(dest / "box.sdf" / "box.sdf").getroot().find("./model/link")
    assert [(collision.get("name"), collision.findtext("./geometry/mesh/uri")) for collision in link.findall("collision")] == [("collider", "box.collider.obj")]
    assert [visual.findtext("./geometry/mesh/uri") for visual in link.findall("visual")] == ["box.dae"]


def test_build_gives_usd_visual_meshes_collision(blender, tmp_path):
    from pxr import Usd, UsdGeom, UsdPhysics

    from arena_assets.impl.build.ObjectDatabaseBuilder import ObjectDatabaseBuilder

    _write_box_source(tmp_path / "input" / "Object" / "box")
    builder = ObjectDatabaseBuilder(tmp_path / "input", tmp_path / "output")
    builder.enable("formats", "usdz")
    dest = tmp_path / "output" / "Object" / "box"
    dest.mkdir(parents=True)
    built = builder.process_entity(builder.read_annotation_file(tmp_path / "input" / "Object" / "box"), dest)

    assert built is not None
    assert built.collider == ""
    stage = Usd.Stage.Open(str(dest / "box.usdz"))
    meshes = [prim for prim in Usd.PrimRange(stage.GetDefaultPrim()) if prim.IsA(UsdGeom.Mesh)]
    assert meshes
    for prim in meshes:
        assert prim.HasAPI(UsdPhysics.CollisionAPI)
        assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)
        assert UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == UsdPhysics.Tokens.none


SEAT_BLOCK = ((-0.25, 0.25), (-0.25, 0.25), (0.0, 0.45))
SEAT_SLOTS = [{"bbox": [[-0.2, 0.2], [-0.2, 0.2]], "yaw": "-y"}, {}]


def _write_seat_source(directory, slots):
    directory.mkdir(parents=True)
    (directory / "seat.usda").write_text(_usda_box("Z", extents=SEAT_BLOCK))
    (directory / ANNOTATION_NAME).write_text(yaml.safe_dump({"desc": "seat", "face": "+x", "affordances": {"sit": slots}}))


def test_build_gives_sit_slots_the_seat_height(blender, tmp_path):
    pytest.importorskip("arena_simulation_setup")
    from arena_assets.impl.build.ObjectDatabaseBuilder import ObjectDatabaseBuilder

    source = tmp_path / "input" / "Object" / "seat"
    _write_seat_source(source, SEAT_SLOTS)
    builder = ObjectDatabaseBuilder(tmp_path / "input", tmp_path / "output")
    builder.enable("formats", "usdz")
    dest = tmp_path / "output" / "Object" / "seat"
    dest.mkdir(parents=True)
    built = builder.process_entity(builder.read_annotation_file(source), dest)

    assert built is not None
    annotated, placed = (slot.unstructure() for slot in built.affordances["sit"])
    assert annotated == {"bbox": [[-0.2, 0.2], [-0.2, 0.2], [0.0, 0.45]], "yaw": "-y"}
    assert placed["bbox"][2] == [0.0, 0.45]
    assert placed["yaw"] == "+x"


def test_annotation_refresh_keeps_the_built_seat_height(blender, tmp_path):
    pytest.importorskip("arena_simulation_setup")
    from arena_assets.impl import AssetType
    from arena_assets.impl.build import DatabaseBuilder, OverwriteMode

    _write_seat_source(tmp_path / "input" / "Object" / "seat", SEAT_SLOTS)
    builder = DatabaseBuilder.Builder(AssetType.OBJECT)
    first = builder(input_path=tmp_path / "input", output_path=tmp_path / "output")
    first.enable("formats", "usdz")
    first.build()
    dest = tmp_path / "output" / "Object" / "seat"
    built = yaml.safe_load((dest / ANNOTATION_NAME).read_text())
    for slot in built["affordances"]["sit"]:
        slot["bbox"][2] = [0.0, 0.5]
    (dest / ANNOTATION_NAME).write_text(yaml.safe_dump(built))
    (dest / "seat.usdz").unlink()
    builder(input_path=tmp_path / "input", output_path=tmp_path / "output", overwrite=OverwriteMode.ANNOTATIONS_ONLY).build()

    slots = yaml.safe_load((dest / ANNOTATION_NAME).read_text())["affordances"]["sit"]
    assert [slot["bbox"][2] for slot in slots] == [[0.0, 0.5], [0.0, 0.5]]


def test_annotation_refresh_places_new_slots_on_the_built_model(blender, tmp_path):
    pytest.importorskip("arena_simulation_setup")
    from arena_assets.impl import AssetType
    from arena_assets.impl.build import DatabaseBuilder, OverwriteMode

    source = tmp_path / "input" / "Object" / "seat"
    _write_seat_source(source, [])
    builder = DatabaseBuilder.Builder(AssetType.OBJECT)
    first = builder(input_path=tmp_path / "input", output_path=tmp_path / "output")
    first.enable("formats", "sdf")
    first.build()
    (source / ANNOTATION_NAME).write_text(yaml.safe_dump({"desc": "seat", "face": "+x", "affordances": {"sit": SEAT_SLOTS}}))
    builder(input_path=tmp_path / "input", output_path=tmp_path / "output", overwrite=OverwriteMode.ANNOTATIONS_ONLY).build()

    slots = yaml.safe_load((tmp_path / "output" / "Object" / "seat" / ANNOTATION_NAME).read_text())["affordances"]["sit"]
    assert [slot["bbox"][2] for slot in slots] == [[0.0, 0.45], [0.0, 0.45]]
    assert (tmp_path / "output" / "Object" / "seat" / "seat.sdf" / "seat.dae").is_file()


def test_annotation_refresh_with_recompute_replaces_a_hand_seat_height(blender, tmp_path):
    pytest.importorskip("arena_simulation_setup")
    from arena_assets.impl import AssetType
    from arena_assets.impl.build import DatabaseBuilder, OverwriteMode

    _write_seat_source(tmp_path / "input" / "Object" / "seat", [{"bbox": [[-0.2, 0.2], [-0.2, 0.2], [0.0, 0.9]], "yaw": "-y"}])
    builder = DatabaseBuilder.Builder(AssetType.OBJECT)
    first = builder(input_path=tmp_path / "input", output_path=tmp_path / "output")
    first.enable("formats", "usdz")
    first.build()
    dest = tmp_path / "output" / "Object" / "seat"
    assert yaml.safe_load((dest / ANNOTATION_NAME).read_text())["affordances"]["sit"][0]["bbox"][2] == [0.0, 0.9]
    refresh = builder(input_path=tmp_path / "input", output_path=tmp_path / "output", overwrite=OverwriteMode.ANNOTATIONS_ONLY)
    refresh.enable("recompute-affordances")
    refresh.build()

    assert yaml.safe_load((dest / ANNOTATION_NAME).read_text())["affordances"]["sit"][0]["bbox"][2] == [0.0, 0.45]


def test_annotation_refresh_keeps_the_asset_when_placing_slots_fails(blender, tmp_path):
    pytest.importorskip("arena_simulation_setup")
    from arena_assets.impl import AssetType
    from arena_assets.impl.build import DatabaseBuilder, OverwriteMode

    _write_seat_source(tmp_path / "input" / "Object" / "seat", SEAT_SLOTS[:1])
    builder = DatabaseBuilder.Builder(AssetType.OBJECT)
    first = builder(input_path=tmp_path / "input", output_path=tmp_path / "output")
    first.enable("formats", "usdz")
    first.build()
    dest = tmp_path / "output" / "Object" / "seat"
    (dest / ANNOTATION_NAME).write_text(yaml.safe_dump({"desc": "seat", "affordances": {"fly": []}}))
    builder(input_path=tmp_path / "input", output_path=tmp_path / "output", overwrite=OverwriteMode.ANNOTATIONS_ONLY).build()

    assert (dest / "seat.usdz").is_file()
    assert yaml.safe_load((dest / ANNOTATION_NAME).read_text())["affordances"]["sit"] == [{"bbox": [[-0.2, 0.2], [-0.2, 0.2]], "yaw": "-y"}]


def test_annotation_refresh_without_a_built_model_leaves_new_slots_unplaced(tmp_path):
    pytest.importorskip("arena_simulation_setup")
    from arena_assets.impl import AssetType
    from arena_assets.impl.build import DatabaseBuilder, OverwriteMode

    _write_seat_source(tmp_path / "input" / "Object" / "seat", SEAT_SLOTS[:1])
    (tmp_path / "output" / "Object" / "seat").mkdir(parents=True)
    DatabaseBuilder.Builder(AssetType.OBJECT)(input_path=tmp_path / "input", output_path=tmp_path / "output", overwrite=OverwriteMode.ANNOTATIONS_ONLY).build()

    slots = yaml.safe_load((tmp_path / "output" / "Object" / "seat" / ANNOTATION_NAME).read_text())["affordances"]["sit"]
    assert slots == [{"bbox": [[-0.2, 0.2], [-0.2, 0.2]], "yaw": "-y"}]


def test_host_and_worker_write_the_same_sdf_model(blender_python):
    from arena_assets.utils.ModelConverter import environment, sdf_model

    glow = [("glow_Shade", "box.glow_0.dae", "<material/>")]
    script = "import json, sys\nfrom arena_assets_blender import sdf_model\nglow = [tuple(g) for g in json.loads(sys.argv[1])]\nprint(json.dumps([sdf_model('box'), sdf_model('box', glow), sdf_model('box', plain=False)]))"
    worker = subprocess.run([blender_python, "-c", script, json.dumps(glow)], capture_output=True, text=True, timeout=120, env=environment(), check=True)
    assert json.loads(worker.stdout) == [sdf_model("box"), sdf_model("box", glow), sdf_model("box", plain=False)]
