import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml
from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils
from typer.testing import CliRunner

from arena_assets.__main__ import app
from arena_assets.impl import collider, converter
from arena_assets.impl.build.ObjectDatabaseBuilder import ObjectAnnotation
from arena_assets.utils.mesh_io import read_mesh
from arena_assets.utils.ModelConverter import sdf_model

runner = CliRunner()

POINTS = np.array([[-0.3, -0.2, 0.0], [0.3, -0.2, 0.0], [0.3, 0.2, 0.0], [-0.3, 0.2, 0.0], [0.0, 0.0, 0.9]])
FACES = [[0, 1, 2, 3], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]]
TRIANGLES = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
ANNOTATION = {"name": "chair", "path": "Object/chair", "desc": "A chair.", "asa": 1, "bounding_box": [[-0.3, 0.3], [-0.2, 0.2], [0.0, 0.9]], "face": "-y", "source": "scan 4"}
GLOW = [("glow_lamp", "chair.glow_0.dae", "")]


def usda(up_axis: str = "Z", meters_per_unit: float = 1.0, root_ops: str = "") -> str:
    return f"""#usda 1.0
(
    defaultPrim = "chair"
    metersPerUnit = {meters_per_unit}
    upAxis = "{up_axis}"
)

def Xform "chair"
{{
{root_ops}
    def Mesh "body"
    {{
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    }}
}}
"""


def make_bundle(path: Path, usd: str | None = ".usda", sdf: bool = True) -> Path:
    path.mkdir()
    if usd is not None:
        (path / f"chair{usd}").write_text(usda())
    if sdf:
        (path / "chair.sdf").mkdir()
        (path / "chair.sdf" / "chair.sdf").write_text(sdf_model("chair", GLOW))
        for mesh in ("chair.dae", "chair.glow_0.dae"):
            (path / "chair.sdf" / mesh).write_text("dae")
    (path / "annotation.yaml").write_text(yaml.safe_dump(ANNOTATION))
    return path


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    return make_bundle(tmp_path / "chair")


@pytest.fixture
def mesh(tmp_path: Path) -> Path:
    path = tmp_path / "hull.obj"
    path.write_text("\n".join([*(f"v {x} {y} {z}" for x, y, z in POINTS), *("f " + " ".join(str(i + 1) for i in face) for face in FACES)]) + "\n")
    return path


def usd_collider(path: Path) -> tuple[Usd.Stage, Usd.Prim, np.ndarray, np.ndarray]:
    """The stage, its baked collider mesh prim, the mesh points in stage coordinates and its triangles."""
    stage = Usd.Stage.Open(str(path))
    prim = stage.GetPrimAtPath("/chair/collider/mesh")
    geometry = UsdGeom.Mesh(prim)
    world = np.asarray(geometry.ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    points = np.array(geometry.GetPointsAttr().Get(), np.float64)
    return stage, prim, points @ world[:3, :3].T + world[:3, 3], np.array(geometry.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)


def sdf_parts(bundle: Path, tag: str) -> list[tuple[str, str]]:
    link = ET.parse(bundle / "chair.sdf" / "chair.sdf").getroot().find("./model/link")
    return [(element.get("name"), element.findtext("./geometry/mesh/uri")) for element in link.findall(tag)]


def files(path: Path) -> list[str]:
    return sorted(str(child.relative_to(path)) for child in path.rglob("*"))


def annotation(bundle: Path) -> dict:
    return yaml.safe_load((bundle / "annotation.yaml").read_text())


def assert_refused(result, *needles: str) -> None:
    assert result.exit_code != 0
    message = result.output.strip()
    assert message.startswith("arena-assets collider: ")
    assert message.endswith("nothing was changed")
    for needle in needles:
        assert needle in message


def test_attach_bakes_the_mesh_into_every_model_and_names_it_in_the_annotation(bundle, mesh):
    visuals = sdf_parts(bundle, "visual")
    result = runner.invoke(app, ["collider", str(bundle), str(mesh)])
    assert result.exit_code == 0, result.output
    assert "baked 6 collider triangles into chair.usda, chair.sdf/chair.sdf" in result.output
    assert result.output.strip().splitlines()[-1] == str(bundle)
    assert files(bundle) == ["annotation.yaml", "chair.collider.obj", "chair.sdf", "chair.sdf/chair.collider.obj", "chair.sdf/chair.dae", "chair.sdf/chair.glow_0.dae", "chair.sdf/chair.sdf", "chair.usda"]
    assert annotation(bundle) == {**ANNOTATION, "collider": "chair.collider.obj"}

    stage, prim, points, triangles = usd_collider(bundle / "chair.usda")
    np.testing.assert_allclose(points, POINTS, atol=1e-6)
    np.testing.assert_array_equal(triangles, TRIANGLES)
    assert prim.HasAPI(UsdPhysics.CollisionAPI)
    assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == UsdPhysics.Tokens.none
    assert UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible
    assert UsdGeom.Imageable(stage.GetPrimAtPath("/chair/body")).ComputeVisibility() == UsdGeom.Tokens.inherited

    assert sdf_parts(bundle, "visual") == visuals
    assert sdf_parts(bundle, "collision") == [("collider", "chair.collider.obj")]
    for obj in (bundle / "chair.collider.obj", bundle / "chair.sdf" / "chair.collider.obj"):
        obj_points, obj_triangles = read_mesh(obj)
        np.testing.assert_allclose(obj_points, POINTS, atol=1e-6)
        np.testing.assert_array_equal(obj_triangles, TRIANGLES)


def test_collider_obj_carries_one_outward_normal_per_face(bundle, mesh):
    assert runner.invoke(app, ["collider", str(bundle), str(mesh)]).exit_code == 0
    lines = (bundle / "chair.collider.obj").read_text().splitlines()
    normals = np.array([[float(value) for value in line.split()[1:]] for line in lines if line.startswith("vn ")])
    faces = [[tuple(int(index) for index in corner.split("//")) for corner in line.split()[1:]] for line in lines if line.startswith("f ")]
    assert len(normals) == len(faces) == len(TRIANGLES)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-6)
    for number, (face, triangle) in enumerate(zip(faces, TRIANGLES, strict=True), 1):
        assert [corner[0] - 1 for corner in face] == list(triangle)
        assert {corner[1] for corner in face} == {number}
        a, b, c = POINTS[triangle]
        np.testing.assert_allclose(normals[number - 1], np.cross(b - a, c - a) / np.linalg.norm(np.cross(b - a, c - a)), atol=1e-6)


def test_usdz_package_keeps_its_files_and_gains_the_collider(tmp_path, mesh):
    source = tmp_path / "source"
    source.mkdir()
    (source / "chair.usda").write_text(usda().replace("point3f[] points", 'asset texture = @./textures/wood.png@\n        point3f[] points'))
    (source / "textures").mkdir()
    (source / "textures" / "wood.png").write_bytes(b"\x89PNG wood")
    bundle = make_bundle(tmp_path / "chair", usd=None, sdf=False)
    assert UsdUtils.CreateNewUsdzPackage(str(source / "chair.usda"), str(bundle / "chair.usdz"))
    with zipfile.ZipFile(bundle / "chair.usdz") as archive:
        before = archive.namelist()

    result = runner.invoke(app, ["collider", str(bundle), str(mesh)])
    assert result.exit_code == 0, result.output
    assert "baked 6 collider triangles into chair.usdz" in result.output
    with zipfile.ZipFile(bundle / "chair.usdz") as archive:
        assert archive.namelist() == before
        assert archive.read("textures/wood.png") == b"\x89PNG wood"
        assert all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist())
    _, _, points, triangles = usd_collider(bundle / "chair.usdz")
    np.testing.assert_allclose(points, POINTS, atol=1e-6)
    np.testing.assert_array_equal(triangles, TRIANGLES)


def test_collider_lands_in_the_bundle_frame_under_a_moved_y_up_centimeter_root(tmp_path, mesh):
    bundle = make_bundle(tmp_path / "chair", usd=None, sdf=False)
    root_ops = '    double3 xformOp:translate = (10, 20, 30)\n    float xformOp:rotateY = 90\n    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateY"]'
    (bundle / "chair.usda").write_text(usda("Y", 0.01, root_ops))
    assert runner.invoke(app, ["collider", str(bundle), str(mesh)]).exit_code == 0
    _, _, stage_points, _ = usd_collider(bundle / "chair.usda")
    np.testing.assert_allclose(stage_points, 100.0 * POINTS @ collider.Y_UP_FROM_Z_UP.T, atol=1e-4)


def test_matrix_maps_the_mesh_into_the_bundle_frame(bundle, mesh):
    matrix = np.array([[0.0, -2.0, 0.0, 1.0], [2.0, 0.0, 0.0, -0.5], [0.0, 0.0, 2.0, 0.25], [0.0, 0.0, 0.0, 1.0]])
    result = runner.invoke(app, ["collider", str(bundle), str(mesh), "--matrix", " ".join(str(value) for value in matrix.ravel())])
    assert result.exit_code == 0, result.output
    _, _, points, _ = usd_collider(bundle / "chair.usda")
    np.testing.assert_allclose(points, POINTS @ matrix[:3, :3].T + matrix[:3, 3], atol=1e-6)


def test_attaching_again_replaces_the_collider(bundle, mesh, tmp_path):
    assert runner.invoke(app, ["collider", str(bundle), str(mesh)]).exit_code == 0
    smaller = tmp_path / "tri.ply"
    smaller.write_text("ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n")
    result = runner.invoke(app, ["collider", str(bundle), str(smaller)])
    assert result.exit_code == 0, result.output
    stage, _, points, triangles = usd_collider(bundle / "chair.usda")
    assert len(points) == 3
    np.testing.assert_array_equal(triangles, [[0, 1, 2]])
    assert [prim.GetName() for prim in stage.GetPrimAtPath("/chair").GetChildren()] == ["body", "collider"]
    assert sdf_parts(bundle, "collision") == [("collider", "chair.collider.obj")]
    assert len(read_mesh(bundle / "chair.collider.obj")[1]) == 1
    assert annotation(bundle)["collider"] == "chair.collider.obj"


def test_remove_restores_the_models_as_built(bundle, mesh):
    before = files(bundle)
    collisions = sdf_parts(bundle, "collision")
    assert collisions == [("collision", "chair.dae"), ("collision_glow_0", "chair.glow_0.dae")]
    assert runner.invoke(app, ["collider", str(bundle), str(mesh)]).exit_code == 0
    result = runner.invoke(app, ["collider", str(bundle), "--remove"])
    assert result.exit_code == 0, result.output
    assert "removed the collider from chair.usda, chair.sdf/chair.sdf" in result.output
    assert files(bundle) == before
    assert sdf_parts(bundle, "collision") == collisions
    link = ET.parse(bundle / "chair.sdf" / "chair.sdf").getroot().find("./model/link")
    assert [child.tag for child in link] == ["visual", "collision", "visual", "collision"]
    stage = Usd.Stage.Open(str(bundle / "chair.usda"))
    assert [prim.GetName() for prim in stage.GetPrimAtPath("/chair").GetChildren()] == ["body"]
    assert annotation(bundle) == ANNOTATION


def visual_collision(path: Path) -> tuple[bool, bool | None]:
    """Whether the body mesh has a collision schema and its collisionEnabled value."""
    stage = Usd.Stage.Open(str(path))
    prim = stage.GetPrimAtPath("/chair/body")
    return prim.HasAPI(UsdPhysics.CollisionAPI), UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() if prim.HasAPI(UsdPhysics.CollisionAPI) else None


def test_collide_visuals_gives_visual_meshes_a_static_triangle_collision(bundle):
    assert visual_collision(bundle / "chair.usda") == (False, None)
    assert collider.collide_visuals(bundle / "chair.usda")
    stage = Usd.Stage.Open(str(bundle / "chair.usda"))
    body = stage.GetPrimAtPath("/chair/body")
    assert body.HasAPI(UsdPhysics.CollisionAPI)
    assert not body.HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdPhysics.MeshCollisionAPI(body).GetApproximationAttr().Get() == UsdPhysics.Tokens.none
    assert UsdGeom.Imageable(body).ComputeVisibility() == UsdGeom.Tokens.inherited


def test_baked_collider_turns_the_visual_collision_off_and_remove_turns_it_back_on(bundle, mesh):
    collider.collide_visuals(bundle / "chair.usda")
    assert visual_collision(bundle / "chair.usda") == (True, True)
    assert runner.invoke(app, ["collider", str(bundle), str(mesh)]).exit_code == 0
    assert visual_collision(bundle / "chair.usda") == (True, False)
    stage, prim, _, _ = usd_collider(bundle / "chair.usda")
    assert UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False
    assert runner.invoke(app, ["collider", str(bundle), "--remove"]).exit_code == 0
    assert visual_collision(bundle / "chair.usda") == (True, True)


def test_collider_leaves_visual_meshes_without_collision_alone(bundle, mesh):
    assert runner.invoke(app, ["collider", str(bundle), str(mesh)]).exit_code == 0
    assert visual_collision(bundle / "chair.usda") == (False, None)
    assert runner.invoke(app, ["collider", str(bundle), "--remove"]).exit_code == 0
    assert visual_collision(bundle / "chair.usda") == (False, None)


def test_remove_deletes_an_sdf_that_holds_only_the_collider(tmp_path, mesh):
    bundle = make_bundle(tmp_path / "chair", sdf=False)
    collider.collision_only_sdf(bundle, "chair")
    assert runner.invoke(app, ["collider", str(bundle), str(mesh)]).exit_code == 0
    assert sdf_parts(bundle, "visual") == []
    assert sdf_parts(bundle, "collision") == [("collider", "chair.collider.obj")]
    assert runner.invoke(app, ["collider", str(bundle), "--remove"]).exit_code == 0
    assert files(bundle) == ["annotation.yaml", "chair.usda"]


def test_remove_on_a_bundle_without_collider_changes_nothing(bundle):
    before = {path: path.read_bytes() for path in bundle.rglob("*") if path.is_file()}
    result = runner.invoke(app, ["collider", str(bundle), "--remove"])
    assert result.exit_code == 0, result.output
    assert "the bundle had no collider" in result.output
    assert {path: path.read_bytes() for path in bundle.rglob("*") if path.is_file()} == before


def test_bundle_without_annotation_is_refused(tmp_path, mesh):
    empty = tmp_path / "loose"
    empty.mkdir()
    assert_refused(runner.invoke(app, ["collider", str(empty), str(mesh)]), "has no annotation.yaml")
    assert not list(empty.iterdir())


def test_bundle_without_a_model_is_refused(tmp_path, mesh):
    bundle = make_bundle(tmp_path / "chair", usd=None, sdf=False)
    assert_refused(runner.invoke(app, ["collider", str(bundle), str(mesh)]), "holds none of chair.usdz", "chair.sdf/chair.sdf")
    assert files(bundle) == ["annotation.yaml"]


def test_usd_model_without_default_prim_is_refused(bundle, mesh):
    (bundle / "chair.usda").write_text(usda().replace('    defaultPrim = "chair"\n', ""))
    before = files(bundle)
    assert_refused(runner.invoke(app, ["collider", str(bundle), str(mesh)]), "has no default prim")
    assert files(bundle) == before
    assert "<collision name=\"collider\">" not in (bundle / "chair.sdf" / "chair.sdf").read_text()


def test_missing_bundle_is_refused(tmp_path, mesh):
    assert_refused(runner.invoke(app, ["collider", str(tmp_path / "absent"), str(mesh)]), "is not a directory")


def test_missing_mesh_argument_is_refused(bundle):
    assert_refused(runner.invoke(app, ["collider", str(bundle)]), "MESH is missing", "--remove")


def test_remove_with_a_mesh_is_refused(bundle, mesh):
    assert_refused(runner.invoke(app, ["collider", str(bundle), str(mesh), "--remove"]), "--remove takes only BUNDLE")
    assert annotation(bundle) == ANNOTATION


def test_mesh_in_an_unknown_format_is_refused(bundle, tmp_path):
    stl = tmp_path / "hull.stl"
    stl.write_text("solid")
    before = files(bundle)
    assert_refused(runner.invoke(app, ["collider", str(bundle), str(stl)]), "not one of the mesh formats .obj, .ply")
    assert annotation(bundle) == ANNOTATION
    assert files(bundle) == before


def test_transposed_matrix_is_refused(bundle, mesh):
    matrix = np.eye(4)
    matrix[3, :3] = (1.0, 2.0, 3.0)
    assert_refused(runner.invoke(app, ["collider", str(bundle), str(mesh), "--matrix", " ".join(str(value) for value in matrix.ravel())]), "must end with the row 0 0 0 1")


def test_bare_verb_prints_usage():
    output = " ".join(runner.invoke(app, ["collider"]).output.split())
    for needle in ("Usage", "BUNDLE", "MESH", "--matrix", "--remove", "collider.obj", "NAME.sdf/NAME.sdf", "arena asset push object"):
        assert needle in output


def test_annotation_carries_the_collider_through_yaml_and_metadata():
    structured = converter.structure({**ANNOTATION, "collider": "chair.collider.obj"}, ObjectAnnotation)
    assert structured.collider == "chair.collider.obj"
    assert converter.unstructure(structured)["collider"] == "chair.collider.obj"
    assert structured.as_metadata["collider"] == "chair.collider.obj"
    assert ObjectAnnotation.from_metadata(structured.as_metadata).collider == "chair.collider.obj"


def test_annotation_without_collider_omits_the_field():
    structured = converter.structure(ANNOTATION, ObjectAnnotation)
    assert structured.collider == ""
    assert "collider" not in converter.unstructure(structured)
    assert "collider" not in structured.as_metadata
    assert ObjectAnnotation.from_metadata(structured.as_metadata).collider == ""
