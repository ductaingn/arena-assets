import pytest

from arena_assets_blender.CoordinateSystem import CoordinateSystem
from arena_assets_blender.geom import BoundingBox

BOX_FACES = [0, 1, 3, 2, 4, 6, 7, 5, 0, 4, 5, 1, 2, 3, 7, 6, 0, 2, 6, 4, 1, 5, 7, 3]
BOX_EXTENTS = {
    "Z": ((-0.5, 0.5), (-1.0, 1.0), (0.0, 3.0)),
    "Y": ((-0.5, 0.5), (0.0, 3.0), (-1.0, 1.0)),
}
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


def _usda_box(up_axis, extra=""):
    xs, ys, zs = BOX_EXTENTS[up_axis]
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
{extra}"""


def _write_box(path, extra=""):
    path.write_text(_usda_box("Z", extra))
    return path


def _transformed(path, extra_type=None):
    import bpy

    from arena_assets_blender.converter import ModelConverter

    with ModelConverter() as model_converter:
        model_converter.load(str(path))
        bpy.context.view_layer.update()
        objects = {obj.type: obj for obj in bpy.context.scene.objects if obj.name != "Root"}
        mesh = objects["MESH"]
        anchor = None if extra_type is None else mesh.matrix_world.inverted() @ objects[extra_type].matrix_world.translation

        model_converter.transform_coordinates(CoordinateSystem("X+", "Y+", "Z+"))
        bpy.context.view_layer.update()

        offset = None if extra_type is None else tuple(objects[extra_type].matrix_world.translation - mesh.matrix_world @ anchor)
        return model_converter.bounding_box().round(4), offset


@pytest.mark.parametrize("extra_type", ["LIGHT", "EMPTY"])
def test_coordinate_transform_pivot_ignores_lights_and_empties(tmp_path, extra_type):
    pytest.importorskip("bpy")

    plain, _ = _transformed(_write_box(tmp_path / "plain.usda"))
    crowded, offset = _transformed(_write_box(tmp_path / "crowded.usda", EXTRAS[extra_type]), extra_type)

    assert plain == BoundingBox(((-0.5, 0.5), (-3.0, 0.0), (-1.0, 1.0)))
    assert crowded == plain
    assert offset == pytest.approx((0.0, 0.0, 0.0), abs=1e-4)
