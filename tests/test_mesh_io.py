import struct
from pathlib import Path

import numpy as np
import pytest

from arena_assets.utils.mesh_io import MeshError, read_mesh

POINTS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.5], [0.0, 1.0, 0.5], [0.5, 0.5, 2.0]])
QUAD = [0, 1, 2, 3]
TRIANGLES = [[0, 1, 4], [1, 2, 4], [2, 3, 4]]
FANNED = np.array([[0, 1, 2], [0, 2, 3], *TRIANGLES])


def ply_header(fmt: str, vertex_properties: list[str], faces: int) -> bytes:
    lines = ["ply", f"format {fmt} 1.0", "comment made by a test", f"element vertex {len(POINTS)}", *(f"property {prop}" for prop in vertex_properties), f"element face {faces}", "property list uchar int vertex_indices", "end_header"]
    return ("\n".join(lines) + "\n").encode()


def binary_ply(path: Path, endian: str, polygons: list[list[int]]) -> Path:
    body = b"".join(struct.pack(f"{endian}3f", *point) for point in POINTS)
    body += b"".join(struct.pack(f"{endian}B{len(polygon)}i", len(polygon), *polygon) for polygon in polygons)
    fmt = "binary_little_endian" if endian == "<" else "binary_big_endian"
    path.write_bytes(ply_header(fmt, ["float x", "float y", "float z"], len(polygons)) + body)
    return path


def test_obj_fans_polygons_and_takes_slashed_and_negative_indices(tmp_path):
    path = tmp_path / "mesh.obj"
    lines = ["# a comment", "o thing", *(f"v {x} {y} {z}" for x, y, z in POINTS), "vt 0 0", "vn 0 0 1", "f 1/1/1 2/1/1 3/1/1 4/1/1", "f 1 2 5", "f 2//1 3//1 5//1", "f -3 -2 -1"]
    path.write_text("\n".join(lines) + "\n")
    points, triangles = read_mesh(path)
    np.testing.assert_array_equal(points, POINTS)
    np.testing.assert_array_equal(triangles, FANNED)


def test_ascii_ply_reads_xyz_among_other_vertex_properties(tmp_path):
    path = tmp_path / "mesh.ply"
    rows = [f"{nx} {x} {y} {z} 255" for nx, (x, y, z) in enumerate(POINTS)]
    faces = ["4 0 1 2 3", *(f"3 {a} {b} {c}" for a, b, c in TRIANGLES)]
    path.write_bytes(ply_header("ascii", ["float nx", "float x", "float y", "float z", "uchar red"], 4) + ("\n".join(rows + faces) + "\n").encode())
    points, triangles = read_mesh(path)
    np.testing.assert_array_equal(points, POINTS)
    np.testing.assert_array_equal(triangles, FANNED)


@pytest.mark.parametrize("endian", ["<", ">"])
def test_binary_ply_with_only_triangles(tmp_path, endian):
    points, triangles = read_mesh(binary_ply(tmp_path / "mesh.ply", endian, TRIANGLES))
    np.testing.assert_array_equal(points, POINTS)
    np.testing.assert_array_equal(triangles, TRIANGLES)


@pytest.mark.parametrize("endian", ["<", ">"])
def test_binary_ply_with_mixed_polygon_sizes(tmp_path, endian):
    points, triangles = read_mesh(binary_ply(tmp_path / "mesh.ply", endian, [QUAD, *TRIANGLES]))
    np.testing.assert_array_equal(points, POINTS)
    np.testing.assert_array_equal(triangles, FANNED)


def test_binary_ply_skips_a_scalar_element_before_the_faces(tmp_path):
    path = tmp_path / "mesh.ply"
    header = ply_header("binary_little_endian", ["double x", "double y", "double z"], len(TRIANGLES)).replace(b"element face", b"element extra 2\nproperty short tag\nelement face")
    body = b"".join(struct.pack("<3d", *point) for point in POINTS) + struct.pack("<2h", 7, 9) + b"".join(struct.pack("<B3i", 3, *triangle) for triangle in TRIANGLES)
    path.write_bytes(header + body)
    points, triangles = read_mesh(path)
    np.testing.assert_array_equal(points, POINTS)
    np.testing.assert_array_equal(triangles, TRIANGLES)


def test_unknown_suffix_is_refused(tmp_path):
    path = tmp_path / "mesh.stl"
    path.write_text("solid")
    with pytest.raises(MeshError, match=r"not one of the mesh formats \.obj, \.ply"):
        read_mesh(path)


def test_mesh_without_faces_is_refused(tmp_path):
    path = tmp_path / "cloud.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\n")
    with pytest.raises(MeshError, match="holds 3 vertices and 0 triangles"):
        read_mesh(path)


def test_face_naming_a_missing_vertex_is_refused(tmp_path):
    path = tmp_path / "mesh.obj"
    path.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 9\n")
    with pytest.raises(MeshError, match="names a vertex outside its 3 vertices"):
        read_mesh(path)


def test_malformed_obj_line_names_its_line(tmp_path):
    path = tmp_path / "mesh.obj"
    path.write_text("v 0 0 0\nv 1 zero 0\n")
    with pytest.raises(MeshError, match=r"mesh\.obj:2 is no OBJ vertex"):
        read_mesh(path)


def test_truncated_binary_ply_is_refused(tmp_path):
    path = binary_ply(tmp_path / "mesh.ply", "<", TRIANGLES)
    path.write_bytes(path.read_bytes()[:-20])
    with pytest.raises(MeshError, match="does not match its header"):
        read_mesh(path)


def test_file_without_ply_header_is_refused(tmp_path):
    path = tmp_path / "mesh.ply"
    path.write_bytes(b"not a mesh at all")
    with pytest.raises(MeshError, match="has no PLY header"):
        read_mesh(path)
