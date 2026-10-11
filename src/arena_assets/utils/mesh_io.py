"""Read triangle meshes from Wavefront OBJ and Stanford PLY files."""

from __future__ import annotations

from pathlib import Path

import numpy as np

PLY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}
PLY_FORMATS = {"ascii": "", "binary_little_endian": "<", "binary_big_endian": ">"}
FORMATS = (".obj", ".ply")


class MeshError(ValueError):
    """The file is not a mesh this reader understands."""


def _fan(polygons: list[list[int]]) -> np.ndarray:
    triangles = [(polygon[0], polygon[i], polygon[i + 1]) for polygon in polygons for i in range(1, len(polygon) - 1)]
    return np.array(triangles, np.int64).reshape(-1, 3)


def _read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    points: list[tuple[float, float, float]] = []
    polygons: list[list[int]] = []
    for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        tokens = line.split()
        if not tokens or tokens[0] not in ("v", "f"):
            continue
        try:
            if tokens[0] == "v":
                points.append((float(tokens[1]), float(tokens[2]), float(tokens[3])))
            else:
                indices = [int(token.split("/")[0]) for token in tokens[1:]]
                polygons.append([index - 1 if index > 0 else len(points) + index for index in indices])
        except (ValueError, IndexError):
            raise MeshError(f"{path}:{number} is no OBJ vertex ('v X Y Z') or face ('f A B C ...') line: {line.strip()!r}") from None
    return np.array(points, np.float64).reshape(-1, 3), _fan(polygons)


def _ply_header(data: bytes, path: Path) -> tuple[str, list[tuple[str, int, list[tuple[str, ...]]]], int]:
    end = data.find(b"end_header")
    newline = data.find(b"\n", end)
    if not data.startswith(b"ply") or end < 0 or newline < 0:
        raise MeshError(f"{path} has no PLY header (it must start with 'ply' and hold an 'end_header' line)")
    endian = None
    elements: list[tuple[str, int, list[tuple[str, ...]]]] = []
    for line in data[:end].decode("ascii", "replace").splitlines():
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == "format":
            if tokens[1] not in PLY_FORMATS:
                raise MeshError(f"{path} has the unknown PLY format {tokens[1]!r}, expected one of {', '.join(PLY_FORMATS)}")
            endian = PLY_FORMATS[tokens[1]]
        elif tokens[0] == "element":
            elements.append((tokens[1], int(tokens[2]), []))
        elif tokens[0] == "property" and elements:
            elements[-1][2].append(tuple(tokens[1:]))
    if endian is None:
        raise MeshError(f"{path} has no PLY 'format' line")
    return endian, elements, newline + 1


def _ply_dtype(properties: list[tuple[str, ...]], endian: str, path: Path) -> np.dtype | None:
    if any(prop[0] == "list" for prop in properties):
        return None
    try:
        return np.dtype([(prop[1], (endian or "<") + PLY_TYPES[prop[0]]) for prop in properties])
    except KeyError as error:
        raise MeshError(f"{path} uses the unknown PLY property type {error.args[0]!r}") from None


def _ply_lists_binary(data: bytes, offset: int, count: int, properties: list[tuple[str, ...]], endian: str, path: Path) -> tuple[list[list[int]], int]:
    if len(properties) != 1 or properties[0][0] != "list":
        raise MeshError(f"{path} has a binary PLY list element with more than its one list property, which this reader does not take")
    count_type, item_type = np.dtype(endian + PLY_TYPES[properties[0][1]]), np.dtype(endian + PLY_TYPES[properties[0][2]])
    uniform = np.dtype([("n", count_type), ("items", item_type, (3,))])
    if offset + count * uniform.itemsize <= len(data):
        rows = np.frombuffer(data, uniform, count, offset)
        if np.all(rows["n"] == 3):
            return rows["items"].astype(np.int64).tolist(), offset + count * uniform.itemsize
    rows = []
    for _ in range(count):
        size = int(np.frombuffer(data, count_type, 1, offset)[0])
        offset += count_type.itemsize
        rows.append(np.frombuffer(data, item_type, size, offset).astype(np.int64).tolist())
        offset += size * item_type.itemsize
    return rows, offset


def _read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = path.read_bytes()
    endian, elements, offset = _ply_header(data, path)
    points = np.zeros((0, 3))
    polygons: list[list[int]] = []
    lines = data[offset:].decode("ascii", "replace").splitlines() if not endian else []
    line = 0
    try:
        for name, count, properties in elements:
            dtype = _ply_dtype(properties, endian, path)
            if not endian:
                rows = [lines[line + index].split() for index in range(count)]
                line += count
                if name == "vertex":
                    columns = [prop[1] for prop in properties]
                    points = np.array([[float(row[columns.index(axis)]) for axis in "xyz"] for row in rows], np.float64).reshape(-1, 3)
                elif name == "face":
                    polygons = [[int(token) for token in row[1 : 1 + int(row[0])]] for row in rows]
            elif dtype is not None:
                table = np.frombuffer(data, dtype, count, offset)
                offset += count * dtype.itemsize
                if name == "vertex":
                    points = np.stack([table[axis].astype(np.float64) for axis in "xyz"], axis=1)
            else:
                rows, offset = _ply_lists_binary(data, offset, count, properties, endian, path)
                if name == "face":
                    polygons = rows
    except MeshError:
        raise
    except (ValueError, IndexError, KeyError):
        raise MeshError(f"{path} ends early or lacks x, y, z vertex properties: its PLY body does not match its header") from None
    return points, _fan(polygons)


def read_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read points (N, 3) and triangles (M, 3) from an .obj or .ply file, polygons fanned into triangles."""
    suffix = path.suffix.lower()
    if suffix not in FORMATS:
        raise MeshError(f"{path} is not one of the mesh formats {', '.join(FORMATS)}")
    points, triangles = _read_obj(path) if suffix == ".obj" else _read_ply(path)
    if not len(points) or not len(triangles):
        raise MeshError(f"{path} holds {len(points)} vertices and {len(triangles)} triangles, a mesh needs both")
    if triangles.min() < 0 or triangles.max() >= len(points):
        raise MeshError(f"{path} has a face that names a vertex outside its {len(points)} vertices")
    return points, triangles
