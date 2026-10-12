"""Colliders: one collision mesh per object bundle, baked into its USD and SDF models and kept as a plain OBJ."""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import yaml
from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdPhysics, Vt

from arena_assets.impl import ANNOTATION_NAME
from arena_assets.utils.mesh_io import MeshError, read_mesh
from arena_assets.utils.ModelConverter import sdf_model

SUFFIX = ".collider.obj"
FIELD = "collider"
PRIM = "collider"
USD_FORMATS = ("usdz", "usd", "usda", "usdc")
Y_UP_FROM_Z_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


class ColliderError(Exception):
    """A user-facing reason the collider was not written."""


def write_obj(path: Path, points: np.ndarray, triangles: np.ndarray) -> None:
    """Write an OBJ with one normal per face, which DART's mesh collision in Gazebo needs."""
    triangles = np.asarray(triangles)
    corners = points[triangles]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.where(lengths > 0, normals / np.where(lengths > 0, lengths, 1.0), (0.0, 0.0, 1.0))
    faces = np.repeat(np.arange(1, len(triangles) + 1)[:, None], 3, axis=1)
    with open(path, "w") as f:
        np.savetxt(f, points, fmt="v %.6f %.6f %.6f")
        np.savetxt(f, normals, fmt="vn %.6f %.6f %.6f")
        np.savetxt(f, np.stack([triangles + 1, faces], axis=2).reshape(-1, 6), fmt="f %d//%d %d//%d %d//%d")


def usd_models(bundle: Path, name: str) -> list[Path]:
    return [path for ext in USD_FORMATS if (path := bundle / f"{name}.{ext}").is_file()]


def sdf_model_file(bundle: Path, name: str) -> Path | None:
    return next((path for path in (bundle / f"{name}.sdf" / f"{name}.sdf", bundle / f"{name}.sdf") if path.is_file()), None)


def _bundle_to_local(stage: Usd.Stage, root: Usd.Prim) -> np.ndarray:
    """Bundle frame (meters, Z up) to the local frame of root."""
    to_stage = np.eye(4)
    if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y:
        to_stage[:3, :3] = Y_UP_FROM_Z_UP
    to_stage[:3, :3] /= UsdGeom.GetStageMetersPerUnit(stage)
    world = np.asarray(UsdGeom.Xformable(root).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    return np.linalg.inv(world) @ to_stage


def _root(stage: Usd.Stage) -> Usd.Prim:
    root = stage.GetDefaultPrim()
    if not root:
        raise ColliderError(f"{stage.GetRootLayer().identifier} has no default prim, which the collider must sit under to travel with the model")
    return root


def _visual_meshes(stage: Usd.Stage) -> list[Usd.Prim]:
    root = _root(stage)
    collider = root.GetPath().AppendChild(PRIM)
    return [prim for prim in Usd.PrimRange(root) if prim.IsA(UsdGeom.Mesh) and not prim.GetPath().HasPrefix(collider)]


def _collide_visuals(stage: Usd.Stage) -> bool:
    meshes = _visual_meshes(stage)
    for prim in meshes:
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(UsdPhysics.Tokens.none)
    return bool(meshes)


def _enable_visual_collision(stage: Usd.Stage, enabled: bool) -> None:
    for prim in _visual_meshes(stage):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(enabled)


def _author_usd(stage: Usd.Stage, points: np.ndarray, triangles: np.ndarray) -> bool:
    root = _root(stage)
    path = root.GetPath().AppendChild(PRIM)
    stage.RemovePrim(path)
    _enable_visual_collision(stage, False)
    xform = UsdGeom.Xform.Define(stage, path)
    xform.MakeInvisible()
    xform.AddTransformOp().Set(Gf.Matrix4d(_bundle_to_local(stage, root).T.tolist()))
    mesh = UsdGeom.Mesh.Define(stage, path.AppendChild("mesh"))
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(triangles), 3, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(triangles.astype(np.int32).ravel()))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.array([points.min(0), points.max(0)], np.float32)))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.none)
    return True


def _strip_usd(stage: Usd.Stage) -> bool:
    path = _root(stage).GetPath().AppendChild(PRIM)
    if not stage.GetPrimAtPath(path):
        return False
    stage.RemovePrim(path)
    _enable_visual_collision(stage, True)
    return True


def collide_visuals(path: Path) -> bool:
    """Give every visual mesh of a USD model a static triangle-mesh collision."""
    return _edit_usd(path, _collide_visuals)


def _edit_usd(path: Path, edit: Callable[[Usd.Stage], bool]) -> bool:
    """Apply edit to the root layer of a USD file or usdz package and save it when edit reports a change."""
    if path.suffix != ".usdz":
        stage = Usd.Stage.Open(str(path))
        changed = edit(stage)
        if changed:
            stage.GetRootLayer().Save()
        return changed
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if not name.endswith("/")]
            archive.extractall(tmp)
        stage = Usd.Stage.Open(str(Path(tmp) / names[0]))
        changed = edit(stage)
        if not changed:
            return False
        stage.GetRootLayer().Save()
        packed = path.with_name(f".{path.name}.tmp")
        writer = Sdf.ZipFileWriter.CreateNew(str(packed))
        for name in names:
            writer.AddFile(str(Path(tmp) / name), name)
        writer.Save()
        packed.replace(path)
    return True


def _links(tree: ET.ElementTree, sdf: Path) -> list[ET.Element]:
    links = tree.getroot().findall("./model/link")
    if not links:
        raise ColliderError(f"{sdf} has no <model><link> to hold the collision")
    return links


def _collision(name: str, uri: str) -> ET.Element:
    collision = ET.Element("collision", name=name)
    ET.SubElement(ET.SubElement(ET.SubElement(collision, "geometry"), "mesh"), "uri").text = uri
    return collision


def _save_sdf(tree: ET.ElementTree, sdf: Path) -> None:
    ET.indent(tree, "  ")
    tree.write(sdf, encoding="unicode", xml_declaration=True)


def _bake_sdf(sdf: Path, name: str, points: np.ndarray, triangles: np.ndarray) -> None:
    mesh = f"{name}{SUFFIX}"
    tree = ET.parse(sdf)
    links = _links(tree, sdf)
    write_obj(sdf.parent / mesh, points, triangles)
    for link in links:
        for collision in link.findall("collision"):
            link.remove(collision)
    links[0].append(_collision(PRIM, mesh))
    _save_sdf(tree, sdf)


def _strip_sdf(sdf: Path, name: str) -> bool:
    """Swap the collider back for one collision per mesh visual, as the build writes them, and delete an SDF left without visuals."""
    tree = ET.parse(sdf)
    links = _links(tree, sdf)
    found = False
    for link in links:
        for collision in link.findall("collision"):
            if collision.get("name") == PRIM:
                link.remove(collision)
                found = True
    if not found:
        return False
    (sdf.parent / f"{name}{SUFFIX}").unlink(missing_ok=True)
    glow = 0
    visuals = 0
    for link in links:
        for index, visual in reversed(list(enumerate(link))):
            if visual.tag != "visual" or (uri := visual.findtext("./geometry/mesh/uri")) is None:
                continue
            visuals += 1
            link.insert(index + 1, _collision("collision" if visual.get("name") == "visual" else "", uri))
        for collision in link.findall("collision"):
            if not collision.get("name"):
                collision.set("name", f"collision_glow_{glow}")
                glow += 1
    if visuals:
        _save_sdf(tree, sdf)
    else:
        sdf.unlink()
        if sdf.parent.name == f"{name}.sdf" and not any(sdf.parent.iterdir()):
            sdf.parent.rmdir()
    return True


def collision_only_sdf(bundle: Path, name: str) -> Path:
    """Write an SDF model without visuals for a bundle that has none, ready for bake to add the collision."""
    path = bundle / f"{name}.sdf" / f"{name}.sdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sdf_model(name, plain=False))
    return path


def _check(bundle: Path, name: str) -> None:
    for path in usd_models(bundle, name):
        try:
            stage = Usd.Stage.Open(str(path))
        except Tf.ErrorException:
            raise ColliderError(f"{path} is not a readable USD stage") from None
        _root(stage)
    if (sdf := sdf_model_file(bundle, name)) is not None:
        try:
            tree = ET.parse(sdf)
        except ET.ParseError as error:
            raise ColliderError(f"{sdf} is not readable XML ({error})") from None
        _links(tree, sdf)


def bake(bundle: Path, name: str, points: np.ndarray, triangles: np.ndarray) -> list[Path]:
    """Write <name>.collider.obj and bake the mesh into the bundle's USD and SDF models, return the models changed."""
    _check(bundle, name)
    write_obj(bundle / f"{name}{SUFFIX}", points, triangles)
    baked = [path for path in usd_models(bundle, name) if _edit_usd(path, lambda stage: _author_usd(stage, points, triangles))]
    if (sdf := sdf_model_file(bundle, name)) is not None:
        _bake_sdf(sdf, name, points, triangles)
        baked.append(sdf)
    return baked


def strip(bundle: Path, name: str) -> list[Path]:
    """Take the collider out of the bundle's models and delete its OBJ, return the models changed."""
    _check(bundle, name)
    (bundle / f"{name}{SUFFIX}").unlink(missing_ok=True)
    stripped = [path for path in usd_models(bundle, name) if _edit_usd(path, _strip_usd)]
    if (sdf := sdf_model_file(bundle, name)) is not None and _strip_sdf(sdf, name):
        stripped.append(sdf)
    return stripped


def _annotation(bundle: Path) -> tuple[Path, dict]:
    path = bundle / ANNOTATION_NAME
    if not path.is_file():
        raise ColliderError(f"{bundle} has no {ANNOTATION_NAME}, pass an object bundle directory (the one 'arena asset pull object NAME' or 'arena-assets splat' writes)")
    annotation = yaml.safe_load(path.read_text()) or {}
    if not annotation.get("name"):
        raise ColliderError(f"{path} names no asset ('name:' is missing), which the model files and the collider are named after")
    return path, annotation


def attach_mesh(bundle: Path, mesh: Path, matrix: np.ndarray | None = None) -> tuple[list[Path], int]:
    """Read an .obj or .ply mesh, map it into the bundle frame with matrix, bake it into the bundle and name it in annotation.yaml."""
    annotation_path, annotation = _annotation(bundle)
    name = annotation["name"]
    if not usd_models(bundle, name) and sdf_model_file(bundle, name) is None:
        raise ColliderError(f"{bundle} holds none of {', '.join(f'{name}.{ext}' for ext in USD_FORMATS)} or {name}.sdf/{name}.sdf to bake the collider into, pull the bundle with its USD or SDF model")
    if not mesh.is_file():
        raise ColliderError(f"MESH {mesh} does not exist, pass an .obj or .ply triangle mesh")
    try:
        points, triangles = read_mesh(mesh)
    except MeshError as error:
        raise ColliderError(str(error)) from None
    if matrix is not None:
        points = points @ matrix[:3, :3].T + matrix[:3, 3]
    baked = bake(bundle, name, points, triangles)
    annotation[FIELD] = f"{name}{SUFFIX}"
    annotation_path.write_text(yaml.safe_dump(annotation))
    return baked, len(triangles)


def remove_collider(bundle: Path) -> list[Path] | None:
    """Take the collider out of the bundle and its annotation entry, None when the annotation names none."""
    annotation_path, annotation = _annotation(bundle)
    if not annotation.pop(FIELD, None):
        return None
    stripped = strip(bundle, annotation["name"])
    annotation_path.write_text(yaml.safe_dump(annotation))
    return stripped
