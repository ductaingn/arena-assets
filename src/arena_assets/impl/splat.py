"""Object bundles from OpenUSD Gaussian splats, anchored to a wall or standing on the floor."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol, Self

import attrs
import numpy as np
import yaml
from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdVol, Vt

from arena_assets import semantic
from arena_assets.impl import ANNOTATION_NAME, Face, collider, converter
from arena_assets.impl.build.ObjectDatabaseBuilder import ObjectAnnotation
from arena_assets.utils.geom import BoundingBox
from arena_assets.utils.mesh_io import MeshError, read_mesh

EXTENT_PAD_SCALES = 3.0
COLLINEAR_RATIO = 1e-6


class _VtArray(Protocol):
    def __init__(self, values: list[object], /) -> None: ...

    @classmethod
    def FromNumpy(cls, array: np.ndarray, /) -> Self: ...

    def __len__(self) -> int: ...

    def __getitem__(self, index: int, /) -> object: ...


class SplatError(Exception):
    """A user-facing reason the bundle was not written."""


def _check_size(width: float, height: float, floor: float) -> None:
    if width <= 0:
        raise SplatError(f"--width must be positive (meters, X from -width/2 to +width/2), got {width:g}")
    if height <= floor:
        raise SplatError(f"--height ({height:g}) must be above --floor ({floor:g}): the crop spans Z from --floor to --height in meters")


@attrs.frozen
class WallCrop:
    """Kept region in the wall frame, meters, with Y = 0 on the captured wall plane."""

    width: float
    height: float
    floor: float = 0.03
    depth: float = 0.05
    behind: float = 0.0
    frame = "the wall frame is meters, origin at the base midpoint of the captured wall, X along the wall, Y into the room from the captured wall plane, Z up"

    def __attrs_post_init__(self) -> None:
        _check_size(self.width, self.height, self.floor)
        if self.depth <= -self.behind:
            raise SplatError(f"--depth ({self.depth:g}) must be above -behind ({-self.behind:g}): the crop spans Y from -behind to --depth in meters")

    @property
    def bounding_box(self) -> BoundingBox:
        return BoundingBox(((-self.width / 2, self.width / 2), (-self.behind, self.depth), (self.floor, self.height)))

    @property
    def bundle_bounding_box(self) -> BoundingBox:
        """The crop in the bundle frame, whose Y = 0 is the back of the crop."""
        return BoundingBox(((-self.width / 2, self.width / 2), (0.0, self.behind + self.depth), (self.floor, self.height)))

    @property
    def to_bundle(self) -> np.ndarray:
        """Wall frame to bundle frame, a shift by +behind along Y."""
        shift = np.eye(4)
        shift[1, 3] = self.behind
        return shift

    @property
    def desc(self) -> str:
        return f"Gaussian splat capture of a wall, {self.width:g} m wide and {self.height:g} m high, {self.behind + self.depth:g} m deep."


@attrs.frozen
class ObjectCrop:
    """Kept region in the object frame, a box centered over the origin on the floor."""

    width: float
    depth: float
    height: float
    floor: float = 0.03
    frame = "the object frame is meters, origin on the floor under the object center, X and Y level with Y toward the object front, Z up"

    def __attrs_post_init__(self) -> None:
        _check_size(self.width, self.height, self.floor)
        if self.depth <= 0:
            raise SplatError(f"--depth must be positive (meters, Y from -depth/2 to +depth/2), got {self.depth:g}")

    @property
    def bounding_box(self) -> BoundingBox:
        return BoundingBox(((-self.width / 2, self.width / 2), (-self.depth / 2, self.depth / 2), (self.floor, self.height)))

    @property
    def bundle_bounding_box(self) -> BoundingBox:
        return self.bounding_box

    @property
    def to_bundle(self) -> np.ndarray:
        return np.eye(4)

    @property
    def desc(self) -> str:
        return f"Gaussian splat capture of an object, {self.width:g} m wide, {self.depth:g} m deep and {self.height:g} m high."


Crop = WallCrop | ObjectCrop


@attrs.frozen
class Similarity:
    """Fitted source-to-target transform, p_target = matrix @ p_stage."""

    matrix: np.ndarray
    scale: float
    rms: float
    pairs: int


@attrs.frozen
class SplatBundle:
    path: Path
    kept: int
    total: int
    collider: str = ""
    collider_kept: int = 0
    collider_total: int = 0


@attrs.frozen
class _Field:
    prim: Usd.Prim
    to_stage: np.ndarray
    positions: np.ndarray
    scales: np.ndarray | None
    keep: np.ndarray


def parse_numbers(text: str, count: int, what: str) -> list[float]:
    """Parse exactly count finite numbers separated by spaces or commas."""
    tokens = text.replace(",", " ").split()
    try:
        values = [float(token) for token in tokens]
    except ValueError:
        raise SplatError(f"{what} takes {count} numbers separated by spaces or commas, got {text!r}") from None
    if len(values) != count or not all(math.isfinite(value) for value in values):
        raise SplatError(f"{what} takes {count} finite numbers separated by spaces or commas, got {len(values)} in {text!r}")
    return values


def parse_matrix(text: str, frames: str = "the source stage frame to the target frame") -> np.ndarray:
    """Parse a row-major 4x4 affine matrix that maps between the named frames."""
    matrix = np.array(parse_numbers(text, 16, f"--matrix (row-major 4x4, {frames} in meters)")).reshape(4, 4)
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0)):
        raise SplatError(f"--matrix must end with the row 0 0 0 1 (row-major, translation in the last column), got {' '.join(f'{value:g}' for value in matrix[3])}: a transposed matrix puts the translation there")
    if abs(np.linalg.det(matrix[:3, :3])) < 1e-12:
        raise SplatError(f"--matrix has a singular 3x3 part, it must hold the rotation and scale from {frames}")
    return matrix


def parse_drop(text: str) -> BoundingBox:
    """Parse a target-frame box given as XMIN YMIN ZMIN XMAX YMAX ZMAX."""
    values = parse_numbers(text, 6, "--drop (target-frame box XMIN YMIN ZMIN XMAX YMAX ZMAX in meters)")
    box = BoundingBox(zip(values[:3], values[3:], strict=True))
    if any(low > high for low, high in box):
        raise SplatError(f"--drop takes the minimum corner first (XMIN YMIN ZMIN XMAX YMAX ZMAX), got {text!r}")
    return box


def read_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read 'sx sy sz tx ty tz' correspondences, skipping blank and # lines."""
    if not path.is_file():
        raise SplatError(f"--points file {path} does not exist, write one line 'sx sy sz tx ty tz' per correspondence (source stage point, target-frame point in meters)")
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        rows.append(parse_numbers(line, 6, f"{path}:{number} (sx sy sz tx ty tz: source stage point, target-frame point in meters)"))
    pairs = np.array(rows, np.float64).reshape(-1, 6)
    return pairs[:, :3], pairs[:, 3:]


def _collinear(points: np.ndarray) -> bool:
    singular = np.linalg.svd(points - points.mean(0), compute_uv=False)
    return bool(singular[1] <= COLLINEAR_RATIO * singular[0])


def fit_similarity(source: np.ndarray, target: np.ndarray) -> Similarity:
    """Umeyama fit of uniform scale, rotation and translation from source to target points."""
    count = len(source)
    if count < 3:
        raise SplatError(f"--points holds {count} pairs, a similarity fit needs at least 3 non-collinear lines 'sx sy sz tx ty tz' (source stage point, target-frame point in meters)")
    if _collinear(source) or _collinear(target):
        raise SplatError(f"--points holds {count} collinear pairs, which leave the rotation about their line open: add a pair off that line (for example one higher up)")
    source_mean, target_mean = source.mean(0), target.mean(0)
    source_centered, target_centered = source - source_mean, target - target_mean
    u, singular, vt = np.linalg.svd(target_centered.T @ source_centered / count)
    signs = np.array([1.0, 1.0, np.sign(np.linalg.det(u) * np.linalg.det(vt))])
    rotation = u @ np.diag(signs) @ vt
    scale = float((singular * signs).sum() / (source_centered**2).sum() * count)
    matrix = np.eye(4)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = target_mean - scale * rotation @ source_mean
    residual = source @ matrix[:3, :3].T + matrix[:3, 3] - target
    return Similarity(matrix=matrix, scale=scale, rms=float(np.sqrt((residual**2).sum(1).mean())), pairs=count)


def particle_fields(stage: Usd.Stage) -> list[Usd.Prim]:
    return [prim for prim in stage.Traverse() if prim.IsA(UsdVol.ParticleField)]


def _per_particle(prim: Usd.Prim, name: str) -> np.ndarray | None:
    for candidate in (name, f"{name}h"):
        attribute = prim.GetAttribute(candidate)
        value = attribute.Get() if attribute.IsValid() else None
        if value is not None and len(value):
            return np.asarray(value, np.float64)
    return None


def _inside(points: np.ndarray, box: BoundingBox) -> np.ndarray:
    low, high = np.array(box).T
    return np.all((points >= low) & (points <= high), axis=1)


def _select(prim: Usd.Prim, to_target: np.ndarray, crop: Crop, drops: list[BoundingBox], max_scale: float, min_opacity: float) -> _Field | None:
    positions = _per_particle(prim, "positions")
    if positions is None:
        return None
    to_stage = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
    field_to_target = to_target @ to_stage
    target = positions @ field_to_target[:3, :3].T + field_to_target[:3, 3]
    keep = _inside(target, crop.bounding_box)
    for box in drops:
        keep &= ~_inside(target, box)
    scales = _per_particle(prim, "scales")
    if scales is not None:
        meters = abs(np.linalg.det(field_to_target[:3, :3])) ** (1 / 3)
        keep &= scales.max(1) * meters <= max_scale
    opacities = _per_particle(prim, "opacities")
    if opacities is not None:
        keep &= opacities >= min_opacity
    return _Field(prim=prim, to_stage=to_stage, positions=positions, scales=scales, keep=keep)


def _filtered(value: _VtArray, kept: np.ndarray, count: int) -> _VtArray:
    array = np.asarray(value)
    stride = len(array) // count
    pick = (kept[:, None] * stride + np.arange(stride)).ravel()
    if array.dtype.kind in "fiub":
        return type(value).FromNumpy(array[pick])
    return type(value)([value[int(index)] for index in pick])


def _copy_field(field: _Field, target: Usd.Prim) -> None:
    kept = np.nonzero(field.keep)[0]
    count = len(field.keep)
    UsdGeom.Xformable(target).AddTransformOp().Set(Gf.Matrix4d(field.to_stage.T.tolist()))
    for schema in field.prim.GetPrimTypeInfo().GetAppliedAPISchemas():
        target.AddAppliedSchema(schema)
    for attribute in field.prim.GetAuthoredAttributes():
        name = attribute.GetName()
        if name.startswith("xformOp") or name == "extent":
            continue
        value = attribute.Get()
        copy = target.CreateAttribute(name, attribute.GetTypeName(), custom=attribute.IsCustom())
        if value is None:
            continue
        if attribute.GetTypeName().isArray and len(value) and len(value) % count == 0:
            value = _filtered(value, kept, count)
        copy.Set(value)
    positions = field.positions[kept]
    pad = EXTENT_PAD_SCALES * float(field.scales[kept].max()) if field.scales is not None else 0.0
    extent = np.array([positions.min(0) - pad, positions.max(0) + pad], np.float32)
    UsdGeom.Boundable(target).CreateExtentAttr(Vt.Vec3fArray.FromNumpy(extent))


def _select_collider(path: Path, to_target: np.ndarray, crop: Crop, drops: list[BoundingBox]) -> tuple[np.ndarray, np.ndarray, int]:
    """Bundle-frame points and triangles of the mesh part inside the crop and outside the drops, plus the source triangle count."""
    if not path.is_file():
        raise SplatError(f"--collider-mesh {path} does not exist, pass an .obj or .ply mesh in the source stage frame")
    try:
        points, triangles = read_mesh(path)
    except MeshError as error:
        raise SplatError(f"--collider-mesh: {error}") from None
    target = points @ to_target[:3, :3].T + to_target[:3, 3]
    keep = _inside(target, crop.bounding_box)[triangles].all(1)
    for box in drops:
        keep &= ~_inside(target, box)[triangles].any(1)
    if not keep.any():
        raise SplatError(f"the crop keeps 0 of {len(triangles)} collider triangles: the mesh must share the source stage frame of SOURCE ({crop.frame}), and a triangle stays only with all its corners inside the crop and none inside a --drop box")
    used, kept = np.unique(triangles[keep], return_inverse=True)
    to_bundle = crop.to_bundle
    return target[used] @ to_bundle[:3, :3].T + to_bundle[:3, 3], kept.reshape(-1, 3), len(triangles)


def frame_splat(
    source: Path,
    out_dir: Path,
    name: str,
    to_target: np.ndarray,
    crop: Crop,
    drops: list[BoundingBox],
    max_scale: float = 0.2,
    min_opacity: float = 0.0,
    collider_mesh: Path | None = None,
) -> SplatBundle:
    """Write <name>.gaussians.usdc and its Z-up wrapper <name>.usda in the bundle frame of the crop, with the optional collider baked in."""
    if not Sdf.Path.IsValidIdentifier(name):
        raise SplatError(f"--name {name!r} is not a valid USD prim name (letters, digits and underscores, not starting with a digit), pass --name or rename OUT_DIR")
    if not source.is_file():
        raise SplatError(f"SOURCE {source} does not exist, pass the USD file 3DGRUT wrote (export_usd.enabled=true)")
    try:
        stage = Usd.Stage.Open(str(source))
    except Tf.ErrorException:
        raise SplatError(f"SOURCE {source} is not a readable USD stage, pass the .usd, .usda, .usdc or .usdz file 3DGRUT wrote (export_usd.enabled=true)") from None
    fields = [field for prim in particle_fields(stage) if (field := _select(prim, to_target, crop, drops, max_scale, min_opacity)) is not None]
    total = sum(len(field.keep) for field in fields)
    kept = sum(int(field.keep.sum()) for field in fields)
    if not total:
        raise SplatError(f"SOURCE {source} has no ParticleField prim with positions, pass a Gaussian splat exported as OpenUSD (3DGRUT with export_usd.enabled=true)")
    if not kept:
        raise SplatError(f"the crop keeps 0 of {total} particles: check the registration ({crop.frame}), then widen --width, --height or --depth, or relax --max-scale and --min-opacity")

    collision = _select_collider(collider_mesh, to_target, crop, drops) if collider_mesh is not None else None

    out_dir.mkdir(parents=True, exist_ok=True)
    gaussians = Usd.Stage.CreateNew(str(out_dir / f"{name}.gaussians.usdc"))
    UsdGeom.SetStageUpAxis(gaussians, UsdGeom.GetStageUpAxis(stage))
    UsdGeom.SetStageMetersPerUnit(gaussians, UsdGeom.GetStageMetersPerUnit(stage))
    gaussians.SetDefaultPrim(UsdGeom.Xform.Define(gaussians, "/Gaussians").GetPrim())
    for index, field in enumerate(fields):
        if field.keep.any():
            _copy_field(field, gaussians.DefinePrim(f"/Gaussians/Field_{index}", field.prim.GetTypeName()))
    gaussians.GetRootLayer().Save()

    path = out_dir / f"{name}.usda"
    wrapper = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(wrapper, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(wrapper, 1.0)
    wrapper.SetDefaultPrim(UsdGeom.Xform.Define(wrapper, f"/{name}").GetPrim())
    body = UsdGeom.Xform.Define(wrapper, f"/{name}/splat")
    body.AddTransformOp().Set(Gf.Matrix4d((crop.to_bundle @ to_target).T.tolist()))
    body.GetPrim().GetReferences().AddReference(f"./{name}.gaussians.usdc")
    wrapper.GetRootLayer().Save()
    collider.strip(out_dir, name)
    if collision is None:
        return SplatBundle(path=path, kept=kept, total=total)
    collider.collision_only_sdf(out_dir, name)
    collider.bake(out_dir, name, collision[0], collision[1])
    return SplatBundle(path=path, kept=kept, total=total, collider=f"{name}{collider.SUFFIX}", collider_kept=len(collision[1]), collider_total=collision[2])


def write_annotation(out_dir: Path, name: str, crop: Crop, desc: str, source: str, collider_file: str = "") -> Path:
    """Write the bundle's annotation.yaml with the bundle-frame crop as its bounding box."""
    annotation = ObjectAnnotation(
        name=name,
        path=f"Object/{name}",
        desc=desc,
        tags=["domain::common", "kind::splat"],
        asa=semantic.CURRENT_ASA,
        bounding_box=crop.bundle_bounding_box.round(),
        face=Face.POS_Y,
        collider=collider_file,
    )
    path = out_dir / ANNOTATION_NAME
    with open(path, "w") as f:
        yaml.safe_dump({**converter.unstructure(annotation), "source": source}, f)
    return path
