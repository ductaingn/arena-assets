"""Gazebo Fuel models built into Common/Object items by the object database builder, Google Scanned Objects first."""

from __future__ import annotations

import json
import math
import os
import posixpath
import re
import shutil
import typing
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterator
from pathlib import Path

import attrs
import yaml

from arena_assets.impl import ANNOTATION_NAME, AssetType
from arena_assets.impl.build import DatabaseBuilder, OverwriteMode
from arena_assets.importers import IMPORTERS, Attribution, Importer
from arena_assets.importers.providers import Fuel
from arena_assets.semantic import fixes
from arena_assets.semantic.taxonomy import load_spec
from arena_assets.utils.ModelConverter import BlenderWorker

DOMAIN = "Common"
ASSET_TYPE = "Object"
VERSION = "tip"
FORMATS = "usdz,sdf,fbx"
SDF = "model.sdf"
CONFIG = "model.config"
THUMBNAILS = "thumbnails/"
MESHES = (".obj", ".dae")
TEXTURES = (".png", ".jpg", ".jpeg", ".tga", ".bmp", ".tif", ".tiff", ".dds")
MODEL_URI = "model://"

Vector = tuple[float, float, float]
Rows = tuple[Vector, Vector, Vector]

IDENTITY: Rows = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
LUMINANCE = (0.212671, 0.715160, 0.072169)
SHADERS = ("phong", "lambert", "blinn", "constant")

_TEXTURE_MAP = re.compile(r"^(\s*map_\w+\s+)(\S+)(\s*)$")
_MATERIAL_LIBRARY = re.compile(r"^mtllib[ \t]+(.+?)[ \t\r]*$", re.MULTILINE)
_XML_NAME = re.compile(r"[A-Za-z_][\w.-]*")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


@attrs.frozen
class Visual:
    """The one mesh a Fuel model shows and its placement in the model frame."""

    mesh: str
    rotation: Rows = IDENTITY
    scale: Vector = (1.0, 1.0, 1.0)
    translation: Vector = (0.0, 0.0, 0.0)
    albedo: str | None = None

    @property
    def placed(self) -> bool:
        return (self.rotation, self.scale, self.translation) != (IDENTITY, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0))

    @property
    def linear(self) -> Rows:
        return typing.cast(Rows, tuple(tuple(value * factor for value, factor in zip(row, self.scale, strict=True)) for row in self.rotation))

    def point(self, vector: Vector) -> Vector:
        x, y, z = _apply(self.linear, vector)
        return (x + self.translation[0], y + self.translation[1], z + self.translation[2])

    def normal(self, vector: Vector) -> Vector:
        return _apply(self.rotation, (vector[0] / self.scale[0], vector[1] / self.scale[1], vector[2] / self.scale[2]))


class FuelObjects(Importer):
    """Models of one Fuel owner, each staged from the mesh its SDF shows and built by the object database builder."""

    OWNER: typing.ClassVar[str]
    ORIGIN: typing.ClassVar[str]
    LICENSE: typing.ClassVar[str]
    AUTHORS: typing.ClassVar[str]
    CITATION: typing.ClassVar[str] = ""
    LABELS: typing.ClassVar[tuple[str, ...]] = ("categories",)

    provider: Fuel

    @classmethod
    def default_provider(cls) -> Fuel:
        return Fuel(cls.OWNER)

    def items(self) -> Iterator[str]:
        return iter(self.provider.list())

    def model(self, item: str) -> str:
        """Fuel model name of an item."""
        return item

    def asset(self, item: str) -> str:
        """Name the built object gets."""
        return item

    def license(self, item: str) -> str:
        return self.LICENSE

    def authors(self, item: str) -> str:
        return self.AUTHORS

    def fetch(self, item: str) -> Path:
        name = self.model(item)
        self.provider.get(name, self.cache / f"{item}.json")
        return self.provider.get(f"{name}/{VERSION}/{name}.zip", self.cache / f"{item}.zip")

    def convert(self, item: str, src: Path, out: Path) -> str:
        name = self.asset(item)
        relpath = Path(DOMAIN) / ASSET_TYPE / name
        target = out / relpath
        if target.exists():
            shutil.rmtree(target)
        dataset = self.cache / "stage" / name
        try:
            measured(self.stage(name, src, dataset), name)
            builder = DatabaseBuilder.Builder(AssetType.OBJECT)(input_path=dataset / DOMAIN, output_path=out / DOMAIN, overwrite=OverwriteMode.OVERWRITE)
            builder.enable("formats", FORMATS)
            builder.enable("previews")
            builder.build()
        finally:
            shutil.rmtree(dataset, ignore_errors=True)
        if not (target / ANNOTATION_NAME).is_file():
            raise RuntimeError(f"object build failed for {name}, the build log above names the step")
        return str(relpath)

    def attribution(self, item: str) -> Attribution:
        return Attribution(source=self.meta.name, url=self.provider.url(self.model(item)), license=self.license(item), authors=self.authors(item), citation=self.CITATION)

    @classmethod
    def stage(cls, item: str, src: Path, dataset: Path) -> Path:
        """Lay one Fuel zip out as object builder input under dataset/Common/Object/<item> and return that directory."""
        source = dataset / DOMAIN / ASSET_TYPE / item
        if dataset.exists():
            shutil.rmtree(dataset)
        source.mkdir(parents=True)
        with zipfile.ZipFile(src) as archive:
            visual = shown(archive, item)
            needed = {visual.mesh, visual.albedo, *_material_libraries(archive, visual.mesh)}
            archive.extractall(source, [name for name in archive.namelist() if name in needed or _texture(name)])
            description = _description(archive, item)
        mesh = source / visual.mesh
        mesh = mesh.rename(mesh.with_suffix(mesh.suffix.lower()))
        for mtl in sorted(source.rglob("*.mtl")):
            _relink_textures(mtl, source)
        if mesh.suffix == ".obj":
            _place_obj(mesh, visual)
        else:
            _place_dae(mesh, visual, source / visual.albedo if visual.albedo else None)
        metadata = _metadata(src.with_suffix(".json"))
        labels = dict.fromkeys(slug for key in cls.LABELS for slug in _slugs(metadata.get(key)))
        annotate(source / ANNOTATION_NAME, item, description, [cls.ORIGIN, *labels], next(iter(_slugs(metadata.get("categories"))), None))
        return source


class GSO(FuelObjects):
    meta = IMPORTERS.meta("object", "gso")
    OWNER = "GoogleResearch"
    ORIGIN = "origin::gso"
    LICENSE = "CC-BY-4.0"
    AUTHORS = "Google Research"
    CITATION = "Downs et al., Google Scanned Objects: A High-Quality Dataset of 3D Scanned Household Items, ICRA 2022"


stage = GSO.stage


def shown(archive: zipfile.ZipFile, item: str) -> Visual:
    """The single mesh visual of a Fuel model, refused when its SDF shows anything else."""
    names = archive.namelist()
    if SDF not in names:
        raise ValueError(f"{item} has no {SDF}, the object builder needs the mesh it names")
    found: set[Visual] = set()
    albedos: set[str] = set()
    for link in ET.fromstring(archive.read(SDF)).iter("link"):
        for visual in link.findall("visual"):
            mesh = visual.find("geometry/mesh")
            if mesh is None:
                raise ValueError(f"{item} has a visual that is not a mesh, the object builder takes one mesh file")
            found.add(_visual(link, visual, mesh))
            albedos.add(_member(visual.findtext("material/pbr/metal/albedo_map") or ""))
    if len(found) != 1:
        raise ValueError(f"{item} shows {len(found)} placed meshes, the object builder takes one mesh file")
    visual = next(iter(found))
    if visual.mesh not in names:
        raise ValueError(f"{item} shows the mesh {visual.mesh}, which its archive does not hold")
    if not visual.mesh.lower().endswith(MESHES):
        raise ValueError(f"{item} shows the mesh {visual.mesh}, staged are {' and '.join(MESHES)} meshes")
    albedo = next(iter(albedos))
    return attrs.evolve(visual, albedo=albedo) if len(albedos) == 1 and albedo in names else visual


def measured(source: Path, item: str) -> None:
    """Refuse a staged model the builder's converter loads empty or rescales."""
    mesh = next(path for path in sorted(source.rglob("*")) if path.suffix in MESHES)
    worker = BlenderWorker()
    try:
        values = worker.run([("load", str(mesh)), ("bounding_box",), ("rectify",), ("bounding_box",)]).values
    finally:
        worker.close()
    loaded = [high - low for low, high in values[1]]
    built = [high - low for low, high in values[3]]
    if not any(loaded):
        raise ValueError(f"{item}: the builder's converter loads no geometry from {mesh.name}")
    if any(abs(before - after) > 1e-6 * max(loaded) for before, after in zip(loaded, built, strict=True)):
        raise ValueError(f"{item} measures {loaded[0]:.2f} m by {loaded[1]:.2f} m, the object builder rescales a footprint of that size to {built[0]:.2f} m by {built[1]:.2f} m")


def scrub(text: str) -> str:
    """Text without the email addresses in it."""
    return _EMAIL.sub("", text).strip()


def _member(uri: str) -> str:
    """Archive member a model-relative or model:// URI names."""
    uri = uri.strip()
    return posixpath.normpath(uri[len(MODEL_URI) :].partition("/")[2] if uri.startswith(MODEL_URI) else uri)


def _visual(link: ET.Element, visual: ET.Element, mesh: ET.Element) -> Visual:
    outer_rotation, outer_translation = _pose(link)
    inner_rotation, inner_translation = _pose(visual)
    x, y, z = _apply(outer_rotation, inner_translation)
    translation = (x + outer_translation[0], y + outer_translation[1], z + outer_translation[2])
    scale = tuple(float(value) for value in (mesh.findtext("scale") or "1 1 1").split())
    return Visual(_member(mesh.findtext("uri") or ""), _product(outer_rotation, inner_rotation), typing.cast(Vector, scale), translation)


def _pose(element: ET.Element) -> tuple[Rows, Vector]:
    x, y, z, roll, pitch, yaw = (float(value) for value in (element.findtext("pose") or "0 0 0 0 0 0").split())
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    rotation = ((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr), (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr), (-sp, cp * sr, cp * cr))
    return rotation, (x, y, z)


def _apply(rows: Rows, vector: Vector) -> Vector:
    return typing.cast(Vector, tuple(row[0] * vector[0] + row[1] * vector[1] + row[2] * vector[2] for row in rows))


def _product(left: Rows, right: Rows) -> Rows:
    columns = tuple(_apply(left, (right[0][n], right[1][n], right[2][n])) for n in range(3))
    return typing.cast(Rows, tuple(tuple(column[n] for column in columns) for n in range(3)))


def _texture(name: str) -> bool:
    return not name.startswith(THUMBNAILS) and name.lower().endswith(TEXTURES)


def _material_libraries(archive: zipfile.ZipFile, mesh: str) -> list[str]:
    if not mesh.lower().endswith(".obj"):
        return []
    libraries = _MATERIAL_LIBRARY.findall(archive.read(mesh).decode(errors="replace"))
    return [posixpath.normpath(posixpath.join(posixpath.dirname(mesh), library)) for library in libraries]


def _description(archive: zipfile.ZipFile, item: str) -> str:
    if CONFIG in archive.namelist():
        description = scrub(ET.fromstring(archive.read(CONFIG)).findtext("description") or "")
        if description:
            return description
    return item.replace("_", " ")


def _metadata(path: Path) -> dict:
    if not path.is_file():
        return {}
    with open(path) as f:
        return json.load(f)


def _slugs(values: list | None) -> list[str]:
    return [slug for slug in (re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_") for value in values or []) if slug]


def annotate(path: Path, item: str, description: str, tags: list[str], kind: str | None) -> None:
    """Write an object annotation with the semantic fixes applied, the kind tag set when the fixes leave none."""
    spec = load_spec()
    _dump(path, {"name": item, "desc": description, "tags": tags})
    fixes.fix_file(path, spec)
    with open(path) as f:
        raw = yaml.safe_load(f)
    if kind is not None and not any(tag.startswith("kind::") for tag in raw["tags"]):
        raw["tags"] = [f"kind::{tag}" if tag == kind else tag for tag in raw["tags"]]
        _dump(path, raw)
        fixes.fix_file(path, spec)


def _dump(path: Path, raw: dict) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(raw, f, sort_keys=True)


def _place_obj(mesh: Path, visual: Visual) -> None:
    """Rewrite a Z-up OBJ in the model frame as the Y-up OBJ the builder's importer expects: (x, y, z) to (x, z, -y)."""
    lines = mesh.read_text().splitlines(keepends=True)
    for n, line in enumerate(lines):
        if line.startswith(("v ", "vn ")):
            key, x, y, z, *rest = line.split()
            placed = (visual.point if key == "v" else visual.normal)((float(x), float(y), float(z)))
            lines[n] = " ".join([key, repr(placed[0]), repr(placed[2]), repr(0.0 - placed[1]), *rest]) + "\n"
    mesh.write_text("".join(lines))


def _place_dae(mesh: Path, visual: Visual, albedo: Path | None) -> None:
    """Rewrite a COLLADA file so the builder's importer shows what Gazebo shows of it."""
    tree = ET.parse(mesh)
    root = tree.getroot()
    namespace = root.tag[1:].partition("}")[0] if root.tag.startswith("{") else ""
    prefix = f"{{{namespace}}}" if namespace else ""
    if namespace:
        ET.register_namespace("", namespace)
    _z_up(root, prefix)
    _name_ids(root, prefix)
    if albedo is not None:
        _paint(root, prefix, Path(os.path.relpath(albedo, mesh.parent)).as_posix())
    _sample_textures(root, prefix)
    _show(root, prefix)
    if visual.placed:
        _hang_scene(root, prefix, visual, f"{mesh.stem}_placement")
    tree.write(mesh, encoding="utf-8", xml_declaration=True)


def _z_up(root: ET.Element, prefix: str) -> None:
    """Declare the axes Gazebo reads the file in: it ignores up_axis and takes every COLLADA file as Z up."""
    asset = root.find(f"{prefix}asset")
    if asset is None:
        asset = ET.Element(f"{prefix}asset")
        root.insert(0, asset)
    axis = asset.find(f"{prefix}up_axis")
    if axis is None:
        axis = ET.SubElement(asset, f"{prefix}up_axis")
    axis.text = "Z_UP"


def _name_ids(root: ET.Element, prefix: str) -> None:
    """Give every element whose id is no XML name one that Blender resolves references through."""
    renamed: dict[str | None, str] = {}
    for n, element in enumerate(root.iter()):
        if "id" in element.attrib and not _XML_NAME.fullmatch(element.attrib["id"]):
            renamed[element.attrib["id"]] = f"id{n}"
            element.set("id", f"id{n}")
    for element in root.iter():
        for key, value in list(element.attrib.items()):
            if key != "id" and value.startswith("#") and value[1:] in renamed:
                element.set(key, f"#{renamed[value[1:]]}")
    for texture in root.iter(f"{prefix}texture"):
        if texture.get("texture") in renamed:
            texture.set("texture", renamed[texture.get("texture")])
    for reference in root.iterfind(f".//{prefix}surface/{prefix}init_from"):
        if reference.text in renamed:
            reference.text = renamed[reference.text]


def _paint(root: ET.Element, prefix: str, albedo: str) -> None:
    """Make the albedo map the SDF assigns the diffuse texture of every material that has none."""
    plain = [diffuse for diffuse in root.iterfind(f".//{prefix}profile_COMMON/{prefix}technique/*/{prefix}diffuse") if diffuse.find(f"{prefix}texture") is None]
    if not plain:
        return
    images = root.find(f"{prefix}library_images")
    if images is None:
        images = ET.Element(f"{prefix}library_images")
        root.insert(1, images)
    ET.SubElement(ET.SubElement(images, f"{prefix}image", {"id": "albedo", "name": "albedo"}), f"{prefix}init_from").text = albedo
    for diffuse in plain:
        diffuse[:] = [ET.Element(f"{prefix}texture", {"texture": "albedo", "texcoord": "CHANNEL0"})]


def _sample_textures(root: ET.Element, prefix: str) -> None:
    """Route every texture that names an image directly through a surface and a sampler parameter, which Blender requires."""
    images = {image.get("id") for image in root.iter(f"{prefix}image")}
    for profile in root.iter(f"{prefix}profile_COMMON"):
        technique = profile.find(f"{prefix}technique")
        if technique is None:
            continue
        parameters = {parameter.get("sid") for parameter in profile.iter(f"{prefix}newparam")}
        for texture in technique.iter(f"{prefix}texture"):
            image = texture.get("texture")
            if image is None or image in parameters or image not in images:
                continue
            surface, sampler = f"{image}-surface", f"{image}-sampler"
            if sampler not in parameters:
                at = list(profile).index(technique)
                profile.insert(at, _parameter(prefix, surface, "surface", "init_from", image, {"type": "2D"}))
                profile.insert(at + 1, _parameter(prefix, sampler, "sampler2D", "source", surface, {}))
                parameters.add(sampler)
            texture.set("texture", sampler)


def _parameter(prefix: str, sid: str, kind: str, field: str, value: str, attributes: dict[str, str]) -> ET.Element:
    parameter = ET.Element(f"{prefix}newparam", {"sid": sid})
    ET.SubElement(ET.SubElement(parameter, f"{prefix}{kind}", attributes), f"{prefix}{field}").text = value
    return parameter


def _show(root: ET.Element, prefix: str) -> None:
    """Restate each material's transparency as the opacity Gazebo derives from it, in the one form Blender reads the same way."""
    for shader in root.iterfind(f".//{prefix}profile_COMMON/{prefix}technique/*"):
        transparent, transparency = shader.find(f"{prefix}transparent"), shader.find(f"{prefix}transparency")
        present = [element for element in (transparent, transparency) if element is not None]
        if shader.tag[len(prefix) :] not in SHADERS or not present or (transparent is not None and transparent.find(f"{prefix}texture") is not None):
            continue
        hidden = _transparency(prefix, transparent, transparency)
        at = list(shader).index(present[0])
        for element in present:
            shader.remove(element)
        if hidden > 0.0:
            opacity = ET.Element(f"{prefix}transparency")
            ET.SubElement(opacity, f"{prefix}float").text = repr(1.0 - hidden)
            shader.insert(at, opacity)


def _transparency(prefix: str, transparent: ET.Element | None, transparency: ET.Element | None) -> float:
    """Transparency gz-common's ColladaLoader gives a material: none without an opaque mode on a transparent color."""
    if transparent is None or transparent.get("opaque") is None:
        return 0.0
    color = [float(value) for value in (transparent.findtext(f"{prefix}color") or "").split()]
    if len(color) < 3:
        return 0.0
    factor = float(transparency.findtext(f"{prefix}float") or 1.0) if transparency is not None else 1.0
    luminance = sum(weight * value for weight, value in zip(LUMINANCE, color, strict=False)) * factor
    alpha = (color[3] if len(color) > 3 else 1.0) * factor
    modes = {"RGB_ZERO": luminance, "RGB_ONE": 1.0 - luminance, "A_ONE": 1.0 - alpha, "A_ZERO": alpha}
    return min(1.0, max(0.0, modes.get(transparent.get("opaque") or "", 0.0)))


def _hang_scene(root: ET.Element, prefix: str, visual: Visual, name: str) -> None:
    """Move the nodes of every scene under one node that carries the placement in the file's own unit."""
    unit = root.find(f"{prefix}asset/{prefix}unit")
    meter = float(unit.get("meter", "1")) if unit is not None else 1.0
    rows = [(*row, offset / meter) for row, offset in zip(visual.linear, visual.translation, strict=True)]
    for scene in root.iter(f"{prefix}visual_scene"):
        nodes = scene.findall(f"{prefix}node")
        if not nodes:
            continue
        placement = ET.Element(f"{prefix}node", {"id": name, "name": name})
        ET.SubElement(placement, f"{prefix}matrix").text = " ".join(repr(value) for row in (*rows, (0.0, 0.0, 0.0, 1.0)) for value in row)
        scene.insert(list(scene).index(nodes[0]), placement)
        for node in nodes:
            scene.remove(node)
            placement.append(node)


def _relink_textures(mtl: Path, root: Path) -> None:
    lines = mtl.read_text().splitlines(keepends=True)
    changed = False
    for n, line in enumerate(lines):
        match = _TEXTURE_MAP.match(line)
        if match is None or (mtl.parent / match.group(2)).is_file():
            continue
        found = next(iter(sorted(root.rglob(Path(match.group(2)).name))), None)
        if found is None:
            continue
        lines[n] = f"{match.group(1)}{os.path.relpath(found, mtl.parent)}{match.group(3)}"
        changed = True
    if changed:
        mtl.write_text("".join(lines))
