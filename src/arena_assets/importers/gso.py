"""Google Scanned Objects from Gazebo Fuel, built into Common/Object items by the object database builder."""

from __future__ import annotations

import json
import os
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterator
from pathlib import Path

import yaml

from arena_assets.impl import ANNOTATION_NAME, AssetType
from arena_assets.impl.build import DatabaseBuilder, OverwriteMode
from arena_assets.importers import IMPORTERS, Attribution, Importer
from arena_assets.importers.providers import Fuel, Provider
from arena_assets.semantic import fixes
from arena_assets.semantic.taxonomy import load_spec

DOMAIN = "Common"
ASSET_TYPE = "Object"
VERSION = "tip"
FORMATS = "usdz,sdf,fbx"
ORIGIN = "origin::gso"
AUTHORS = "Google Research"
CITATION = "Downs et al., Google Scanned Objects: A High-Quality Dataset of 3D Scanned Household Items, ICRA 2022"
SOURCE_DIRS = ("meshes/", "materials/")

_TEXTURE_MAP = re.compile(r"^(\s*map_\w+\s+)(\S+)(\s*)$")


class GSO(Importer):
    meta = IMPORTERS.meta("object", "gso")

    @classmethod
    def default_provider(cls) -> Provider:
        return Fuel("GoogleResearch")

    def items(self) -> Iterator[str]:
        return iter(self.provider.list())

    def fetch(self, item: str) -> Path:
        self.provider.get(item, self.cache / f"{item}.json")
        return self.provider.get(f"{item}/{VERSION}/{item}.zip", self.cache / f"{item}.zip")

    def convert(self, item: str, src: Path, out: Path) -> str:
        relpath = Path(DOMAIN) / ASSET_TYPE / item
        target = out / relpath
        if target.exists():
            shutil.rmtree(target)
        dataset = self.cache / "stage" / item
        stage(item, src, dataset)
        builder = DatabaseBuilder.Builder(AssetType.OBJECT)(input_path=dataset / DOMAIN, output_path=out / DOMAIN, overwrite=OverwriteMode.OVERWRITE)
        builder.enable("formats", FORMATS)
        builder.enable("previews")
        try:
            builder.build()
        finally:
            shutil.rmtree(dataset)
        if not (target / ANNOTATION_NAME).is_file():
            raise RuntimeError(f"object build failed for {item}, the build log above names the step")
        return str(relpath)

    def attribution(self, item: str) -> Attribution:
        return Attribution(source=self.meta.name, url=self.provider.url(item), license="CC-BY-4.0", authors=AUTHORS, citation=CITATION)


def stage(item: str, src: Path, dataset: Path) -> Path:
    """Lay one Fuel zip out as object builder input under dataset/Common/Object/<item> and return that directory."""
    source = dataset / DOMAIN / ASSET_TYPE / item
    if dataset.exists():
        shutil.rmtree(dataset)
    source.mkdir(parents=True)
    with zipfile.ZipFile(src) as archive:
        archive.extractall(source, [name for name in archive.namelist() if name.startswith(SOURCE_DIRS)])
        description = _description(archive, item)
    for mtl in sorted(source.rglob("*.mtl")):
        _relink_textures(mtl, source)
    for mesh in sorted(source.rglob("*.obj")):
        _y_up(mesh)
    _annotate(source / ANNOTATION_NAME, item, description, _categories(src.with_suffix(".json")))
    return source


def _description(archive: zipfile.ZipFile, item: str) -> str:
    if "model.config" in archive.namelist():
        element = ET.fromstring(archive.read("model.config")).find("description")
        if element is not None and element.text and element.text.strip():
            return element.text.strip()
    return item.replace("_", " ")


def _categories(metadata: Path) -> list[str]:
    if not metadata.is_file():
        return []
    with open(metadata) as f:
        categories = json.load(f).get("categories") or []
    return [slug for slug in (re.sub(r"[^a-z0-9]+", "_", str(category).lower()).strip("_") for category in categories) if slug]


def _annotate(path: Path, item: str, description: str, categories: list[str]) -> None:
    spec = load_spec()
    _dump(path, {"name": item, "desc": description, "tags": [ORIGIN, *categories]})
    fixes.fix_file(path, spec)
    with open(path) as f:
        raw = yaml.safe_load(f)
    if categories and not any(tag.startswith("kind::") for tag in raw["tags"]):
        raw["tags"] = [f"kind::{tag}" if tag == categories[0] else tag for tag in raw["tags"]]
        _dump(path, raw)
        fixes.fix_file(path, spec)


def _dump(path: Path, raw: dict) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(raw, f, sort_keys=True)


def _y_up(mesh: Path) -> None:
    """Rewrite a Z-up OBJ as the Y-up OBJ the builder's importer expects: (x, y, z) to (x, z, -y)."""
    lines = mesh.read_text().splitlines(keepends=True)
    for n, line in enumerate(lines):
        if line.startswith(("v ", "vn ")):
            key, x, y, z, *rest = line.split()
            lines[n] = " ".join([key, x, z, _negated(y), *rest]) + "\n"
    mesh.write_text("".join(lines))


def _negated(number: str) -> str:
    return number[1:] if number.startswith("-") else number if float(number) == 0 else f"-{number}"


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
