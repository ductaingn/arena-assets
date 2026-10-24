"""PBR texture sets built into Common/Material items: an OmniPBR <name>.mdl beside the maps it binds."""

from __future__ import annotations

import abc
import re
import shutil
import typing
from pathlib import Path

import attrs
import yaml

from arena_assets.impl import ANNOTATION_NAME, AssetType
from arena_assets.impl.build import DatabaseBuilder, OverwriteMode, replacing
from arena_assets.importers import Attribution, Importer
from arena_assets.semantic import fixes
from arena_assets.semantic.taxonomy import load_spec

DOMAIN = "Common"
ASSET_TYPE = "Material"
ASSET_CLASS = "material"
SRGB = "::tex::gamma_srgb"
LINEAR = "::tex::gamma_linear"
SUFFIXES = {"color": "BaseColor", "roughness": "R", "metalness": "M", "orm": "ORM", "occlusion": "AO", "normal": "N"}

_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


@attrs.frozen
class Maps:
    """Texture set files: sRGB color, OpenGL normal, linear rest, orm as occlusion, roughness, metalness in RGB."""

    color: Path
    roughness: Path | None = None
    metalness: Path | None = None
    orm: Path | None = None
    occlusion: Path | None = None
    normal: Path | None = None


def slugs(values: list | None) -> list[str]:
    return [slug for slug in (re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_") for value in values or []) if slug]


def mdl(name: str, textures: dict[str, str]) -> str:
    """OmniPBR material *name* binding the texture path of each slot in *textures*, a roughness or metalness map at full influence."""

    def texture(slot: str, gamma: str) -> str:
        return f'texture_2d("{textures[slot]}", {gamma})'

    arguments = [f"diffuse_texture: {texture('color', SRGB)}"]
    if "roughness" in textures or "orm" in textures:
        arguments.append("reflection_roughness_texture_influence: 1.f")
    if "roughness" in textures:
        arguments.append(f"reflectionroughness_texture: {texture('roughness', LINEAR)}")
    if "metalness" in textures or "orm" in textures:
        arguments.append("metallic_texture_influence: 1.f")
    if "metalness" in textures:
        arguments.append(f"metallic_texture: {texture('metalness', LINEAR)}")
    if "orm" in textures:
        arguments += ["enable_ORM_texture: true", f"ORM_texture: {texture('orm', LINEAR)}"]
    if "occlusion" in textures:
        arguments.append(f"ao_texture: {texture('occlusion', LINEAR)}")
    if "normal" in textures:
        arguments += [f"normalmap_texture: {texture('normal', LINEAR)}", "flip_tangent_v: false"]
    body = ",\n".join(f"    {argument}" for argument in arguments)
    return f"mdl 1.4;\n\nimport ::OmniPBR::OmniPBR;\nimport ::tex::gamma_mode;\n\nexport material {name}(*)\n = ::OmniPBR::OmniPBR(\n{body});\n"


def annotate(path: Path, name: str, description: str, tags: list[str]) -> None:
    """Write the annotation of a staged material, the first tag naming a material kind of the taxonomy, singular or a simple plural, as its kind."""
    spec = load_spec()
    kinds = spec.facets["kind"].values_for(ASSET_CLASS)
    named = {tag: tag if tag in kinds else tag[:-1] for tag in tags if tag in kinds or (tag.endswith("s") and tag[:-1] in kinds)}
    first = next(iter(named), None)
    with open(path, "w") as f:
        yaml.safe_dump({"name": name, "desc": description, "tags": [f"kind::{named[tag]}" if tag == first else tag for tag in tags]}, f, sort_keys=True)
    fixes.fix_file(path, spec)


def stage(name: str, maps: Maps, description: str, tags: list[str], dataset: Path) -> Path:
    """Lay one texture set out as material builder input under dataset/Common/Material/<name> and return that directory."""
    if not _IDENTIFIER.fullmatch(name):
        raise ValueError(f"'{name}' is no MDL identifier, a material name is a letter followed by letters, digits and underscores")
    source = dataset / DOMAIN / ASSET_TYPE / name
    if dataset.exists():
        shutil.rmtree(dataset)
    (source / name).mkdir(parents=True)
    textures: dict[str, str] = {}
    for slot, src in attrs.asdict(maps, recurse=False).items():
        if src is None:
            continue
        relpath = f"{name}/{name}_{SUFFIXES[slot]}{src.suffix.lower()}"
        shutil.copyfile(src, source / relpath)
        textures[slot] = f"./{relpath}"
    (source / f"{name}.mdl").write_text(mdl(name, textures))
    annotate(source / ANNOTATION_NAME, name, description, tags)
    return source


class MaterialImporter(Importer):
    """Texture sets of one source, each staged as an OmniPBR material and built by the material database builder."""

    ORIGIN: typing.ClassVar[str]
    LICENSE: typing.ClassVar[str]

    @abc.abstractmethod
    def maps(self, item: str, src: Path) -> Maps:
        """Map files of a fetched item."""

    @abc.abstractmethod
    def authors(self, item: str) -> str: ...

    @abc.abstractmethod
    def page(self, item: str) -> str:
        """Web page of an item at its source."""

    def asset(self, item: str) -> str:
        """Name the built material gets."""
        return item

    def description(self, item: str) -> str:
        return ""

    def labels(self, item: str) -> list[str]:
        """Bare tags of an item."""
        return []

    def convert(self, item: str, src: Path, out: Path) -> str:
        name = self.asset(item)
        relpath = Path(DOMAIN) / ASSET_TYPE / name
        target = out / relpath
        dataset = self.cache / "stage" / name
        try:
            with replacing(target):
                stage(name, self.maps(item, src), self.description(item), [self.ORIGIN, *self.labels(item)], dataset)
                DatabaseBuilder.Builder(AssetType.MATERIAL)(input_path=dataset / DOMAIN, output_path=out / DOMAIN, overwrite=OverwriteMode.OVERWRITE).build()
                if not (target / ANNOTATION_NAME).is_file():
                    raise RuntimeError(f"material build failed for {name}, the build log above names the step")
        finally:
            shutil.rmtree(dataset, ignore_errors=True)
        return str(relpath)

    def attribution(self, item: str) -> Attribution:
        return Attribution(source=self.meta.name, url=self.page(item), license=self.LICENSE, authors=self.authors(item))
