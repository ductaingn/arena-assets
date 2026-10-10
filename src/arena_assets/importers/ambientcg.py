"""ambientCG CC0 materials built into Common/Material items."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterator
from pathlib import Path

from arena_assets.importers import IMPORTERS
from arena_assets.importers.material import Maps, MaterialImporter, slugs
from arena_assets.importers.providers import AmbientCG as AmbientCGHost

ATTRIBUTE = "2K-JPG"
SUFFIXES = {"color": "Color", "normal": "NormalGL", "roughness": "Roughness", "metalness": "Metalness", "occlusion": "AmbientOcclusion"}
AUTHORS = "ambientCG"


def members(archive: zipfile.ZipFile) -> dict[str, str]:
    """Archive member of each map slot an ambientCG zip fills."""
    names = archive.namelist()
    found = {slot: next((name for name in names if Path(name).stem.endswith(f"_{suffix}")), None) for slot, suffix in SUFFIXES.items()}
    return {slot: name for slot, name in found.items() if name is not None}


class AmbientCG(MaterialImporter):
    meta = IMPORTERS.meta("material", "ambientcg")
    ORIGIN = "origin::ambientcg"
    LICENSE = "CC0-1.0"

    provider: AmbientCGHost

    @classmethod
    def default_provider(cls) -> AmbientCGHost:
        return AmbientCGHost()

    def items(self) -> Iterator[str]:
        return iter(self.provider.list())

    def record(self, item: str) -> dict:
        """API record of an item, fetched once."""
        path = self.cache / f"{item}.json"
        if not path.is_file():
            record = self.provider.asset(item)
            if record is None or record["dataType"] != "Material":
                raise KeyError(f"no material '{item}' on ambientCG, list the items with: import material {self.meta.name}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record))
        with open(path) as f:
            return json.load(f)

    def download(self, item: str) -> str:
        """File name of the item's ATTRIBUTE zip."""
        downloads = self.record(item)["downloadFolders"]["default"]["downloadFiletypeCategories"]["zip"]["downloads"]
        names = {download["attribute"]: download["fileName"] for download in downloads}
        if ATTRIBUTE not in names:
            raise ValueError(f"{item} has no {ATTRIBUTE} download, it comes as {', '.join(sorted(names))}")
        return names[ATTRIBUTE]

    def fetch(self, item: str) -> Path:
        return self.provider.get(self.download(item), self.cache / f"{item}.zip")

    def maps(self, item: str, src: Path) -> Maps:
        extracted = self.cache / "maps" / item
        with zipfile.ZipFile(src) as archive:
            found = members(archive)
            if "color" not in found:
                raise ValueError(f"{item} has no {SUFFIXES['color']} map, its archive holds {', '.join(sorted(archive.namelist()))}")
            return Maps(**{slot: Path(archive.extract(name, extracted)) for slot, name in found.items()})

    def asset(self, item: str) -> str:
        return self.record(item)["assetId"]

    def description(self, item: str) -> str:
        record = self.record(item)
        return record["description"] or record["displayName"]

    def labels(self, item: str) -> list[str]:
        record = self.record(item)
        named = record["assetId"].lower()
        return [tag for tag in dict.fromkeys(slugs([record["displayCategory"], *record["tags"]])) if not tag.isdigit() and tag != named]

    def authors(self, item: str) -> str:
        return AUTHORS

    def page(self, item: str) -> str:
        return self.record(item)["shortLink"]
