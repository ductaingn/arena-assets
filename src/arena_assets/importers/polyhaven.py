"""Poly Haven CC0 textures built into Common/Material items."""

from __future__ import annotations

import json
import urllib.error
from collections.abc import Iterator
from pathlib import Path

from arena_assets.importers import IMPORTERS
from arena_assets.importers.material import Maps, MaterialImporter, slugs
from arena_assets.importers.providers import PolyHaven as PolyHavenHost

RESOLUTION = "2k"
FORMAT = "jpg"
COLOR = "Diffuse"
PACKED = {"normal": "nor_gl", "orm": "arm"}
SEPARATE = {"normal": "nor_gl", "roughness": "Rough", "metalness": "Metal", "occlusion": "AO"}
PAGE = "https://polyhaven.com/a"


def wanted(item: str, files: dict) -> dict[str, dict]:
    """Download entry of each map slot a Poly Haven file listing fills, the packed arm map in place of its three parts."""
    if COLOR not in files:
        raise ValueError(f"{item} has no {COLOR} map, it has {', '.join(sorted(files))}")
    keys = {"color": COLOR, **(PACKED if PACKED["orm"] in files else SEPARATE)}
    entries: dict[str, dict] = {}
    for slot, key in keys.items():
        if key not in files:
            continue
        sizes = files[key]
        if RESOLUTION not in sizes or FORMAT not in sizes[RESOLUTION]:
            raise ValueError(f"{item} has no {RESOLUTION} {FORMAT} {key} map, it comes in {', '.join(sorted(sizes))}")
        entries[slot] = sizes[RESOLUTION][FORMAT]
    return entries


class PolyHaven(MaterialImporter):
    meta = IMPORTERS.meta("material", "polyhaven")
    ORIGIN = "origin::polyhaven"
    LICENSE = "CC0-1.0"

    provider: PolyHavenHost

    @classmethod
    def default_provider(cls) -> PolyHavenHost:
        return PolyHavenHost()

    def items(self) -> Iterator[str]:
        return iter(self.provider.list())

    def record(self, item: str, name: str) -> dict:
        """API record *name* (info or files) of an item, fetched once."""
        try:
            path = self.provider.get(f"{name}/{item}", self.cache / item / f"{name}.json")
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
            raise KeyError(f"no texture '{item}' on Poly Haven, list the items with: import material {self.meta.name}") from None
        with open(path) as f:
            return json.load(f)

    def fetch(self, item: str) -> Path:
        src = self.cache / item
        self.record(item, "info")
        for slot, entry in wanted(item, self.record(item, "files")).items():
            self.provider.get(entry["url"], src / f"{slot}.{FORMAT}")
        return src

    def maps(self, item: str, src: Path) -> Maps:
        return Maps(**{slot: src / f"{slot}.{FORMAT}" for slot in wanted(item, self.record(item, "files"))})

    def description(self, item: str) -> str:
        info = self.record(item, "info")
        return info.get("description") or info["name"]

    def labels(self, item: str) -> list[str]:
        return slugs(self.record(item, "info").get("categories"))

    def authors(self, item: str) -> str:
        return ", ".join(self.record(item, "info")["authors"])

    def page(self, item: str) -> str:
        return f"{PAGE}/{item}"
