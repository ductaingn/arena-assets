"""Open Robotics furniture, appliance, kitchen, hospital and warehouse models from Gazebo Fuel, built into Common/Object items."""

from __future__ import annotations

import json
import re
import urllib.error
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterator
from pathlib import Path

from arena_assets.importers import IMPORTERS
from arena_assets.importers.gso import CONFIG, FuelObjects, scrub
from arena_assets.importers.providers import Fuel

CATEGORIES = frozenset({"Furniture and Home", "Appliance", "Kitchen"})
TAGS = frozenset({"hospital", "warehouse"})
OTHER_CATEGORIES = frozenset({"Robots", "Architecture", "Places and Landscapes"})
OTHER_TAGS = frozenset({"subt"})
LICENSES = {"Creative Commons Attribution 4.0 International": "CC-BY-4.0", "Creative Commons Zero v1.0 Universal": "CC0-1.0"}
PUBLISHER = "Open Robotics"
PLACEHOLDER_AUTHOR = "name"


def asset_name(model: str) -> str:
    """Item id and object name of a Fuel model name: every run of other characters becomes one underscore."""
    return re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_")


def wanted(model: dict) -> bool:
    """Whether a Fuel model entry is a CC0 or CC BY 4.0 furniture, appliance, kitchen, hospital or warehouse object."""
    categories = set(model.get("categories") or [])
    tags = {str(tag).lower() for tag in model.get("tags") or []}
    if categories & OTHER_CATEGORIES or tags & OTHER_TAGS:
        return False
    return model["license_name"] in LICENSES and bool(categories & CATEGORIES or tags & TAGS)


def credited(archive: zipfile.ZipFile) -> list[str]:
    """Author names a Fuel model's model.config gives, without the email addresses beside them."""
    if CONFIG not in archive.namelist():
        return []
    names = (scrub(author.findtext("name") or "") for author in ET.fromstring(archive.read(CONFIG)).iter("author"))
    return [name for name in names if name and name.lower() != PLACEHOLDER_AUTHOR]


class OpenRobotics(FuelObjects):
    meta = IMPORTERS.meta("object", "openrobotics")
    OWNER = "OpenRobotics"
    ORIGIN = "origin::openrobotics"
    LABELS = ("categories", "tags")

    def __init__(self, cache: Path | None = None, provider: Fuel | None = None):
        super().__init__(cache, provider)
        self._names: dict[str, str] | None = None

    def items(self) -> Iterator[str]:
        return (asset_name(model["name"]) for model in self.provider.models() if wanted(model))

    def listed(self, item: str) -> str:
        """Fuel model name behind an item id, from the owner's model listing."""
        if self._names is None:
            self._names = {asset_name(model["name"]): model["name"] for model in self.provider.models()}
        try:
            return self._names[item]
        except KeyError:
            raise KeyError(f"no model '{item}' of {self.OWNER} on Fuel, list the items with: import object {self.meta.name}") from None

    def metadata(self, item: str) -> dict:
        """Fuel record of an item, looked up by Fuel model name first and by item id second."""
        path = self.cache / f"{item}.json"
        if not path.is_file():
            try:
                self.provider.get(item, path)
            except urllib.error.HTTPError as error:
                if error.code != 404:
                    raise
                self.provider.get(self.listed(item), path)
        with open(path) as f:
            return json.load(f)

    def model(self, item: str) -> str:
        return self.metadata(item)["name"]

    def asset(self, item: str) -> str:
        return asset_name(self.model(item))

    def license(self, item: str) -> str:
        metadata = self.metadata(item)
        if metadata["license_name"] not in LICENSES:
            raise ValueError(f"{metadata['name']} is under {metadata['license_name']}, only CC0 and CC BY 4.0 models are imported")
        return LICENSES[metadata["license_name"]]

    def authors(self, item: str) -> str:
        with zipfile.ZipFile(self.fetch(item)) as archive:
            return ", ".join(dict.fromkeys([*credited(archive), PUBLISHER]))

    def fetch(self, item: str) -> Path:
        self.license(item)
        return super().fetch(item)
