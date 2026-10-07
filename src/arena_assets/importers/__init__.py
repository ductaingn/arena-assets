"""Importer registry and the adapter contract every dataset importer implements."""

from __future__ import annotations

import abc
import enum
import importlib
import os
import typing
from collections.abc import Iterable, Iterator
from pathlib import Path

import attrs
import yaml

from arena_assets.importers.providers import Provider

ATTRIBUTION_FILE = "ATTRIBUTION.yaml"


class License(enum.Enum):
    YES = "yes"
    NC = "nc"
    SA = "sa"
    NO = "no"
    UNCLEAR = "unclear"


class OnDemand(enum.Enum):
    ITEM = "item"
    CHUNK = "chunk"
    BULK = "bulk"


@attrs.frozen
class ImporterMeta:
    name: str
    kind: str
    package: str
    target: str
    license: License
    host: str
    on_demand: OnDemand
    description: str


@attrs.frozen
class Attribution:
    source: str
    url: str
    license: str
    authors: str
    citation: str = ""


def write_attribution(path: Path, attribution: Attribution) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(attrs.asdict(attribution), f, sort_keys=False)


def read_attribution(path: Path) -> Attribution:
    with open(path) as f:
        return Attribution(**yaml.safe_load(f))


class Importer(abc.ABC):
    meta: typing.ClassVar[ImporterMeta]

    def __init__(self, cache: Path | None = None, provider: Provider | None = None):
        self.cache = cache if cache is not None else default_cache(self.meta.name)
        self.provider = provider if provider is not None else self.default_provider()

    @classmethod
    def default_provider(cls) -> Provider:
        raise NotImplementedError(f"{cls.__name__} has no default provider, pass one")

    @classmethod
    def default_out(cls) -> Path:
        """Root the native resolvers of this kind read local items from."""
        return import_root()

    @abc.abstractmethod
    def items(self) -> Iterable[str]: ...

    @abc.abstractmethod
    def fetch(self, item: str) -> Path: ...

    @abc.abstractmethod
    def convert(self, item: str, src: Path, out: Path) -> str: ...

    @abc.abstractmethod
    def attribution(self, item: str) -> Attribution: ...

    def import_item(self, item: str, out: Path) -> Path:
        src = self.fetch(item)
        relpath = self.convert(item, src, out)
        target = out / relpath
        write_attribution(target / ATTRIBUTION_FILE, self.attribution(item))
        return target


def import_root() -> Path:
    """Imported assets root the resolvers read: ARENA_ASSETS_DIR_IMPORTED, else ARENA_ASSETS_DIR/_imported."""
    imported = os.environ.get("ARENA_ASSETS_DIR_IMPORTED")
    if imported:
        return Path(imported)
    root = os.environ.get("ARENA_ASSETS_DIR")
    if not root:
        raise RuntimeError("neither ARENA_ASSETS_DIR_IMPORTED nor ARENA_ASSETS_DIR is set, pass an explicit target dir")
    return Path(root) / "_imported"


def default_cache(name: str) -> Path:
    root = os.environ.get("ARENA_ASSETS_DIR")
    if not root:
        raise RuntimeError("ARENA_ASSETS_DIR is not set, pass an explicit cache dir")
    return Path(root) / ".import" / name


class ImporterRegistry:
    """Importers keyed by the kind they produce and their source, so one source can serve several kinds."""

    def __init__(self):
        self._metas: dict[tuple[str, str], ImporterMeta] = {}
        self._classes: dict[tuple[str, str], type[Importer]] = {}

    def register(self, meta: ImporterMeta) -> None:
        if (meta.kind, meta.name) in self._metas:
            raise ValueError(f"{meta.kind} importer '{meta.name}' is already registered")
        self._metas[(meta.kind, meta.name)] = meta

    def __iter__(self) -> Iterator[ImporterMeta]:
        return iter(self._metas[key] for key in sorted(self._metas))

    def __len__(self) -> int:
        return len(self._metas)

    def __contains__(self, key: object) -> bool:
        return key in self._metas

    def kinds(self) -> list[str]:
        return sorted({kind for kind, _ in self._metas})

    def names(self, kind: str) -> list[str]:
        names = sorted(name for registered, name in self._metas if registered == kind)
        if not names:
            raise KeyError(f"no importer kind '{kind}', kinds: {', '.join(self.kinds())}")
        return names

    def kinds_of(self, name: str) -> list[str]:
        """Kinds a source is registered under, empty for an unknown source."""
        return sorted(kind for kind, registered in self._metas if registered == name)

    def meta(self, kind: str, name: str) -> ImporterMeta:
        try:
            return self._metas[(kind, name)]
        except KeyError:
            raise KeyError(f"no {kind} importer '{name}', registered: {', '.join(self.names(kind))}") from None

    def load(self, kind: str, name: str) -> type[Importer]:
        key = (kind, name)
        if key not in self._classes:
            meta = self.meta(kind, name)
            module_name, _, attr = meta.target.partition(":")
            cls = getattr(importlib.import_module(module_name), attr)
            if not (isinstance(cls, type) and issubclass(cls, Importer)):
                raise TypeError(f"{meta.target} is not an Importer subclass")
            self._classes[key] = cls
        return self._classes[key]


IMPORTERS = ImporterRegistry()

for _meta in (
    ImporterMeta("gso", "object", "arena_assets", "arena_assets.importers.gso:GSO", License.YES, "fuel", OnDemand.ITEM, "Google Scanned Objects, 1k household objects from Gazebo Fuel"),
    ImporterMeta("ai2thor", "object", "arena_assets", "arena_assets.importers.ai2thor:AI2THOR", License.YES, "hfhub", OnDemand.ITEM, "AI2-THOR objects, the ProcTHOR, iTHOR and RoboTHOR asset set, ranged reads from the ManiSkill copy of the HSSD glTF conversion"),
    ImporterMeta("procthor", "world", "arena_simulation_setup", "arena_simulation_setup.adapters.procthor:ProcTHOR", License.YES, "github", OnDemand.CHUNK, "ProcTHOR-10K houses as worlds of typed rooms, walls, doors and AI2-THOR furniture, split files from GitHub LFS"),
    ImporterMeta("barn", "world", "arena_simulation_setup", "arena_simulation_setup.adapters.barn:BARNWorld", License.YES, "github", OnDemand.ITEM, "BARN challenge worlds 0 to 299 with the challenge start and goal as scenario, world files from Daffan/the-barn-challenge"),
    ImporterMeta("barn", "suite", "arena_evaluation", "arena_evaluation.adapters.barn:BARN", License.YES, "github", OnDemand.ITEM, "BARN challenge suite with its 300 worlds bundled plus a contest and a report manifest, world files and reference paths from Daffan/the-barn-challenge"),
    ImporterMeta(
        "eth_ucy", "scenario", "task_generator", "task_generator.adapters.eth_ucy:ETHUCY", License.UNCLEAR, "github", OnDemand.ITEM, "ETH and UCY pedestrian sequences (eth, hotel, univ, zara01, zara02) as replay scenarios, each bringing a world of the area its pedestrians walked, SocNavBench's interpolated copies"
    ),
    ImporterMeta("fsd50k", "sound", "arena_simulation_setup", "arena_simulation_setup.adapters.fsd50k:FSD50K", License.YES, "hfhub", OnDemand.ITEM, "FSD50K CC0 and CC BY 3.0 clips as sound items"),
    ImporterMeta("musohu", "episode", "arena_evaluation", "arena_evaluation.adapters.musohu:MuSoHu", License.YES, "http", OnDemand.ITEM, "MuSoHu rosbag1 human navigation runs as evaluation recorder episodes"),
):
    IMPORTERS.register(_meta)
del _meta
