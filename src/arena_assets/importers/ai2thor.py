"""AI2-THOR objects from the HSSD glTF conversion, built into Common/Object items by the object database builder."""

from __future__ import annotations

import io
import json
import re
import shutil
import tempfile
import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path

import yaml

from arena_assets.impl import ANNOTATION_NAME, AssetType
from arena_assets.impl.build import DatabaseBuilder, OverwriteMode
from arena_assets.importers import ATTRIBUTION_FILE, IMPORTERS, Attribution, Importer, write_attribution
from arena_assets.importers.providers import HFHub, Http, Provider
from arena_assets.semantic import fixes
from arena_assets.semantic.taxonomy import load_spec
from arena_assets.utils.logging import get_logger

REPO = "haosulab/AI2THOR"
REVISION = "1a173d5de042aaad8f1af09d4d2bc2ce4004b28a"
ARCHIVE = "ai2thor.zip"
MEMBERS = "ai2thor/ai2thorhab-uncompressed/assets/objects/"
TYPES_BASE = "https://raw.githubusercontent.com/allenai/procthor/53d5bd4c8c96a699e6a615dc390abb670cc9d353/procthor/databases"
TYPES = "asset-database.json"
DOMAIN = "Common"
ASSET_TYPE = "Object"
FORMATS = "usdz,sdf,fbx"
ORIGIN = "origin::ai2thor"
LICENSE = "CC-BY-4.0"
AUTHORS = "Allen Institute for AI (AI2-THOR assets), converted to glTF by the HSSD team (hssd/ai2thor-hab) and repackaged by the ManiSkill team (haosulab/AI2THOR)"
CITATION = "Kolve et al., AI2-THOR: An Interactive 3D Environment for Visual AI, arXiv:1712.05474, 2017"
READAHEAD = 1 << 16

_WORD_START = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

logger = get_logger("import.ai2thor")


class RemoteZip(io.RawIOBase):
    """Seekable view of a remote zip: the central directory from a local copy, member bytes by ranged reads."""

    def __init__(self, provider: Provider, relpath: str, scratch: Path, size: int, directory: tuple[int, bytes] | None = None):
        self.provider = provider
        self.relpath = relpath
        self.scratch = scratch
        self.size = size
        self.directory = directory
        self._pos = 0
        self._start = 0
        self._buffer = b""

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self._pos = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self.size}[whence] + offset
        return self._pos

    def read(self, n: int = -1) -> bytes:
        n = self.size - self._pos if n is None or n < 0 else min(n, self.size - self._pos)
        if self.directory is not None and self._pos >= self.directory[0]:
            offset = self._pos - self.directory[0]
            data = self.directory[1][offset : offset + n]
        else:
            if not (self._start <= self._pos and self._pos + n <= self._start + len(self._buffer)):
                end = min(self.size, self._pos + max(n, READAHEAD)) - 1
                self._start, self._buffer = self._pos, self.provider.get(self.relpath, self.scratch, range=(self._pos, end)).read_bytes()
            data = self._buffer[self._pos - self._start : self._pos - self._start + n]
        self._pos += len(data)
        return data


class AI2THOR(Importer):
    meta = IMPORTERS.meta("object", "ai2thor")

    def __init__(self, cache: Path | None = None, provider: Provider | None = None):
        super().__init__(cache, provider)
        self._zip: zipfile.ZipFile | None = None
        self._types: dict[str, str] | None = None

    @classmethod
    def default_provider(cls) -> Provider:
        return HFHub(REPO, revision=REVISION)

    def archive(self) -> zipfile.ZipFile:
        """The source archive, its directory fetched once into the cache."""
        if self._zip is None:
            index = self.cache / f"{ARCHIVE}.directory.json"
            scratch = self.cache / f"{ARCHIVE}.range"
            if index.is_file():
                meta = json.loads(index.read_text())
                directory = (meta["start"], (self.cache / f"{ARCHIVE}.directory").read_bytes())
                self._zip = zipfile.ZipFile(RemoteZip(self.provider, ARCHIVE, scratch, meta["size"], directory))
            else:
                size = self.provider.size(ARCHIVE)
                self._zip = zipfile.ZipFile(RemoteZip(self.provider, ARCHIVE, scratch, size))
                self.provider.get(ARCHIVE, self.cache / f"{ARCHIVE}.directory", range=(self._zip.start_dir, size - 1))
                index.write_text(json.dumps({"start": self._zip.start_dir, "size": size}))
        return self._zip

    def object_type(self, item: str) -> str:
        """AI2-THOR object type of an asset from the ProcTHOR asset database, empty when it lists none."""
        if self._types is None:
            path = Http(TYPES_BASE).get(TYPES, self.cache / TYPES)
            with open(path) as f:
                self._types = {asset["assetId"]: object_type for object_type, assets in json.load(f).items() for asset in assets}
        return self._types.get(item, "")

    def items(self) -> Iterator[str]:
        for name in self.archive().namelist():
            if name.startswith(MEMBERS) and name.endswith(".glb"):
                yield name[len(MEMBERS) : -len(".glb")]

    def fetch(self, item: str) -> Path:
        dest = self.cache / "objects" / f"{item}.glb"
        if dest.is_file():
            return dest
        try:
            data = self.archive().read(f"{MEMBERS}{item}.glb")
        except KeyError:
            raise KeyError(f"no AI2-THOR object '{item}' in {REPO}") from None
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        part.write_bytes(data)
        part.replace(dest)
        return dest

    def convert(self, item: str, src: Path, out: Path) -> str:
        built = self.build({item: src}, out)
        if item not in built:
            raise RuntimeError(f"object build failed for {item}, the build log above names the step")
        return built[item]

    def build(self, sources: dict[str, Path], out: Path) -> dict[str, str]:
        """Build every staged item in one builder run and return the relpaths of those that built."""
        self.cache.mkdir(parents=True, exist_ok=True)
        dataset = Path(tempfile.mkdtemp(prefix="stage-", dir=self.cache))
        for item, src in sources.items():
            target = out / DOMAIN / ASSET_TYPE / item
            if target.exists():
                shutil.rmtree(target)
            stage(item, src, dataset, self.object_type(item))
        builder = DatabaseBuilder.Builder(AssetType.OBJECT)(input_path=dataset / DOMAIN, output_path=out / DOMAIN, overwrite=OverwriteMode.OVERWRITE)
        builder.enable("formats", FORMATS)
        builder.enable("previews")
        try:
            builder.build()
        finally:
            shutil.rmtree(dataset)
        return {item: str(Path(DOMAIN) / ASSET_TYPE / item) for item in sources if (out / DOMAIN / ASSET_TYPE / item / ANNOTATION_NAME).is_file()}

    def import_items(self, items: Iterable[str], out: Path) -> dict[str, Path]:
        """Fetch, build and attribute several items with one builder run, skipping those the source does not hold."""
        sources: dict[str, Path] = {}
        for item in items:
            try:
                sources[item] = self.fetch(item)
            except KeyError as e:
                logger.warning("%s", e.args[0])
        built = self.build(sources, out) if sources else {}
        for item in sorted(set(sources) - set(built)):
            logger.warning("object build failed for %s", item)
        for item, relpath in built.items():
            write_attribution(out / relpath / ATTRIBUTION_FILE, self.attribution(item))
        return {item: out / relpath for item, relpath in built.items()}

    def attribution(self, item: str) -> Attribution:
        return Attribution(source=self.meta.name, url=f"{self.provider.url(ARCHIVE)}#{MEMBERS}{item}.glb", license=LICENSE, authors=AUTHORS, citation=CITATION)


def stage(item: str, src: Path, dataset: Path, object_type: str) -> Path:
    """Lay one glb out as object builder input under dataset/Common/Object/<item> and return that directory."""
    source = dataset / DOMAIN / ASSET_TYPE / item
    source.mkdir(parents=True)
    shutil.copyfile(src, source / f"{item}.glb")
    words = _WORD_START.sub(" ", object_type)
    _annotate(source / ANNOTATION_NAME, item, words or item.replace("_", " "), [words.lower().replace(" ", "_")] if words else [])
    return source


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
