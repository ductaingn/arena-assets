import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from arena_assets.importers import (
    ATTRIBUTION_FILE,
    IMPORTERS,
    Attribution,
    Importer,
    ImporterMeta,
    ImporterRegistry,
    License,
    OnDemand,
    read_attribution,
    write_attribution,
)
from arena_assets.importers.providers import Http


def test_registry_lists_importers_by_kind_then_source():
    assert [(meta.kind, meta.name) for meta in IMPORTERS] == [
        ("episode", "musohu"),
        ("object", "ai2thor"),
        ("object", "gso"),
        ("object", "openrobotics"),
        ("scenario", "eth_ucy"),
        ("sound", "fsd50k"),
        ("suite", "barn"),
        ("world", "barn"),
        ("world", "procthor"),
    ]


def test_kinds_and_the_sources_of_a_kind():
    assert IMPORTERS.kinds() == ["episode", "object", "scenario", "sound", "suite", "world"]
    assert IMPORTERS.names("world") == ["barn", "procthor"]
    assert IMPORTERS.kinds_of("barn") == ["suite", "world"]
    assert IMPORTERS.kinds_of("nope") == []


def test_listing_imports_nothing():
    code = "import sys\nfrom arena_assets.importers import IMPORTERS\nlist(IMPORTERS)\nprint(' '.join(sorted(m for m in sys.modules if m.startswith('arena_assets.importers.') and m != 'arena_assets.importers.providers')))"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip() == ""


def test_meta_lookup_and_membership():
    assert ("object", "gso") in IMPORTERS
    assert ("world", "gso") not in IMPORTERS
    assert ("object", "nope") not in IMPORTERS
    meta = IMPORTERS.meta("object", "gso")
    assert meta.kind == "object"
    assert meta.license is License.YES
    assert meta.on_demand is OnDemand.ITEM


def test_meta_error_lists_the_sources_of_the_kind():
    with pytest.raises(KeyError, match="no object importer 'nope', registered: ai2thor, gso, openrobotics"):
        IMPORTERS.meta("object", "nope")


def test_unknown_kind_lists_the_kinds():
    with pytest.raises(KeyError, match="no importer kind 'nope', kinds: episode, object, scenario, sound, suite, world"):
        IMPORTERS.meta("nope", "gso")


def test_register_rejects_a_duplicate_of_kind_and_source():
    registry = ImporterRegistry()
    meta = ImporterMeta("x", "object", "pkg", "pkg.x:X", License.NO, "http", OnDemand.BULK, "")
    registry.register(meta)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(meta)
    assert list(registry) == [meta]


def test_one_source_registers_under_several_kinds():
    registry = ImporterRegistry()
    as_object = ImporterMeta("x", "object", "pkg", "pkg.x:X", License.NO, "http", OnDemand.BULK, "")
    as_world = ImporterMeta("x", "world", "pkg", "pkg.x:XWorld", License.NO, "http", OnDemand.BULK, "")
    registry.register(as_world)
    registry.register(as_object)
    assert list(registry) == [as_object, as_world]
    assert registry.meta("world", "x") is as_world


def test_load_rejects_non_importer():
    registry = ImporterRegistry()
    registry.register(ImporterMeta("bad", "object", "pkg", "pathlib:Path", License.NO, "http", OnDemand.BULK, ""))
    with pytest.raises(TypeError, match="not an Importer"):
        registry.load("object", "bad")


def test_attribution_roundtrip(tmp_path):
    attribution = Attribution(source="x", url="https://example.org/x", license="CC-BY-4.0", authors="A, B", citation="A et al. 2020")
    path = tmp_path / "item" / ATTRIBUTION_FILE
    write_attribution(path, attribution)
    with open(path) as f:
        assert list(yaml.safe_load(f)) == ["source", "url", "license", "authors", "citation"]
    assert read_attribution(path) == attribution


class Counting(Importer):
    meta = ImporterMeta("counting", "object", "arena_assets", "tests.test_importers:Counting", License.YES, "http", OnDemand.ITEM, "test double")

    def items(self):
        return ["a", "b"]

    def fetch(self, item: str) -> Path:
        src = self.cache / f"{item}.txt"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text(item)
        return src

    def convert(self, item: str, src: Path, out: Path) -> str:
        target = out / "object" / item
        target.mkdir(parents=True)
        (target / "model.txt").write_text(src.read_text().upper())
        return f"object/{item}"

    def attribution(self, item: str) -> Attribution:
        return Attribution(source="counting", url=f"https://example.org/{item}", license="CC0-1.0", authors="nobody")


def test_import_item_template(tmp_path):
    importer = Counting(cache=tmp_path / "cache", provider=Http("https://example.org"))
    target = importer.import_item("a", tmp_path / "out")
    assert target == tmp_path / "out" / "object" / "a"
    assert (target / "model.txt").read_text() == "A"
    assert read_attribution(target / ATTRIBUTION_FILE).url == "https://example.org/a"
    assert sorted(p.name for p in target.iterdir()) == [ATTRIBUTION_FILE, "model.txt"]
    assert (tmp_path / "cache" / "a.txt").is_file()


def test_default_cache_needs_env(without_assets_dir):
    with pytest.raises(RuntimeError, match="ARENA_ASSETS_DIR"):
        Counting(provider=Http("https://example.org"))


@pytest.fixture
def without_assets_dir():
    import os

    saved = os.environ.pop("ARENA_ASSETS_DIR", None)
    yield
    if saved is not None:
        os.environ["ARENA_ASSETS_DIR"] = saved
