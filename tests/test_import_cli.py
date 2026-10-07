import json
import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from arena_assets.__main__ import app
from arena_assets.importers import ATTRIBUTION_FILE, IMPORTERS

ITEM = "Jawbone_UP24_Wireless_Activity_Tracker_Pink_Coral_L"
FIXTURE = Path(__file__).parent / "fixtures" / "gso" / f"{ITEM}.zip"
RELPATH = Path("Common") / "Object" / ITEM

runner = CliRunner()


def seed(cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURE, cache / f"{ITEM}.zip")
    (cache / f"{ITEM}.json").write_text(json.dumps({"name": ITEM, "categories": ["Consumer Goods"]}))


def test_bare_import_lists_the_registry():
    result = runner.invoke(app, ["import"])
    assert result.exit_code == 0
    lines = result.output.strip().splitlines()
    assert lines[0].split() == ["kind", "source", "license", "host", "on_demand", "description"]
    assert [tuple(line.split()[:2]) for line in lines[1:]] == [(meta.kind, meta.name) for meta in IMPORTERS]
    gso = next(line for line in lines if line.split()[:2] == ["object", "gso"])
    assert gso.split()[2:5] == ["yes", "fuel", "item"]


def test_bare_import_json():
    result = runner.invoke(app, ["import", "--json"])
    assert result.exit_code == 0
    rows = json.loads(result.output)
    assert [(row["kind"], row["name"]) for row in rows] == [(meta.kind, meta.name) for meta in IMPORTERS]
    assert next(row for row in rows if row["name"] == "gso") == {"kind": "object", "name": "gso", "license": "yes", "host": "fuel", "on_demand": "item", "description": IMPORTERS.meta("object", "gso").description}


def test_kind_alone_lists_its_sources():
    result = runner.invoke(app, ["import", "world"])
    assert result.exit_code == 0
    lines = result.output.strip().splitlines()
    assert [tuple(line.split()[:2]) for line in lines if line.startswith("world ")] == [("world", "barn"), ("world", "houseexpo"), ("world", "procthor")]
    assert not any(line.startswith("object ") for line in lines)
    assert "3 source(s) of kind world, list one's items with: import world SOURCE" in lines


def test_unknown_kind_lists_the_kinds():
    result = runner.invoke(app, ["import", "nope", "x"])
    assert result.exit_code != 0
    assert "no importer kind 'nope'" in result.output
    assert all(kind in result.output for kind in IMPORTERS.kinds())


def test_a_source_in_place_of_the_kind_names_the_command():
    result = runner.invoke(app, ["import", "gso", ITEM])
    assert result.exit_code != 0
    assert "'gso' is a source of kind object: import object gso" in " ".join(result.output.replace("\u2502", " ").split())


def test_unknown_source_lists_the_sources_of_the_kind():
    result = runner.invoke(app, ["import", "object", "nope", "x", "--out", "/tmp/x", "--cache", "/tmp/x"])
    assert result.exit_code != 0
    assert "no object importer 'nope'" in result.output
    assert all(name in result.output for name in IMPORTERS.names("object"))


def test_a_source_under_the_wrong_kind_names_its_kinds():
    result = runner.invoke(app, ["import", "object", "barn", "0"])
    assert result.exit_code != 0
    assert "'barn' is a source of kind suite and world: import suite barn" in " ".join(result.output.replace("\u2502", " ").split())


def test_items_and_all_are_refused(tmp_path):
    result = runner.invoke(app, ["import", "object", "gso", ITEM, "--all", "--out", str(tmp_path), "--cache", str(tmp_path)])
    assert result.exit_code != 0
    assert "not both" in result.output


def test_out_default_needs_assets_dir(tmp_path):
    result = runner.invoke(app, ["import", "object", "gso", ITEM, "--cache", str(tmp_path)], env={"ARENA_ASSETS_DIR": None, "ARENA_ASSETS_DIR_IMPORTED": None})
    assert result.exit_code != 0
    assert "ARENA_ASSETS_DIR" in result.output
    assert "--out" in result.output


def test_import_from_seeded_cache_writes_item_and_attribution(blender, tmp_path):
    cache = tmp_path / "cache"
    seed(cache)
    out = tmp_path / "out"
    result = runner.invoke(app, ["-s", "import", "object", "gso", ITEM, "--out", str(out), "--cache", str(cache)])
    assert result.exit_code == 0, result.output
    target = out / RELPATH
    assert result.output.strip() == str(target)
    assert (target / ATTRIBUTION_FILE).is_file()
    assert (target / f"{ITEM}.usdz").is_file()


def test_out_defaults_to_the_imported_root_and_cache_to_the_assets_dir(blender, tmp_path):
    seed(tmp_path / ".import" / "gso")
    result = runner.invoke(app, ["-s", "import", "object", "gso", ITEM], env={"ARENA_ASSETS_DIR": str(tmp_path), "ARENA_ASSETS_DIR_IMPORTED": None})
    assert result.exit_code == 0, result.output
    assert result.output.strip() == str(tmp_path / "_imported" / RELPATH)


def test_out_default_follows_the_imported_assets_dir(blender, tmp_path):
    seed(tmp_path / ".import" / "gso")
    result = runner.invoke(app, ["-s", "import", "object", "gso", ITEM], env={"ARENA_ASSETS_DIR": str(tmp_path), "ARENA_ASSETS_DIR_IMPORTED": str(tmp_path / "mine")})
    assert result.exit_code == 0, result.output
    assert result.output.strip() == str(tmp_path / "mine" / RELPATH)


def test_bare_source_listing_caches_the_items_for_completion(tmp_path):
    pytest.importorskip("arena_simulation_setup.adapters.barn")
    result = runner.invoke(app, ["-s", "import", "world", "barn", "--cache", str(tmp_path)])
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert lines[:3] == ["0", "1", "2"]
    cached = json.loads((tmp_path / "items.world.json").read_text())
    assert cached["items"] == [str(index) for index in range(300)]
    assert cached["expires_at"] is None
