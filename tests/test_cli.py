import json

from typer.testing import CliRunner

from arena_assets.__main__ import app
from arena_assets.impl.fetch import package_version
from arena_assets.cli.db.query import parse_filters

runner = CliRunner()


def test_parse_filters():
    assert parse_filters([]) is None
    assert parse_filters(["height<0.75"]) == {"height": {"$lt": 0.75}}
    assert parse_filters(["height<=1", "volume>0.1"]) == {"$and": [{"height": {"$lte": 1.0}}, {"volume": {"$gt": 0.1}}]}


def test_parse_filters_equality_on_strings_and_booleans():
    assert parse_filters(["kind=alarm"]) == {"kind": {"$eq": "alarm"}}
    assert parse_filters(["loop=true", "kind = motor"]) == {"$and": [{"loop": {"$eq": True}}, {"kind": {"$eq": "motor"}}]}


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.splitlines()[0] == f"Arena Assets v{package_version()}" != "Arena Assets vunknown"


def test_invalid_log_level():
    result = runner.invoke(app, ["--log-level", "bogus", "version"])
    assert result.exit_code == 1


def test_db_query_outputs_path(database_path):
    result = runner.invoke(app, ["-s", "db", str(database_path), "query", "material", "wooden floor"])
    assert result.exit_code == 0
    assert result.output.strip() == str(database_path / "materials/oak")


def test_db_query_filter(database_path):
    result = runner.invoke(
        app,
        [
            "-s",
            "db",
            str(database_path),
            "query",
            "object",
            "chair to sit on",
            "-n",
            "2",
            "--filter",
            "height<1.0",
        ],
    )
    assert result.exit_code == 0
    assert result.output.strip() == str(database_path / "objects/chair")


def test_db_query_invalid_filter(database_path):
    result = runner.invoke(
        app,
        [
            "-s",
            "db",
            str(database_path),
            "query",
            "object",
            "chair",
            "--filter",
            "height!!1",
        ],
    )
    assert result.exit_code != 0


def test_db_query_scores(database_path):
    result = runner.invoke(
        app,
        [
            "-s",
            "db",
            str(database_path),
            "query",
            "material",
            "wooden floor",
            "-n",
            "2",
            "--scores",
        ],
    )
    assert result.exit_code == 0
    lines = result.output.strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        path, score = line.split("\t")
        float(score)


def test_net_default_lists_the_public_models(store):
    store.put("public", "models/Common/Object/a/annotation.yaml", b"a")
    store.put("public", "worlds/w1/world.yaml", b"w")
    result = runner.invoke(app, ["-s", "net", "default", "list"])
    assert result.exit_code == 0, result.output
    assert result.stdout.split() == ["Common/Object/a"]


def test_net_delete_removes_the_named_objects(store, monkeypatch):
    store.tokens["tok"] = "alice"
    monkeypatch.setenv("ARENA_ASSETS_TOKEN", "tok")
    store.put("arena-internal", "a.bin", b"1")
    store.put("arena-internal", "b.bin", b"2")
    result = runner.invoke(app, ["-s", "net", "arena-internal", "delete", "a.bin", "b.bin"])
    assert result.exit_code == 0, result.output
    assert store.buckets["arena-internal"] == {}


def test_net_delete_recursive_removes_a_folder_and_lists_what_went(store, monkeypatch):
    store.tokens["tok"] = "alice"
    monkeypatch.setenv("ARENA_ASSETS_TOKEN", "tok")
    for name in ["w/world.yaml", "w/0/map.png", "w2/world.yaml"]:
        store.put("arena-internal", name, b"x")
    result = runner.invoke(app, ["-s", "net", "arena-internal", "delete", "-r", "w"])
    assert result.exit_code == 0, result.output
    assert result.stdout.split() == ["w/0/map.png", "w/world.yaml"]
    assert list(store.buckets["arena-internal"]) == ["w2/world.yaml"]


def test_net_browse_prints_folders_then_files(store):
    store.put("public", "models/Common/Object/a/annotation.yaml", b"a")
    store.put("public", "models/Common/readme.md", b"r")
    result = runner.invoke(app, ["-s", "net", "default", "browse", "Common"])
    assert result.exit_code == 0, result.output
    lines = result.stdout.splitlines()
    assert lines[0].split()[0] == "Object/" and lines[0].endswith("1 file(s)")
    assert lines[1].split()[0] == "readme.md"


def test_net_browse_json(store):
    store.put("public", "worlds/w1/world.yaml", b"ww")
    result = runner.invoke(app, ["-s", "net", "public", "browse", "worlds", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"source": "public", "path": "worlds", "folders": [{"name": "w1", "files": 1, "bytes": 2}], "files": []}


def test_net_author_prints_what_it_sent_and_skipped(store, monkeypatch, tmp_path):
    store.tokens["tok"] = "alice"
    monkeypatch.setenv("ARENA_ASSETS_TOKEN", "tok")
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.txt").write_text("same")
    (tmp_path / "a" / "y.txt").write_text("new")
    store.put("arena-internal", "d/a/x.txt", b"same")
    result = runner.invoke(app, ["-s", "net", "arena-internal", "author", str(tmp_path), "-d", "d"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout.splitlines()[-1]) == {"files": 2, "sent": 1, "skipped": 1, "bytes": 3}


def test_net_errors_are_one_line_without_a_traceback(store):
    store.buckets["arena-internal"] = {}
    store.private.add("arena-internal")
    result = runner.invoke(app, ["-s", "net", "arena-internal", "browse"])
    assert result.exit_code == 1
    assert result.stderr.strip().splitlines() == ["error: GET arena-internal/index.json: 401 bearer token required, set ARENA_ASSETS_TOKEN or log in with 'gcloud auth login'"]
