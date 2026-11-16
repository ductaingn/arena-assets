import json
import subprocess
import sys
import tomllib
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from arena_assets.impl.fetch import Bucket, _agent, package_version, user_agent

REPO = Path(__file__).resolve().parents[1]
PYTHON = f"Python-urllib/{sys.version_info.major}.{sys.version_info.minor}"
HEAD = "fcf7e76db938b462667b460aacd61f3e9604b067"


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.org", *args], capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def no_release(monkeypatch):
    monkeypatch.delenv("ARENA_FORKS_DIR", raising=False)
    monkeypatch.delenv("ARENA_RELEASE", raising=False)


@pytest.fixture
def checkout(tmp_path, no_release):
    root = tmp_path / "arena_assets"
    root.mkdir()
    (root / "package.xml").write_text("<package><version>1.0.0</version></package>")
    git(root, "init", "-q")
    git(root, "add", "package.xml")
    git(root, "commit", "-q", "-m", "init")
    return root


def release(tmp_path: Path, monkeypatch, repos: dict) -> Path:
    path = tmp_path / "forks" / "releases" / "r1" / "release.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"id": "r1", "repos": repos}))
    monkeypatch.setenv("ARENA_FORKS_DIR", str(tmp_path / "forks"))
    monkeypatch.setenv("ARENA_RELEASE", "r1")
    root = tmp_path / "src" / "Arena" / "arena_assets"
    root.mkdir(parents=True)
    return root


def test_checkout_agent_names_the_head_commit(checkout):
    assert _agent("1.0.0", checkout) == f"arena_assets/1.0.0+g{git(checkout, 'rev-parse', '--short=7', 'HEAD')} {PYTHON}"


def test_checkout_agent_marks_uncommitted_changes(checkout):
    (checkout / "package.xml").write_text("<package><version>1.0.1</version></package>")
    assert _agent("1.0.0", checkout) == f"arena_assets/1.0.0+g{git(checkout, 'rev-parse', '--short=7', 'HEAD')}.dirty {PYTHON}"


@pytest.mark.parametrize(("dirty", "local"), [(False, "+gfcf7e76"), (True, "+gfcf7e76.dirty")])
def test_fork_agent_reads_the_release_without_git(tmp_path, monkeypatch, dirty, local):
    root = release(tmp_path, monkeypatch, {".": {"head": "0" * 40, "dirty": False}, "Arena/arena_assets": {"head": HEAD, "dirty": dirty}})
    assert _agent("1.0.0", root) == f"arena_assets/1.0.0{local} {PYTHON}"


def test_agent_is_the_version_alone_without_git_or_release(tmp_path, no_release):
    assert _agent("1.0.0", tmp_path) == f"arena_assets/1.0.0 {PYTHON}"


def test_agent_without_the_repo_in_the_release_is_the_version_alone(tmp_path, monkeypatch):
    root = release(tmp_path, monkeypatch, {"Arena": {"head": HEAD, "dirty": False}})
    assert _agent("1.0.0", root) == f"arena_assets/1.0.0 {PYTHON}"


def test_failing_git_falls_back_to_the_release(checkout, tmp_path, monkeypatch):
    fork = tmp_path / "fork"
    fork.mkdir()
    root = release(fork, monkeypatch, {"Arena/arena_assets": {"head": HEAD, "dirty": False}})
    (root / ".git").write_text("gitdir: /nonexistent\n")
    monkeypatch.setenv("PATH", str(tmp_path / "no-git"))
    assert _agent("1.0.0", root) == f"arena_assets/1.0.0+gfcf7e76 {PYTHON}"
    assert _agent("1.0.0", checkout) == f"arena_assets/1.0.0 {PYTHON}"


def test_package_version_matches_package_xml_and_pyproject():
    declared = ET.parse(REPO / "package.xml").getroot().findtext("version")
    assert package_version() == declared
    assert tomllib.loads((REPO / "src" / "pyproject.toml").read_text())["project"]["version"] == declared


def test_store_requests_carry_the_agent(store):
    store.put("public", "models/a/x.obj", b"1")
    Bucket("public/models").listdir("")
    assert [headers["User-Agent"] for headers in store.seen("GET", "/public/index.json")] == [user_agent()]
    assert user_agent().startswith(f"arena_assets/{package_version()}")
