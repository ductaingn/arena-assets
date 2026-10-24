import contextlib
import http.server
import logging
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
from pathlib import Path

import attrs
import pytest

import arena_assets
from arena_assets.importers.providers import HFHub, Http, Provider, logger

SRC = str(Path(arena_assets.__file__).parent.parent)


@attrs.frozen
class Reply:
    status: int = 200
    body: bytes = b""
    announced: int | None = None
    stall: float = 0.0


class ScriptedServer(http.server.ThreadingHTTPServer):
    """Answers request n with reply n of its script, and repeats the last reply after that."""

    def __init__(self, script: list[Reply]):
        super().__init__(("127.0.0.1", 0), ScriptedHandler)
        self.script = script
        self.seen: list[tuple[str, str, str | None]] = []

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


class ScriptedHandler(http.server.BaseHTTPRequestHandler):
    server: ScriptedServer

    def _reply(self):
        reply = self.server.script[min(len(self.server.seen), len(self.server.script) - 1)]
        self.server.seen.append((self.command, self.path, self.headers.get("Range")))
        time.sleep(reply.stall)
        with contextlib.suppress(ConnectionError):
            self.send_response(reply.status)
            self.send_header("Content-Length", str(len(reply.body) if reply.announced is None else reply.announced))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(reply.body)

    do_GET = _reply
    do_HEAD = _reply

    def log_message(self, format: str, *args: object):
        pass


@contextlib.contextmanager
def serving(*script: Reply):
    server = ScriptedServer(list(script))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class Messages(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.seen: list[str] = []

    def emit(self, record: logging.LogRecord):
        self.seen.append(record.getMessage())


@contextlib.contextmanager
def warnings_of_the_providers():
    handler = Messages()
    level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    try:
        yield handler.seen
    finally:
        logger.setLevel(level)
        logger.removeHandler(handler)


class Quick(Http):
    backoff = 0.0


class Impatient(Quick):
    timeout = 0.2


class Pausing(Http):
    backoff = 0.05


def test_get_retries_a_failing_host_until_it_serves(tmp_path):
    with serving(Reply(500), Reply(503), Reply(body=b"payload")) as server, warnings_of_the_providers() as warnings:
        dest = Quick(server.base).get("dir/file.bin", tmp_path / "file.bin")
    assert dest.read_bytes() == b"payload"
    assert server.seen == [("GET", "/dir/file.bin", None)] * 3
    assert sorted(p.name for p in tmp_path.iterdir()) == ["file.bin"]
    assert warnings == [
        f"{server.base}/dir/file.bin failed (HTTP Error 500: Internal Server Error), attempt 2 of 4 in 0 s",
        f"{server.base}/dir/file.bin failed (HTTP Error 503: Service Unavailable), attempt 3 of 4 in 0 s",
    ]


def test_get_gives_up_after_its_attempts_and_leaves_no_file(tmp_path):
    with serving(Reply(500)) as server, pytest.raises(urllib.error.HTTPError) as raised:
        Quick(server.base).get("file.bin", tmp_path / "file.bin")
    assert raised.value.code == 500
    assert len(server.seen) == Quick.attempts
    assert not (tmp_path / "file.bin").exists()


def test_get_does_not_retry_a_missing_file(tmp_path):
    with serving(Reply(404)) as server, pytest.raises(urllib.error.HTTPError) as raised:
        Quick(server.base).get("file.bin", tmp_path / "file.bin")
    assert raised.value.code == 404
    assert len(server.seen) == 1


def test_get_retries_a_body_cut_short(tmp_path):
    with serving(Reply(body=b"abcd", announced=10), Reply(body=b"abcdefghij")) as server:
        dest = Quick(server.base).get("file.bin", tmp_path / "file.bin")
    assert dest.read_bytes() == b"abcdefghij"
    assert len(server.seen) == 2


def test_get_never_keeps_a_body_cut_short(tmp_path):
    with serving(Reply(body=b"abcd", announced=10)) as server, pytest.raises(ConnectionError, match="body ended after 4 of 10 bytes"):
        Quick(server.base).get("file.bin", tmp_path / "file.bin")
    assert len(server.seen) == Quick.attempts
    assert not (tmp_path / "file.bin").exists()


def test_get_retries_a_host_that_stalls_past_the_timeout(tmp_path):
    with serving(Reply(body=b"late", stall=1.0), Reply(body=b"payload")) as server:
        dest = Impatient(server.base).get("file.bin", tmp_path / "file.bin")
    assert dest.read_bytes() == b"payload"
    assert len(server.seen) == 2


def test_get_repeats_its_range_on_every_attempt(tmp_path):
    with serving(Reply(502), Reply(206, body=b"cdef")) as server:
        dest = Quick(server.base).get("file.bin", tmp_path / "slice", range=(2, 5))
    assert dest.read_bytes() == b"cdef"
    assert server.seen == [("GET", "/file.bin", "bytes=2-5")] * 2


def test_size_and_read_retry(tmp_path):
    with serving(Reply(503), Reply(body=b"abcdef")) as server:
        assert Quick(server.base).size("file.bin") == 6
    assert [command for command, _, _ in server.seen] == ["HEAD", "HEAD"]
    with serving(Reply(504), Reply(body=b'{"a": 1}')) as server:
        body, headers = Quick(server.base).read(f"{server.base}/api/listing?id=1")
    assert body == b'{"a": 1}'
    assert headers["Content-Length"] == "8"
    assert server.seen == [("GET", "/api/listing?id=1", None)] * 2


def test_get_raises_once_a_host_that_refuses_connections_stays_down(tmp_path):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with pytest.raises(urllib.error.URLError):
        Quick(f"http://127.0.0.1:{port}").get("file.bin", tmp_path / "file.bin")
    assert list(tmp_path.iterdir()) == []


def test_pauses_double_between_attempts(tmp_path):
    with serving(Reply(500)) as server:
        start = time.monotonic()
        with pytest.raises(urllib.error.HTTPError):
            Pausing(server.base).get("file.bin", tmp_path / "file.bin")
        elapsed = time.monotonic() - start
    assert len(server.seen) == Pausing.attempts
    assert elapsed >= 0.05 + 0.1 + 0.2


def test_http_url_joins_and_quotes():
    assert Http("https://example.org/base/").url("/dir/a b.zip") == "https://example.org/base/dir/a%20b.zip"


def test_http_has_no_listing():
    with pytest.raises(NotImplementedError, match="http"):
        list(Http("https://example.org").list())


def test_hfhub_dataset_url():
    assert HFHub("Fhrozen/FSD50k").url("clips/dev/1.wav") == "https://huggingface.co/datasets/Fhrozen/FSD50k/resolve/main/clips/dev/1.wav"


def test_hfhub_model_url_and_revision():
    assert HFHub("org/model", repo_type="model", revision="v1.0").url("config.json") == "https://huggingface.co/org/model/resolve/v1.0/config.json"


def test_hfhub_tree_url():
    assert HFHub("Fhrozen/FSD50k")._tree_url("") == "https://huggingface.co/api/datasets/Fhrozen/FSD50k/tree/main?recursive=true"
    assert HFHub("Fhrozen/FSD50k")._tree_url("/metadata/") == "https://huggingface.co/api/datasets/Fhrozen/FSD50k/tree/main/metadata?recursive=true"


def _authorization(env: dict[str, str], token: str | None = None) -> str:
    """Authorization header an HFHub built in a child with exactly *env* sends, compared there so no token is printed."""
    script = f"import sys; from arena_assets.importers.providers import HFHub; header = HFHub('org/data', token={token!r})._request('https://huggingface.co').get_header('Authorization'); print({{None: 'none', 'Bearer env-token': 'env', 'Bearer file-token': 'file', 'Bearer path-token': 'path', 'Bearer tok': 'explicit'}}.get(header, 'other'))"
    base = {**{key: os.environ[key] for key in ("PATH", "VIRTUAL_ENV") if key in os.environ}, "PYTHONPATH": SRC}
    proc = subprocess.run([sys.executable, "-c", script], env={**base, **env}, capture_output=True, text=True, check=True)
    return proc.stdout.strip()


def _hf_home(root: Path, token: str | None) -> Path:
    home = root / "hf_home"
    home.mkdir()
    if token is not None:
        (home / "token").write_text(token + "\n")
    return home


def test_hfhub_token_header(tmp_path):
    req = HFHub("org/data", token="tok")._request("https://huggingface.co")
    assert req.get_header("Authorization") == "Bearer tok"
    empty = _hf_home(tmp_path, None)
    assert _authorization({"HF_HOME": str(empty), "HOME": str(tmp_path)}) == "none"


def test_hfhub_token_stays_off_a_redirect_to_another_host(host, tmp_path):
    hub = HFHub("org/data", token="tok", host=host.base)
    resolve = urllib.parse.urlsplit(hub.url("data/file.bin")).path
    host.redirects[resolve] = f"http://localhost:{host.server_address[1]}/cdn/file.bin"
    host.pages["/cdn/file.bin"] = b"payload"
    assert hub.get("data/file.bin", tmp_path / "file.bin").read_bytes() == b"payload"
    assert host.authorizations == {resolve: "Bearer tok", "/cdn/file.bin": None}


def test_hfhub_reads_the_token_file_under_hf_home(tmp_path):
    home = _hf_home(tmp_path, "file-token")
    assert _authorization({"HF_HOME": str(home), "HOME": str(tmp_path / "elsewhere")}) == "file"


def test_hfhub_defaults_to_the_cache_under_home(tmp_path):
    cache = tmp_path / ".cache" / "huggingface"
    cache.mkdir(parents=True)
    (cache / "token").write_text("file-token")
    assert _authorization({"HOME": str(tmp_path)}) == "file"


def test_hfhub_token_lookup_order(tmp_path):
    home = _hf_home(tmp_path, "file-token")
    (tmp_path / "path_token").write_text("path-token")
    assert _authorization({"HF_HOME": str(home), "HF_TOKEN_PATH": str(tmp_path / "path_token")}) == "path"
    assert _authorization({"HF_HOME": str(home), "HF_TOKEN": "env-token"}) == "env"
    assert _authorization({"HF_HOME": str(home), "HF_TOKEN": "env-token"}, token="tok") == "explicit"


def test_provider_names():
    assert Http.name == "http"
    assert HFHub.name == "hfhub"
    assert all(issubclass(cls, Provider) for cls in (Http, HFHub))


@pytest.mark.network
def test_hfhub_lists_metadata_tree():
    files = list(HFHub("Fhrozen/FSD50k").list("metadata"))
    assert "metadata/collection/collection_dev.csv" in files
    assert all(path.startswith("metadata/") for path in files)


@pytest.mark.network
def test_hfhub_gets_small_file_idempotently(tmp_path):
    hub = HFHub("Fhrozen/FSD50k")
    dest = hub.get("README.md", tmp_path / "README.md")
    assert dest.read_text().startswith("---")
    mtime = dest.stat().st_mtime_ns
    assert hub.get("README.md", dest) == dest
    assert dest.stat().st_mtime_ns == mtime
