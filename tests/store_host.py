import email.message
import gzip
import hashlib
import http.server
import json
import urllib.parse


class Store(http.server.ThreadingHTTPServer):
    def __init__(self):
        super().__init__(("127.0.0.1", 0), StoreHandler)
        self.buckets: dict[str, dict[str, bytes]] = {"public": {}}
        self.private: set[str] = set()
        self.tokens: dict[str, str] = {}
        self.owners: dict[tuple[str, str], str] = {}
        self.failing: set[str] = set()
        self.generation = 0
        self.index_format: int | None = None
        self.requests: list[tuple[str, str, email.message.Message]] = []

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def put(self, bucket: str, name: str, data: bytes) -> None:
        self.buckets.setdefault(bucket, {})[name] = data
        self.generation += 1

    def index(self, bucket: str) -> bytes:
        objects = [{"name": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest(), "updated": "2026-11-03T00:00:00Z"} for name, data in sorted(self.buckets[bucket].items())]
        return json.dumps({"bucket": bucket, "generated": "2026-11-03T00:00:00Z", **({"format": self.index_format} if self.index_format is not None else {}), "objects": objects}).encode()

    def seen(self, method: str, path: str) -> list[email.message.Message]:
        return [headers for m, p, headers in self.requests if m == method and p == path]


class StoreHandler(http.server.BaseHTTPRequestHandler):
    server: Store

    def _target(self) -> tuple[str, str]:
        bucket, _, name = urllib.parse.unquote(urllib.parse.urlparse(self.path).path).removeprefix("/").partition("/")
        return bucket, name

    def _user(self) -> str | None:
        auth = self.headers.get("Authorization") or ""
        return self.server.tokens.get(auth.removeprefix("Bearer "))

    def _reply(self, status: int, body: bytes = b"", headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._reply(status, json.dumps({"error": message}).encode(), {"Content-Type": "application/json"})

    def _record(self) -> None:
        self.server.requests.append((self.command, urllib.parse.urlparse(self.path).path, self.headers))

    def do_GET(self):
        self._record()
        bucket, name = self._target()
        if bucket not in self.server.buckets:
            self._error(404, "no such bucket")
            return
        if bucket in self.server.private and self._user() is None:
            self._error(401, "bearer token required")
            return
        if name in self.server.failing:
            self._error(500, "broken")
            return
        if name == "index.json":
            etag = f'"{bucket}-{self.server.generation}"'
            if self.headers.get("If-None-Match") == etag:
                self._reply(304, headers={"ETag": etag})
                return
            body = self.server.index(bucket)
            headers = {"ETag": etag, "Content-Type": "application/json"}
            if "gzip" in (self.headers.get("Accept-Encoding") or ""):
                body = gzip.compress(body)
                headers["Content-Encoding"] = "gzip"
            self._reply(200, body, headers)
            return
        data = self.server.buckets[bucket].get(name)
        if data is None:
            self._error(404, "no such object")
            return
        ranged = self.headers.get("Range")
        if ranged:
            start = int(ranged.removeprefix("bytes=").removesuffix("-"))
            if start >= len(data):
                self._error(416, "range not satisfiable")
                return
            self._reply(206, data[start:], {"Content-Range": f"bytes {start}-{len(data) - 1}/{len(data)}"})
            return
        self._reply(200, data)

    def _writer(self, bucket: str, name: str) -> str | None:
        user = self._user()
        if user is None:
            self._error(401, "bearer token required")
            return None
        owner = self.server.owners.get((bucket, name))
        if owner is not None and owner != user:
            self._error(403, f"{bucket}/{name} belongs to {owner}")
            return None
        return user

    def do_PUT(self):
        self._record()
        bucket, name = self._target()
        user = self._writer(bucket, name)
        if user is None:
            return
        body = self.rfile.read(int(self.headers["Content-Length"]))
        if name in self.server.failing:
            self._error(500, "broken")
            return
        if self.headers.get("X-Content-SHA256") != hashlib.sha256(body).hexdigest():
            self._error(400, "sha256 mismatch")
            return
        if self.headers.get("If-None-Match") == "*" and name in self.server.buckets.get(bucket, {}):
            self._error(412, "exists")
            return
        self.server.put(bucket, name, body)
        self.server.owners[(bucket, name)] = user
        self._reply(201)

    def do_DELETE(self):
        self._record()
        bucket, name = self._target()
        if self._writer(bucket, name) is None:
            return
        if self.server.buckets.get(bucket, {}).pop(name, None) is None:
            self._error(404, "no such object")
            return
        self.server.generation += 1
        self._reply(204)

    def log_message(self, format: str, *args: object):
        pass
