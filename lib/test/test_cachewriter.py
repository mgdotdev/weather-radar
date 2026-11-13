"""End-to-end tests for the single cache-writer process (cachewriter.py).

A fake `requests` module is installed before importing cachewriter so no real
NOAA calls are made. The writer runs in a real child process (fork), and we
exercise it through the public client helpers `ensure` / `ensure_stream`,
verifying:
  * ensure() blocks and atomically publishes a cache file;
  * ensure_stream() reports incremental 0..N progress;
  * atomic replace leaves no leftover .tmp files;
  * a still-fresh cache file is not re-fetched/rewritten.

These run on Linux (fork start method), which matches the container.
"""

import os
import sys
import json
import time
import types
import socket
import tempfile
import importlib.util
import multiprocessing

import pytest


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def cw(monkeypatch):
    """Import cachewriter with a fake `requests` and an isolated cache dir and
    port, then yield the module with a running writer process."""
    cache_dir = tempfile.mkdtemp()
    monkeypatch.setenv("CACHE_DIR", cache_dir)
    port = _free_port()
    monkeypatch.setenv("CACHE_WRITER_PORT", str(port))

    # Fake requests: every GET returns 200 with a far-future Expires.
    fake_requests = types.ModuleType("requests")

    # Provide a minimal `exceptions` namespace so _retry_get's
    # `except (requests.exceptions.ConnectionError, requests.exceptions.Timeout)`
    # resolves even though these are never raised here.
    fake_exceptions = types.ModuleType("requests.exceptions")
    class ConnectionError(Exception): ...
    class Timeout(Exception): ...
    fake_exceptions.ConnectionError = ConnectionError
    fake_exceptions.Timeout = Timeout
    fake_requests.exceptions = fake_exceptions

    class FakeResp:
        def __init__(self, url):
            self.status_code = 200
            self.headers = {"Expires": "Wed, 01 Jan 2099 00:00:00 GMT"}
            self._url = url

        def json(self):
            return {"ok": True, "url": self._url}

        def raise_for_status(self):
            pass

    class FakeSession:
        def __init__(self):
            self.headers = {}

        def get(self, url, timeout=None):
            return FakeResp(url)

    fake_requests.session = lambda: FakeSession()
    monkeypatch.setitem(sys.modules, "requests", fake_requests)

    # Load cachewriter fresh (so it reads the patched env + fake requests).
    here = os.path.dirname(__file__)
    src = os.path.abspath(os.path.join(
        here, "..", "src", "weather_radar", "lib", "cachewriter.py"))
    spec = importlib.util.spec_from_file_location("cachewriter_under_test", src)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "cachewriter_under_test", mod)
    spec.loader.exec_module(mod)

    ready = multiprocessing.Event()
    proc = multiprocessing.Process(target=mod.run_writer, args=(ready,), daemon=True)
    proc.start()
    assert ready.wait(timeout=10), "writer process never became ready"
    try:
        yield mod
    finally:
        proc.terminate()
        proc.join(timeout=5)


class TestCacheWriter:
    def test_ensure_writes_atomically(self, cw):
        cw.ensure(["/points/47.68,-116.78"])
        _, _, cache_file = cw._tempfile_for("/points/47.68,-116.78")
        assert os.path.isfile(cache_file)
        body = json.load(open(cache_file))
        assert body["json"]["ok"] is True
        assert body["headers"]["Expires"].startswith("Wed, 01 Jan 2099")

    def test_ensure_stream_progress(self, cw):
        paths = [f"/gridpoints/OTX/{i},{i}" for i in range(5)]
        events = list(cw.ensure_stream(paths))
        dones = [e["done"] for e in events]
        assert {e["total"] for e in events} == {5}
        assert dones[0] == 0 and dones[-1] == 5
        assert dones == sorted(dones)
        for p in paths:
            _, _, cf = cw._tempfile_for(p)
            assert os.path.isfile(cf)

    def test_no_leftover_temp_files(self, cw):
        cw.ensure([f"/gridpoints/OTX/{i},0" for i in range(3)])
        leftover = []
        for root, _dirs, files in os.walk(os.environ["CACHE_DIR"]):
            leftover += [f for f in files if f.endswith(".tmp")]
        assert leftover == []

    def test_fresh_file_not_refetched(self, cw):
        cw.ensure(["/points/1,1"])
        _, _, cache_file = cw._tempfile_for("/points/1,1")
        m1 = os.path.getmtime(cache_file)
        time.sleep(0.05)
        cw.ensure(["/points/1,1"])  # still fresh -> should be skipped
        assert os.path.getmtime(cache_file) == m1


class TestContentChangeMarker:
    """The change-marker sidecar (`data.json.changed`) tracks when CONTENT
    last changed, so derived artifacts (splines) refit only on a real change —
    NOT on every refresh. These unit-test the helpers directly (no writer
    process needed), mirroring how _fetch_one uses them."""

    def _load_cw(self, monkeypatch, cache_dir):
        monkeypatch.setenv("CACHE_DIR", cache_dir)
        # A minimal fake requests so the module imports (helpers don't use it).
        fake = types.ModuleType("requests")
        exc = types.ModuleType("requests.exceptions")
        exc.ConnectionError = type("ConnectionError", (Exception,), {})
        exc.Timeout = type("Timeout", (Exception,), {})
        fake.exceptions = exc
        fake.session = lambda: None
        monkeypatch.setitem(sys.modules, "requests", fake)
        here = os.path.dirname(__file__)
        src = os.path.abspath(os.path.join(
            here, "..", "src", "weather_radar", "lib", "cachewriter.py"))
        spec = importlib.util.spec_from_file_location("cw_marker_under_test", src)
        mod = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, "cw_marker_under_test", mod)
        spec.loader.exec_module(mod)
        return mod

    def test_identical_content_leaves_marker_mtime(self, monkeypatch):
        cache_dir = tempfile.mkdtemp()
        cw = self._load_cw(monkeypatch, cache_dir)
        cache_file = os.path.join(cache_dir, "data.json")

        payload = {"headers": {"Expires": "x"}, "json": {"values": [1, 2, 3]}}
        h = cw._content_hash(payload)
        with open(cache_file, "w") as f:
            json.dump(payload, f)
        cw._write_change_marker(cache_file, h)
        marker = cw._change_marker(cache_file)
        m0 = os.path.getmtime(marker)

        # Refetch returns identical content: hash matches -> DON'T touch marker.
        time.sleep(0.02)
        assert cw._read_existing_hash(cache_file) == h
        # (identical path in _fetch_one performs no _write_change_marker)
        assert os.path.getmtime(marker) == m0

    def test_changed_content_advances_marker_mtime(self, monkeypatch):
        cache_dir = tempfile.mkdtemp()
        cw = self._load_cw(monkeypatch, cache_dir)
        cache_file = os.path.join(cache_dir, "data.json")

        p1 = {"headers": {"Expires": "x"}, "json": {"values": [1, 2, 3]}}
        cw._write_change_marker(cache_file, cw._content_hash(p1))
        marker = cw._change_marker(cache_file)
        m0 = os.path.getmtime(marker)

        time.sleep(0.02)
        p2 = {"headers": {"Expires": "x"}, "json": {"values": [1, 2, 4]}}
        assert cw._content_hash(p2) != cw._content_hash(p1)
        cw._write_change_marker(cache_file, cw._content_hash(p2))
        assert os.path.getmtime(marker) > m0

    def test_content_hash_ignores_headers_and_key_order(self, monkeypatch):
        cw = self._load_cw(monkeypatch, tempfile.mkdtemp())
        a = {"headers": {"Expires": "A", "Date": "1"}, "json": {"b": 2, "a": 1}}
        b = {"headers": {"Expires": "B", "Date": "2"}, "json": {"a": 1, "b": 2}}
        assert cw._content_hash(a) == cw._content_hash(b)

    def test_read_existing_hash_falls_back_to_body(self, monkeypatch):
        """A legacy cache file with no marker: the stored body is hashed so an
        unchanged refetch is still recognised as unchanged."""
        cache_dir = tempfile.mkdtemp()
        cw = self._load_cw(monkeypatch, cache_dir)
        cache_file = os.path.join(cache_dir, "data.json")
        payload = {"headers": {"Expires": "x"}, "json": {"values": [9]}}
        with open(cache_file, "w") as f:
            json.dump(payload, f)
        # No marker written. Fallback must hash the body and match.
        assert not os.path.exists(cw._change_marker(cache_file))
        assert cw._read_existing_hash(cache_file) == cw._content_hash(payload)


class TestLocalPointsResolution:
    """Option A: the writer resolves /points/{lat},{lon} locally from cached
    /gridpoints polygons, synthesizing the /points doc and skipping the
    network. These exercise the helpers directly (no writer process / network),
    the same pattern as TestContentChangeMarker."""

    def _load_cw(self, monkeypatch, cache_dir):
        monkeypatch.setenv("CACHE_DIR", cache_dir)
        fake = types.ModuleType("requests")
        exc = types.ModuleType("requests.exceptions")
        exc.ConnectionError = type("ConnectionError", (Exception,), {})
        exc.Timeout = type("Timeout", (Exception,), {})
        fake.exceptions = exc
        fake.session = lambda: None
        monkeypatch.setitem(sys.modules, "requests", fake)
        here = os.path.dirname(__file__)
        src = os.path.abspath(os.path.join(
            here, "..", "src", "weather_radar", "lib", "cachewriter.py"))
        spec = importlib.util.spec_from_file_location("cw_points_under_test", src)
        mod = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, "cw_points_under_test", mod)
        spec.loader.exec_module(mod)
        return mod

    def _seed_gridpoint(self, cw, wfo, x, y, ring):
        """Write a cached /gridpoints doc carrying `ring` as its cell polygon."""
        _url, d, cache_file = cw._tempfile_for(f"/gridpoints/{wfo}/{x},{y}")
        os.makedirs(d, exist_ok=True)
        doc = {
            "headers": {"Expires": "Wed, 01 Jan 2099 00:00:00 GMT"},
            "json": {"geometry": {"type": "Polygon", "coordinates": [ring]}},
        }
        with open(cache_file, "w") as f:
            json.dump(doc, f)

    @staticmethod
    def _square(lon0, lat0, size):
        return [[lon0, lat0], [lon0 + size, lat0], [lon0 + size, lat0 + size],
                [lon0, lat0 + size], [lon0, lat0]]

    def test_point_inside_cached_polygon_resolves_without_network(self, monkeypatch):
        cache_dir = tempfile.mkdtemp()
        cw = self._load_cw(monkeypatch, cache_dir)
        # A cached gridpoint whose polygon is a 0.05deg square.
        self._seed_gridpoint(cw, "OTX", 164, 91, self._square(-116.70, 47.75, 0.05))

        # A point inside that square. _fetch_one would normally network-fetch;
        # here we drive the local resolver directly.
        path = "/points/47.76,-116.68"
        _url, d, cache_file = cw._tempfile_for(path)
        assert cw._resolve_points_locally(path, cache_file, d) is True

        # The synthesized /points doc is on disk, fresh, and carries the grid.
        assert os.path.isfile(cache_file)
        assert cw._is_fresh(cache_file)
        doc = json.load(open(cache_file))
        props = doc["json"]["properties"]
        assert (props["gridId"], props["gridX"], props["gridY"]) == ("OTX", 164, 91)

    def test_point_outside_any_polygon_is_not_resolved(self, monkeypatch):
        cache_dir = tempfile.mkdtemp()
        cw = self._load_cw(monkeypatch, cache_dir)
        self._seed_gridpoint(cw, "OTX", 164, 91, self._square(-116.70, 47.75, 0.05))
        # Far from the only cached polygon -> falls through to the network.
        path = "/points/40.0,-100.0"
        _url, d, cache_file = cw._tempfile_for(path)
        assert cw._resolve_points_locally(path, cache_file, d) is False
        assert not os.path.isfile(cache_file)

    def test_disabled_by_env(self, monkeypatch):
        cache_dir = tempfile.mkdtemp()
        cw = self._load_cw(monkeypatch, cache_dir)
        self._seed_gridpoint(cw, "OTX", 164, 91, self._square(-116.70, 47.75, 0.05))
        monkeypatch.setenv("WR_POLYGON_INDEX", "0")
        path = "/points/47.76,-116.68"
        _url, d, cache_file = cw._tempfile_for(path)
        # Even though the point is inside a cached polygon, the index is off.
        assert cw._resolve_points_locally(path, cache_file, d) is False
        assert not os.path.isfile(cache_file)

    def test_index_rebuilds_when_new_gridpoint_cached(self, monkeypatch):
        cache_dir = tempfile.mkdtemp()
        cw = self._load_cw(monkeypatch, cache_dir)
        self._seed_gridpoint(cw, "OTX", 164, 91, self._square(-116.70, 47.75, 0.05))

        # Point inside a not-yet-cached neighboring cell: initially unresolved.
        path = "/points/47.72,-116.74"
        _url, d, cache_file = cw._tempfile_for(path)
        assert cw._resolve_points_locally(path, cache_file, d) is False

        # Cache the neighbor; the dir mtime bumps, so the index rebuilds and
        # the same point now resolves locally.
        time.sleep(0.02)
        self._seed_gridpoint(cw, "OTX", 163, 90, self._square(-116.75, 47.70, 0.05))
        assert cw._resolve_points_locally(path, cache_file, d) is True
        doc = json.load(open(cache_file))
        props = doc["json"]["properties"]
        assert (props["gridId"], props["gridX"], props["gridY"]) == ("OTX", 163, 90)

    def test_non_points_path_ignored(self, monkeypatch):
        cw = self._load_cw(monkeypatch, tempfile.mkdtemp())
        assert cw._parse_points_path("/gridpoints/OTX/1,2") is None
        assert cw._parse_points_path("/points/47.1,-116.2") == (47.1, -116.2)
