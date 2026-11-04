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
