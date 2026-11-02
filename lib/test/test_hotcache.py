"""Tests for the hot-cache scheduler engine (hotcache._Scheduler).

These exercise the in-process scheduling logic directly (no manager/process):
URL resolution, the expiry heap, dynamic watch/unwatch, due-detection with
lead time, and rescheduling by new expiry after a refresh. `cachewriter.ensure`
and the expiry/URL helpers are monkeypatched so the tests are offline and
deterministic.
"""

import os
import json
import time
import types
import importlib.util

import pytest


def _load_hotcache(monkeypatch, tmp_path):
    """Load hotcache with a stub cachewriter and isolated CACHE_DIR."""
    monkeypatch.setenv("CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("HOTCACHE_LEAD_SECONDS", "120")

    # Stub the cachewriter module hotcache imports (`from . import cachewriter`).
    fake_cw = types.ModuleType("weather_radar.lib.cachewriter")
    fake_cw.ensured = []
    def ensure(paths):
        fake_cw.ensured.append(list(paths))
    fake_cw.ensure = ensure

    import sys
    monkeypatch.setitem(sys.modules, "weather_radar.lib.cachewriter", fake_cw)

    # Load hotcache.py by path under a package-y name so its relative import
    # `from . import cachewriter` resolves to our stub.
    here = os.path.dirname(__file__)
    src = os.path.abspath(os.path.join(
        here, "..", "src", "weather_radar", "lib", "hotcache.py"))

    # Create a lightweight package shell so relative imports work.
    pkg = types.ModuleType("wr_lib_pkg")
    pkg.__path__ = [os.path.dirname(src)]
    monkeypatch.setitem(sys.modules, "wr_lib_pkg", pkg)
    monkeypatch.setitem(sys.modules, "wr_lib_pkg.cachewriter", fake_cw)

    spec = importlib.util.spec_from_file_location("wr_lib_pkg.hotcache", src)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "wr_lib_pkg.hotcache", mod)
    spec.loader.exec_module(mod)
    return mod, fake_cw


@pytest.fixture
def hc(monkeypatch, tmp_path):
    mod, fake_cw = _load_hotcache(monkeypatch, tmp_path)
    # Deterministic URL -> paths and expiry, so we don't touch the network.
    url_map = {
        "http://x/map/models/temperature/?lat=1&lon=2&width=3&height=4":
            ["/points/1,2", "/gridpoints/OTX/1,1", "/gridpoints/OTX/2,2"],
    }
    expiries = {}  # path -> timestamp
    monkeypatch.setattr(mod, "paths_for_url", lambda u: list(url_map[u]))
    monkeypatch.setattr(mod, "expiry_of", lambda p: expiries.get(p, 0.0))
    return types.SimpleNamespace(mod=mod, cw=fake_cw, url_map=url_map, expiries=expiries)


URL = "http://x/map/models/temperature/?lat=1&lon=2&width=3&height=4"


class TestScheduler:
    def test_watch_populates_heap(self, hc):
        s = hc.mod._Scheduler()
        res = json.loads(s.watch(URL))
        assert res["ok"] and res["paths"] == 3
        assert len(s._heap) == 3
        assert URL in s._url_paths

    def test_watch_persists_and_reloads(self, hc):
        s = hc.mod._Scheduler()
        s.watch(URL)
        # A fresh scheduler should reload the persisted URL.
        s2 = hc.mod._Scheduler()
        s2._load_persisted()
        assert URL in s2._url_paths

    def test_unwatch_orphans_dropped(self, hc):
        s = hc.mod._Scheduler()
        s.watch(URL)
        s.unwatch(URL)
        assert URL not in s._url_paths
        # Heap entries remain but are all orphaned now.
        assert all(not s._is_watched_path(p) for _, p in s._heap)

    def test_list_watched(self, hc):
        s = hc.mod._Scheduler()
        s.watch(URL)
        info = json.loads(s.list_watched())
        assert info["urls"] == [URL]
        assert info["paths"] == 3
        assert info["scheduled"] == 3

    def test_due_detection_respects_lead(self, hc):
        # Make all paths expire 60s from now; with LEAD=120 they are "due" now.
        now = time.time()
        for p in hc.url_map[URL]:
            hc.expiries[p] = now + 60
        s = hc.mod._Scheduler()
        s.watch(URL)
        # Simulate one scan of the loop's due-collection step.
        due = []
        import heapq
        while s._heap and s._heap[0][0] - hc.mod.LEAD_SECONDS <= time.time():
            _exp, path = heapq.heappop(s._heap)
            if s._is_watched_path(path):
                due.append(path)
        assert set(due) == set(hc.url_map[URL])

    def test_not_due_when_far_future(self, hc):
        now = time.time()
        for p in hc.url_map[URL]:
            hc.expiries[p] = now + 10_000  # well beyond lead
        s = hc.mod._Scheduler()
        s.watch(URL)
        soonest = s._heap[0][0]
        assert soonest - hc.mod.LEAD_SECONDS > time.time()
