"""Tests for the geocode proxy's upstream rate limit (<=1 req/s to Nominatim).

The gate serializes ACTUAL upstream calls so we respect Nominatim's absolute
1 req/s cap; cache hits bypass it entirely. These test the gate primitive and
its wiring into the endpoint (cache-hit bypass, load shedding) with a mocked
`requests` so no real Nominatim call is made.

FastAPI is only in the server container, so SKIPPED on a host without it.
"""
import time
import types

import pytest

pytest.importorskip("fastapi")
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import weather_radar.server.api.geocode as geocode  # noqa: E402


class TestRateGate:
    def test_sequential_calls_are_spaced(self):
        gate = geocode._RateGate(0.5)
        t0 = time.monotonic()
        assert gate.acquire(5.0)
        assert gate.acquire(5.0)
        assert gate.acquire(5.0)
        # 0 + 0.5 + 0.5 = ~1.0s for three.
        assert 0.9 <= time.monotonic() - t0 <= 1.3

    def test_sheds_when_slot_too_far_out(self):
        gate = geocode._RateGate(1.0)
        assert gate.acquire(5.0) is True
        # Next slot is ~1s out; a 0.1s max_wait must refuse.
        assert gate.acquire(0.1) is False

    def test_shed_does_not_consume_a_slot(self):
        gate = geocode._RateGate(1.0)
        assert gate.acquire(5.0) is True
        assert gate.acquire(0.0) is False  # shed
        # After the interval elapses, a new acquire still succeeds (the shed
        # didn't push _next_allowed further out).
        time.sleep(1.0)
        assert gate.acquire(5.0) is True


@pytest.fixture
def client(monkeypatch):
    """TestClient with a mocked Nominatim and a fresh fast gate/cache."""
    calls = {"n": 0}

    class FakeResp:
        ok = True
        status_code = 200

        def json(self):
            return [{"lat": "47.6", "lon": "-116.8", "display_name": "X"}]

    def fake_get(url, params=None, headers=None, timeout=None):
        calls["n"] += 1
        return FakeResp()

    fake_requests = types.ModuleType("requests")
    fake_requests.get = fake_get
    fake_requests.RequestException = Exception
    monkeypatch.setitem(__import__("sys").modules, "requests", fake_requests)

    # Isolate shared state and use a tiny interval so tests stay fast.
    geocode._cache.clear()
    monkeypatch.setattr(geocode, "_rate_gate", geocode._RateGate(0.3))

    app = FastAPI()
    app.include_router(geocode.router, prefix="/api")
    return TestClient(app), calls


class TestEndpointRateLimit:
    def test_cache_hit_bypasses_gate(self, client):
        c, calls = client
        # First call: miss -> one upstream call.
        r1 = c.get("/api/geocode/", params={"q": "spokane"})
        assert r1.status_code == 200 and calls["n"] == 1
        # Immediate repeat: cache HIT -> no upstream call, no rate wait.
        t0 = time.monotonic()
        r2 = c.get("/api/geocode/", params={"q": "spokane"})
        assert r2.status_code == 200
        assert calls["n"] == 1                       # still one upstream call
        assert time.monotonic() - t0 < 0.2           # not gated

    def test_distinct_misses_are_throttled(self, client):
        c, calls = client
        t0 = time.monotonic()
        c.get("/api/geocode/", params={"q": "a-town"})
        c.get("/api/geocode/", params={"q": "b-town"})
        # Two distinct queries -> two upstream calls, spaced by ~0.3s.
        assert calls["n"] == 2
        assert time.monotonic() - t0 >= 0.3
