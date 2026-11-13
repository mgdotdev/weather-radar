"""Tests for the compact point-summary endpoint (/api/models/point).

This is the machine-facing "current conditions at a lat/lon" endpoint (Home
Assistant REST sensors, scripts): one call returns all common attributes as a
flat {key: {value, uom}} object, plus an optional `hourly` array, across every
weather source.

FastAPI is only present in the server container, so these are SKIPPED on a host
without it (run them via `./run.sh test`). The per-attribute sampler
(`_point_sampler`) is monkeypatched so the tests exercise the endpoint's
shaping/aggregation logic deterministically, with NO network or cache
dependence.
"""
from datetime import datetime, timedelta

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import weather_radar.server.api.models as models_mod  # noqa: E402


@pytest.fixture
def client(monkeypatch):
    """A TestClient over just the models router, with a deterministic sampler.

    The fake sampler returns a constant value per attribute over a fixed 48h
    window, EXCEPT `wind_gust`, which returns None (simulating an attribute the
    active source can't provide — e.g. not in NDFD CONUS)."""
    now = datetime.now()
    start = now - timedelta(hours=1)
    end = now + timedelta(hours=48)

    _UOM = {"temperature": "degF", "relative-humidity": "percent",
            "quantitative-precipitation": "in"}
    _VAL = {"temperature": 50.0, "relative-humidity": 70.0,
            "quantitative-precipitation": 0.25}

    def fake_sampler(kind, param, coord):
        if param == "wind-gust":
            return None  # source has no data for this attribute
        val = _VAL.get(param, 1.0)
        uom = _UOM.get(param, "")

        def sample(when):
            if when.astimezone() < start.astimezone() or when.astimezone() > end.astimezone():
                return None
            return val

        return sample, uom, start, end

    monkeypatch.setattr(models_mod, "_point_sampler", fake_sampler)
    monkeypatch.setattr(models_mod, "weather_source", lambda: "gridpoints")

    app = FastAPI()
    app.include_router(models_mod.router, prefix="/api")
    return TestClient(app)


class TestPointSummary:
    def test_current_shape_and_values(self, client):
        r = client.get("/api/models/point", params={"lat": 47.6, "lon": -116.8})
        assert r.status_code == 200
        body = r.json()
        assert set(body) == {"meta", "current", "hourly"}
        assert body["meta"]["source"] == "gridpoints"
        assert body["meta"]["lat"] == 47.6 and body["meta"]["lon"] == -116.8
        assert body["meta"]["window"]["start"] and body["meta"]["window"]["end"]

        cur = body["current"]
        # All catalog keys present (stable shape), with value+uom sub-objects.
        assert set(cur) == set(models_mod._POINT_ATTRIBUTES)
        assert cur["temperature"] == {"value": 50.0, "uom": "degF"}
        assert cur["relative_humidity"] == {"value": 70.0, "uom": "percent"}
        assert cur["quantitative_precipitation"] == {"value": 0.25, "uom": "in"}
        # No hourly unless requested.
        assert body["hourly"] == []

    def test_missing_attribute_is_null_not_error(self, client):
        r = client.get("/api/models/point", params={"lat": 47.6, "lon": -116.8})
        cur = r.json()["current"]
        # wind_gust's sampler returned None -> null value, null uom, 200 OK.
        assert cur["wind_gust"] == {"value": None, "uom": None}

    def test_attrs_subset(self, client):
        r = client.get("/api/models/point", params={
            "lat": 47.6, "lon": -116.8,
            "attrs": "temperature,quantitative_precipitation,bogus_key",
        })
        cur = r.json()["current"]
        assert set(cur) == {"temperature", "quantitative_precipitation"}
        assert cur["temperature"]["value"] == 50.0

    def test_hourly_window_and_step(self, client):
        r = client.get("/api/models/point", params={
            "lat": 47.6, "lon": -116.8, "hours": 6, "step": 3600,
            "attrs": "temperature",
        })
        hourly = r.json()["hourly"]
        # now .. now+6h inclusive at 1h step -> 7 samples.
        assert len(hourly) == 7
        first = hourly[0]
        assert set(first) == {"time", "values"}
        assert first["values"]["temperature"]["value"] == 50.0
        # Times are ISO-8601 and strictly increasing.
        times = [h["time"] for h in hourly]
        assert times == sorted(times)

    def test_hours_capped(self, client):
        # hours is capped at 168 (7 days); a huge request doesn't explode.
        r = client.get("/api/models/point", params={
            "lat": 47.6, "lon": -116.8, "hours": 100000, "step": 3600,
            "attrs": "temperature",
        })
        hourly = r.json()["hourly"]
        assert len(hourly) <= 169  # 168h at 1h step + the inclusive endpoint

    def test_step_floor(self, client):
        # step below the floor (600s) is clamped up, so a 1h window can't
        # produce an unbounded number of samples.
        r = client.get("/api/models/point", params={
            "lat": 47.6, "lon": -116.8, "hours": 1, "step": 1,
            "attrs": "temperature",
        })
        hourly = r.json()["hourly"]
        # 1h / 600s step -> at most 7 samples (0,600,...,3600).
        assert 1 <= len(hourly) <= 7
