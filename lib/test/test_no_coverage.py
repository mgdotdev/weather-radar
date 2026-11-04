"""Regression tests for out-of-coverage (e.g. Canada) handling.

NOAA's /points endpoint returns HTTP 404 (/problems/InvalidPoint) for any
coordinate it can't serve — outside the US, open ocean, etc. These tests pin
the graceful behavior:

  * retry_get treats 404 as PERMANENT (NoCoverageError), not transient — it
    fails fast instead of burning the full retry/backoff ladder;
  * gridpoint_from_map_coordinate returns None for such a point;
  * CoordinateArea.iter_gridpoints SKIPS the None points, so an area that
    straddles the coverage boundary resolves only its in-coverage gridpoints
    instead of aborting the whole resolution;
  * the single-point model surfaces a BoundsError ("no data here") rather than
    constructing a model on a None gridpoint.

All HTTP is mocked so these run offline and deterministically.
"""

import pytest

from weather_radar.lib import connection
from weather_radar.lib.connection import NoCoverageError, retry_get
from weather_radar.lib.area import (
    MapCoordinate, CoordinateArea, NOAAGridpoint,
    gridpoint_from_map_coordinate,
)
from weather_radar.lib.models.observation import ObservationModel
from weather_radar.lib.models.utils import BoundsError
import weather_radar.lib.area as area_mod


class _Resp:
    def __init__(self, status_code, headers=None, body=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._body = body or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code != 200:
            raise RuntimeError(f"HTTP {self.status_code}")


class _Session:
    """Minimal requests-session stand-in that returns a scripted status."""
    def __init__(self, status_code, body=None):
        self.calls = 0
        self._status = status_code
        self._body = body

    def get(self, url, timeout=None):
        self.calls += 1
        return _Resp(self._status, body=self._body)


class _FlakySession:
    """Raises a transient error for the first `fail_times` calls, then returns
    200. Models a brief DNS/connection blip."""
    def __init__(self, fail_times, exc):
        self.calls = 0
        self._fail_times = fail_times
        self._exc = exc

    def get(self, url, timeout=None):
        self.calls += 1
        if self.calls <= self._fail_times:
            raise self._exc
        return _Resp(200, body={"ok": True})


class TestRetryGet404:
    def test_404_raises_no_coverage_without_retry(self):
        session = _Session(404)
        with pytest.raises(NoCoverageError):
            retry_get(session, "https://api.weather.gov/points/56.1,-106.3")
        # Fast-fail: exactly one request, no backoff ladder.
        assert session.calls == 1

    def test_200_still_returns(self):
        session = _Session(200, body={"ok": True})
        resp = retry_get(session, "https://api.weather.gov/points/47.6,-116.7")
        assert resp.status_code == 200
        assert session.calls == 1


class TestRetryGetTransient:
    def test_dns_blip_is_retried(self, monkeypatch):
        import weather_radar.lib.connection as conn
        import requests
        # Don't actually sleep through the backoff.
        monkeypatch.setattr(conn.time, "sleep", lambda s: None)
        # Fail twice with a DNS/connection error, then succeed.
        err = requests.exceptions.ConnectionError("name resolution failed")
        session = _FlakySession(fail_times=2, exc=err)
        resp = retry_get(session, "https://api.weather.gov/gridpoints/MSO/58,132")
        assert resp.status_code == 200
        assert session.calls == 3  # 2 failures + 1 success

    def test_timeout_is_retried(self, monkeypatch):
        import weather_radar.lib.connection as conn
        import requests
        monkeypatch.setattr(conn.time, "sleep", lambda s: None)
        err = requests.exceptions.Timeout("read timed out")
        session = _FlakySession(fail_times=1, exc=err)
        resp = retry_get(session, "https://api.weather.gov/points/47.6,-116.7")
        assert resp.status_code == 200
        assert session.calls == 2

    def test_persistent_connection_error_eventually_raises(self, monkeypatch):
        import weather_radar.lib.connection as conn
        import requests
        monkeypatch.setattr(conn.time, "sleep", lambda s: None)
        err = requests.exceptions.ConnectionError("down")
        session = _FlakySession(fail_times=999, exc=err)
        with pytest.raises(Exception):
            retry_get(session, "https://api.weather.gov/points/47.6,-116.7")
        # Exhausts the retry ladder (6 retries): bounded, doesn't loop forever.
        assert session.calls <= 8


class TestGridpointResolution:
    def test_out_of_coverage_returns_none(self, monkeypatch):
        # Mock the connection so /points resolution raises NoCoverageError.
        def fake_get(self, path):
            raise NoCoverageError(path)
        monkeypatch.setattr(
            "weather_radar.lib.connection.NOAAConnection.get", fake_get, raising=True
        )
        assert gridpoint_from_map_coordinate(MapCoordinate(56.1, -106.3)) is None

    def test_in_coverage_returns_gridpoint(self, monkeypatch):
        def fake_get(self, path):
            return {"properties": {"gridId": "OTX", "gridX": 50, "gridY": 100}}
        monkeypatch.setattr(
            "weather_radar.lib.connection.NOAAConnection.get", fake_get, raising=True
        )
        gp = gridpoint_from_map_coordinate(MapCoordinate(47.6, -116.7))
        assert gp == NOAAGridpoint("OTX", 50, 100)

    def test_mixed_area_skips_out_of_coverage(self, monkeypatch):
        # Simulate an area straddling the border: coordinates with lat >= 49
        # (roughly the US/Canada line) have no coverage; the rest resolve to a
        # single shared gridpoint. iter_gridpoints must yield the in-coverage
        # gridpoint(s) and silently skip the None ones.
        covered = NOAAGridpoint("OTX", 50, 100)

        def fake_resolve(coordinate):
            return None if coordinate.lat >= 49.0 else covered

        monkeypatch.setattr(area_mod, "gridpoint_from_map_coordinate", fake_resolve)
        # Neutralize the writer pre-warm (no network / no cachewriter needed).
        monkeypatch.setattr(
            "weather_radar.lib.cachewriter.ensure", lambda paths: None, raising=False
        )
        # Start from a clean resolved-gridpoint cache so a prior test's entry
        # can't satisfy this area via the nearby-reuse path.
        area_mod._GRIDPOINTS_CACHE.clear()

        # Center near the border so the sample lattice spans both sides.
        area = CoordinateArea(MapCoordinate(49.0, -116.7), 40, 40)
        gps = list(area.iter_gridpoints())
        # Only the in-coverage gridpoint survives; no None leaks through.
        assert covered in gps
        assert all(gp is not None for gp in gps)
        assert set(gps) == {covered}


class TestSinglePointModel:
    def test_point_model_out_of_coverage_raises_bounds(self, monkeypatch):
        # The model imports gridpoint_from_map_coordinate into its OWN module
        # namespace, so patch it there (not in area).
        monkeypatch.setattr(
            "weather_radar.lib.models.observation.gridpoint_from_map_coordinate",
            lambda c: None,
            raising=True,
        )
        with pytest.raises(BoundsError):
            ObservationModel.from_map_coordinate(
                MapCoordinate(56.1, -106.3), "temperature"
            )
