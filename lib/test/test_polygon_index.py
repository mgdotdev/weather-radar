"""Tests for the cached-polygon spatial index: a sample that falls inside an
already-cached gridpoint's polygon is resolved locally, without a /points call.
This is what makes a small pan/zoom cheap (overlapping views reuse polygons).

A synthetic NOAA is mocked: a simple axis-aligned 0.05deg grid where /points
maps a coordinate to its cell and /gridpoints returns that cell's square
polygon. We resolve one area (warming the index), then a shifted overlapping
area, and assert the overlap is served from the index (far fewer /points).
"""
import math
import pytest

import weather_radar.lib.area as area_mod
from weather_radar.lib.area import (
    MapCoordinate, CoordinateArea, NOAAGridpoint,
    lookup_gridpoint_in_index, register_gridpoint_polygon,
    _POLY_INDEX,
)
from weather_radar.lib.connection import NoCoverageError


CELL = 0.05
LON0, LAT0 = -117.0, 47.0


def _cell_index(lon, lat):
    return int(math.floor((lon - LON0) / CELL)), int(math.floor((lat - LAT0) / CELL))


def _ring(x, y):
    lon = LON0 + x * CELL
    lat = LAT0 + y * CELL
    return [[lon, lat], [lon + CELL, lat], [lon + CELL, lat + CELL],
            [lon, lat + CELL], [lon, lat]]


class _FakeConn:
    def __init__(self):
        self.points_calls = 0
        self.gridpoints_calls = 0

    def get(self, path):
        if path.startswith("/points/"):
            self.points_calls += 1
            lat, lon = (float(v) for v in path[len("/points/"):].split(","))
            x, y = _cell_index(lon, lat)
            return {"properties": {"gridId": "AAA", "gridX": x, "gridY": y}}
        if path.startswith("/gridpoints/"):
            self.gridpoints_calls += 1
            rest = path[len("/gridpoints/"):]
            _office, xy = rest.split("/")
            x, y = (int(v) for v in xy.split(","))
            return {"geometry": {"type": "Polygon", "coordinates": [_ring(x, y)]}}
        raise NoCoverageError(path)


@pytest.fixture
def fake(monkeypatch):
    conn = _FakeConn()
    monkeypatch.setattr(
        "weather_radar.lib.connection.NOAAConnection.get",
        lambda self, path: conn.get(path),
        raising=True,
    )
    # get_with_expiry is used by _ring_from_cached_gridpoint only via disk; the
    # index_cached_gridpoints path reads disk files which don't exist in tests,
    # so instead we register polygons directly in tests that need the index
    # warmed. Clear shared state between tests.
    area_mod._GRIDPOINTS_CACHE.clear()
    _POLY_INDEX.clear()
    area_mod._POLY_INDEXED_GPS.clear()
    monkeypatch.setenv("WR_POLYGON_INDEX", "1")
    monkeypatch.setattr(
        "weather_radar.lib.cachewriter.ensure", lambda paths: None, raising=False
    )
    return conn


class TestPolygonIndex:
    def test_lookup_hit_and_miss(self, fake):
        gp = NOAAGridpoint("AAA", 3, 4)
        register_gridpoint_polygon(gp, _ring(3, 4))
        # A point inside the cell resolves to it.
        clon = LON0 + 3 * CELL + CELL / 2
        clat = LAT0 + 4 * CELL + CELL / 2
        assert lookup_gridpoint_in_index(clon, clat) == gp
        # A point far outside returns None.
        assert lookup_gridpoint_in_index(LON0 + 50 * CELL, LAT0 + 50 * CELL) is None

    def test_overlap_served_from_index(self, fake, monkeypatch):
        # Resolve an initial area and warm the index directly from the resolved
        # gridpoints' polygons (simulating what index_cached_gridpoints does
        # from disk, which isn't available in the test).
        a1 = CoordinateArea(MapCoordinate(47.5, -116.5), 20, 20)
        gps1 = list(a1.iter_gridpoints())
        assert gps1
        for gp in gps1:
            register_gridpoint_polygon(gp, _ring(gp.x, gp.y))
        calls_after_first = fake.points_calls
        assert calls_after_first > 0  # first (cold) area had to call /points

        # Now a heavily-overlapping area. Clear the resolved-set cache so it
        # must re-resolve samples, but the polygon index is warm.
        area_mod._GRIDPOINTS_CACHE.clear()
        fake.points_calls = 0
        a2 = CoordinateArea(MapCoordinate(47.52, -116.48), 20, 20)  # small shift
        gps2 = list(a2.iter_gridpoints())
        assert gps2
        # Most samples should be served from the index -> few/no /points calls.
        # (The shift exposes a thin new edge; interior is all index hits.)
        assert fake.points_calls < calls_after_first, (
            f"expected fewer /points on overlap: {fake.points_calls} vs {calls_after_first}"
        )

    def test_index_disabled(self, fake, monkeypatch):
        monkeypatch.setenv("WR_POLYGON_INDEX", "0")
        gp = NOAAGridpoint("AAA", 3, 4)
        register_gridpoint_polygon(gp, _ring(3, 4))
        area_mod._GRIDPOINTS_CACHE.clear()
        a = CoordinateArea(MapCoordinate(47.2, -116.85), 20, 20)
        list(a.iter_gridpoints())
        # With the index disabled, every sample goes to /points (no reuse).
        assert fake.points_calls > 0
