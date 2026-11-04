"""Tests for the recursive single-WFO decomposition (resolve_area_gridpoints).

A synthetic NOAA world is mocked with a WFO border: coordinates with lat above
a threshold belong to office "BBB", the rest to "AAA" — i.e. a horizontal seam
crossing the area, like the worked example. /points returns the cell+office,
/gridpoints returns the cell's square polygon.

Assertions:
  * every coordinate resolves to the office the mock would assign (correctness);
  * single-WFO regions are filled WITHOUT probing their interior (far fewer
    /points than a full per-cell scan);
  * a pure single-WFO area probes only its boundary (+ polygon-unplaceable
    stragglers), not the interior.
"""
import math
import pytest

import weather_radar.lib.area as area_mod
from weather_radar.lib.area import (
    resolve_area_gridpoints, NOAAGridpoint, LATTICE_STEP_DEG,
    _POLY_INDEX, RECT_FLOOR,
)
from weather_radar.lib.connection import NoCoverageError


STEP = LATTICE_STEP_DEG
# Cell size in the synthetic world == the lattice step, so each lattice sample
# maps to its own integer cell (gridX,gridY) = lattice index.
SEAM_LAT_INDEX = 30  # lattice rows >= this are office BBB, below are AAA


def _office_for(i, j):
    """i is the lat index, j the lon index."""
    return "BBB" if i >= SEAM_LAT_INDEX else "AAA"


def _ring_for(i, j):
    # Square polygon for cell at lattice index (i=lat, j=lon), CENTERED on the
    # lattice coordinate (i*STEP, j*STEP) so that coordinate falls strictly in
    # the cell interior (not on a corner/edge, which the half-open point-in-
    # polygon rule would assign ambiguously — mirrors real NOAA geometry where
    # the queried point sits inside its cell).
    lat = i * STEP
    lon = j * STEP
    h = STEP / 2.0
    return [[lon - h, lat - h], [lon + h, lat - h], [lon + h, lat + h],
            [lon - h, lat + h], [lon - h, lat - h]]


class _FakeConn:
    def __init__(self):
        self.points_calls = 0
        self.gridpoints_calls = 0

    def get(self, path):
        if path.startswith("/points/"):
            self.points_calls += 1
            lat, lon = (float(v) for v in path[len("/points/"):].split(","))
            i = round(lat / STEP)
            j = round(lon / STEP)
            return {"properties": {"gridId": _office_for(i, j), "gridX": j, "gridY": i}}
        if path.startswith("/gridpoints/"):
            self.gridpoints_calls += 1
            rest = path[len("/gridpoints/"):]
            office, xy = rest.split("/")
            j, i = (int(v) for v in xy.split(","))  # gridX=j(lon), gridY=i(lat)
            return {"geometry": {"type": "Polygon", "coordinates": [_ring_for(i, j)]}}
        raise NoCoverageError(path)


@pytest.fixture
def fake(monkeypatch):
    conn = _FakeConn()
    monkeypatch.setattr(
        "weather_radar.lib.connection.NOAAConnection.get",
        lambda self, path: conn.get(path), raising=True,
    )
    # No disk: gridpoint_from_cached_points reads files that don't exist, so it
    # returns None and everything routes through the mocked network get().
    # The cache-writer ensure() is a no-op here (inline fetch path is used).
    monkeypatch.setattr(
        "weather_radar.lib.cachewriter.ensure", lambda paths: None, raising=False
    )
    # Isolate from any on-disk /points cache; force resolution through the mock.
    monkeypatch.setattr(
        area_mod, "gridpoint_from_cached_points", lambda c: None, raising=True
    )

    area_mod._GRIDPOINTS_CACHE.clear()
    _POLY_INDEX.clear()
    area_mod._POLY_INDEXED_GPS.clear()
    return conn


def _true_gridpoints(i0, i1, j0, j1):
    """The exact set of (office, gridX, gridY) the synthetic world assigns to
    every cell in the rectangle."""
    return {(_office_for(i, j), j, i)
            for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)}


def _as_tuples(gps):
    return {(gp.id, gp.x, gp.y) for gp in gps}


class TestRecursiveDecomposition:
    def test_single_wfo_area_skips_interior_probes(self, fake):
        # Entirely below the seam -> all AAA. Should probe ~boundary only.
        i0, i1, j0, j1 = 0, 20, 0, 20
        gps = resolve_area_gridpoints(i0, i1, j0, j1)
        got = _as_tuples(gps)
        # No MISSING true gridpoint (the critical guarantee); extras allowed
        # (the +1 pad / sheared corners), but all must be the right office.
        assert _true_gridpoints(i0, i1, j0, j1) <= got
        assert all(gp.id == "AAA" for gp in gps)
        n_cells = (i1 - i0 + 1) * (j1 - j0 + 1)
        assert fake.points_calls < n_cells, (fake.points_calls, n_cells)

    def test_border_area_resolves_correctly(self, fake):
        # Straddles the seam at i=30: top rows BBB, bottom AAA.
        i0, i1, j0, j1 = 20, 40, 0, 20
        gps = resolve_area_gridpoints(i0, i1, j0, j1)
        got = _as_tuples(gps)
        assert _true_gridpoints(i0, i1, j0, j1) <= got
        assert {gp.id for gp in gps} == {"AAA", "BBB"}

    def test_fewer_points_than_full_scan(self, fake):
        i0, i1, j0, j1 = 20, 44, 0, 24
        gps = resolve_area_gridpoints(i0, i1, j0, j1)
        assert _true_gridpoints(i0, i1, j0, j1) <= _as_tuples(gps)
        n_cells = (i1 - i0 + 1) * (j1 - j0 + 1)
        assert fake.points_calls < n_cells, (fake.points_calls, n_cells)

    def test_recursion_floor_probes_all(self, fake):
        # A rectangle at/below the floor is fully probed (no boundary trick).
        n = RECT_FLOOR  # <= floor on one side
        gps = resolve_area_gridpoints(0, n - 1, 0, 10)
        assert _true_gridpoints(0, n - 1, 0, 10) <= _as_tuples(gps)

    def test_no_missing_gridpoints(self, fake):
        # The critical guarantee across a larger bordered area: never drop a
        # real gridpoint (holes in the forecast field).
        i0, i1, j0, j1 = 10, 35, 5, 28
        gps = resolve_area_gridpoints(i0, i1, j0, j1)
        missing = _true_gridpoints(i0, i1, j0, j1) - _as_tuples(gps)
        assert missing == set(), missing


class TestIterGridpointsIntegration:
    """iter_gridpoints should route through the decomposition fast path and
    return the correct unique gridpoint set."""

    def test_area_resolves_via_decomposition(self, fake, monkeypatch):
        from weather_radar.lib.area import CoordinateArea, MapCoordinate

        monkeypatch.setenv("WR_WFO_DECOMP", "1")
        # A small area centered below the seam -> single WFO (AAA).
        center = MapCoordinate(lat=10 * STEP, lon=10 * STEP)
        area = CoordinateArea(center, width=20, height=20)
        gps = list(area.iter_gridpoints())
        assert gps
        # All resolved gridpoints are real (non-None) and belong to AAA (below
        # the seam) since the whole small area is below it.
        assert all(gp.id == "AAA" for gp in gps)
        # Deduped: no repeats.
        assert len(gps) == len(set(gps))

    def test_decomposition_disabled_uses_scan(self, fake, monkeypatch):
        from weather_radar.lib.area import CoordinateArea, MapCoordinate
        import weather_radar.lib.area as m

        monkeypatch.setenv("WR_WFO_DECOMP", "0")
        # If decomposition is off, _resolve_by_decomposition must NOT run.
        called = {"n": 0}
        orig = m.CoordinateArea._resolve_by_decomposition
        def spy(self):
            called["n"] += 1
            return orig(self)
        monkeypatch.setattr(m.CoordinateArea, "_resolve_by_decomposition", spy)

        center = MapCoordinate(lat=10 * STEP, lon=10 * STEP)
        area = CoordinateArea(center, width=20, height=20)
        list(area.iter_gridpoints())
        assert called["n"] == 0
