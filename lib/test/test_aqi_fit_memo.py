"""AQI fit performance: cells are gathered per frame (never whole-frame
stacks) and identical fits are computed once and shared."""
import threading

import numpy as np

from weather_radar.lib import aqi

H = 3600
T0 = 1_700_000_000


class _Tree:
    def __init__(self, cell=0):
        self.cell = cell

    def query(self, pts, k=1):
        n = len(pts)
        return np.zeros(n), np.full(n, self.cell, dtype=int)


class _Coord:
    def __init__(self, lat, lon):
        self.lat, self.lon = lat, lon


class _Area:
    def __init__(self, *cells):
        self.map_coordinates = [_Coord(*c) for c in (cells or [(45.0, -118.0)])]


def _grid(monkeypatch, n_cells=1, t0=T0, hours=24, level=5.0, cell=0):
    frames = [(t0 + i * H, np.full(n_cells, level + i * 0.1)) for i in range(hours)]
    g = {"size": 1, "tree": _Tree(cell), "frames": frames}
    monkeypatch.setattr(aqi, "ensure_grid", lambda pollutant="pm25": g)
    monkeypatch.setattr(aqi, "_prior_frames", lambda pollutant: None)
    return g


class TestGatherOnlyNeededCells:
    def test_no_whole_frame_stack(self, monkeypatch):
        _grid(monkeypatch, n_cells=200_000)            # big frames, 1-point area
        seen = []
        real = np.vstack
        monkeypatch.setattr(np, "vstack",
                            lambda arrs, *a, **k: (seen.extend(len(x) for x in arrs),
                                                   real(arrs, *a, **k))[1])
        fit = aqi.fit_area(_Area(), "pm25")
        assert fit is not None
        assert seen and max(seen) == 1                 # never 200_000 per frame

    def test_values_come_from_the_right_cell(self, monkeypatch):
        g = _grid(monkeypatch, n_cells=5, cell=3)
        for i, (_, v) in enumerate(g["frames"]):
            v[:] = 0.0
            v[3] = 12.0 + i                            # only cell 3 is "ours"
        epochs, series, *_ = aqi._combined_series(_Area(), "pm25")
        assert series.shape == (24, 1)
        assert list(series[:, 0]) == [12.0 + i for i in range(24)]


class TestFitMemo:
    def test_same_area_shares_one_fit(self, monkeypatch):
        _grid(monkeypatch)
        a = aqi.fit_area(_Area(), "pm25")
        assert aqi.fit_area(_Area(), "pm25") is a

    def test_different_area_is_a_different_fit(self, monkeypatch):
        _grid(monkeypatch)
        a = aqi.fit_area(_Area((45.0, -118.0)), "pm25")
        assert aqi.fit_area(_Area((46.0, -117.0)), "pm25") is not a

    def test_different_pollutant_variant_is_a_different_fit(self, monkeypatch):
        _grid(monkeypatch)
        assert aqi.fit_area(_Area(), "pm25") is not aqi.fit_area(_Area(), "pm25-nowcast")

    def test_new_cycle_refits(self, monkeypatch):
        _grid(monkeypatch, t0=T0)
        a = aqi.fit_area(_Area(), "pm25")
        _grid(monkeypatch, t0=T0 + 6 * H)               # a newer cycle is loaded
        b = aqi.fit_area(_Area(), "pm25")
        assert b is not a and b["start"] == T0 + 6 * H

    def test_same_shape_but_different_data_refits(self, monkeypatch):
        _grid(monkeypatch, level=5.0)
        a = aqi.fit_area(_Area(), "pm25")
        _grid(monkeypatch, level=40.0)                  # same epochs, new values
        b = aqi.fit_area(_Area(), "pm25")
        assert b is not a
        assert float(b["splines"][0](T0)) > float(a["splines"][0](T0))

    def test_unavailable_grid_is_not_cached(self, monkeypatch):
        monkeypatch.setattr(aqi, "ensure_grid", lambda pollutant="pm25": None)
        assert aqi.fit_area(_Area(), "pm25") is None
        _grid(monkeypatch)
        assert aqi.fit_area(_Area(), "pm25") is not None

    def test_dense_fit_is_shared_per_upsample(self, monkeypatch):
        _grid(monkeypatch, n_cells=1)
        area = _Area((45.0, -118.0), (45.0, -117.9), (45.1, -118.0), (45.1, -117.9))
        monkeypatch.setattr(aqi, "ensure_grid",
                            lambda pollutant="pm25": {"size": 1, "tree": _Tree(0),
                                "frames": [(T0 + i * H, np.full(4, 5.0 + i)) for i in range(24)]})
        d2 = aqi.fit_area_dense(area, "pm25", upsample=2)
        assert aqi.fit_area_dense(area, "pm25", upsample=2) is d2
        assert aqi.fit_area_dense(area, "pm25", upsample=3) is not d2

    def test_concurrent_identical_requests_compute_once(self, monkeypatch):
        _grid(monkeypatch)
        calls = []
        real = aqi._CONVERTER_ARRAY["pm25"]
        monkeypatch.setitem(aqi._CONVERTER_ARRAY, "pm25",
                            lambda c: (calls.append(1), real(c))[1])
        gate = threading.Barrier(6)
        out = []

        def go():
            gate.wait()
            out.append(aqi.fit_area(_Area(), "pm25"))

        ts = [threading.Thread(target=go) for _ in range(6)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        assert len(out) == 6 and all(o is out[0] for o in out)
        assert len(calls) == 1                          # one build, shared by all 6

    def test_cache_is_bounded(self, monkeypatch):
        monkeypatch.setattr(aqi, "_FIT_CACHE_MAX", 3)
        _grid(monkeypatch)
        for i in range(8):
            aqi.fit_area(_Area((40.0 + i, -118.0)), "pm25")
        assert len(aqi._FIT_CACHE) <= 3
