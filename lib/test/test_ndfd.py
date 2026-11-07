"""Tests for the NDFD GRIB weather source (lib/ndfd.py) and its ensemble
adapters (lib/models/ndfd_model.py), plus the env-var source selection
(lib/models.weather_source / ensemble_kinds).

These avoid the network / real GRIB: the pure logic (element mapping, unit
conversions, freshness TTL, path/url construction, source selection) is tested
directly, and the ensemble adapter is tested against a monkeypatched
`ndfd.fit_area` so the ObservationEnsemble-contract behavior is verified
without eccodes or a GRIB file.
"""
import os
import time
from datetime import datetime, timedelta

import pytest

from weather_radar.lib import ndfd
from weather_radar.lib.area import MapCoordinate
from weather_radar.lib.models.utils import BoundsError


class _StubArea:
    """Minimal area exposing just what ndfd.fit_area / the ensemble need."""
    def __init__(self, n=3):
        self._n = n
        self.time_zone = "America/Los_Angeles"

    @property
    def map_coordinates(self):
        base_lat, base_lon = 45.6, -118.8
        return [MapCoordinate(lat=base_lat + i * 0.1, lon=base_lon + j * 0.1)
                for i in range(self._n) for j in range(self._n)]


# --- Element mapping + unit conversions -------------------------------------

class TestElementMapping:
    def test_supported_attributes(self):
        for a in ("temperature", "dewpoint", "relative-humidity", "wind-speed",
                  "sky-cover", "quantitative-precipitation", "visibility"):
            assert ndfd.supports(a)

    def test_unsupported_attributes(self):
        # NDFD CONUS doesn't publish these standard elements.
        for a in ("heat-index", "wind-chill", "pressure", "not-a-real-field"):
            assert not ndfd.supports(a)

    def test_kelvin_to_fahrenheit(self):
        spec = ndfd.ELEMENTS["temperature"]
        # 273.15K = 32F, 300K ~= 80.33F
        assert spec["convert"](273.15) == pytest.approx(32.0)
        assert spec["convert"](300.0) == pytest.approx(80.33, abs=0.01)
        assert spec["units"] == "degF"

    def test_ms_to_mph(self):
        spec = ndfd.ELEMENTS["wind-speed"]
        assert spec["convert"](10.0) == pytest.approx(22.369, abs=0.01)
        assert spec["units"] == "mph"
        assert spec["nonneg"] is True

    def test_mm_to_inches(self):
        spec = ndfd.ELEMENTS["quantitative-precipitation"]
        assert spec["convert"](25.4) == pytest.approx(1.0)

    def test_percent_passthrough(self):
        # RH/sky/pop are already %, no conversion.
        assert ndfd.ELEMENTS["relative-humidity"]["convert"] is None
        assert ndfd.ELEMENTS["sky-cover"]["convert"] is None


# --- URL / path construction ------------------------------------------------

class TestPaths:
    def test_element_url(self):
        url = ndfd._element_url("temp", "VP.001-003")
        assert url.endswith("/AR.conus/VP.001-003/ds.temp.bin")

    def test_element_path_is_sector_range_qualified(self):
        p = ndfd._element_path("temp", "VP.001-003")
        base = os.path.basename(p)
        # Sector + range encoded so different sectors/ranges don't collide.
        assert base == "ndfd.ARconus.VP001-003.temp.grib2"

    def test_time_ranges_cover_week(self):
        assert ndfd.TIME_RANGES == ("VP.001-003", "VP.004-007")


# --- Freshness (TTL, since tgftp sends no Expires) --------------------------

class TestFreshness:
    def test_missing_file_not_fresh(self, tmp_path):
        assert ndfd._is_fresh(str(tmp_path / "nope.grib2")) is False

    def test_recent_file_fresh(self, tmp_path, monkeypatch):
        f = tmp_path / "f.grib2"
        f.write_bytes(b"x")
        monkeypatch.setattr(ndfd, "NDFD_TTL_SECONDS", 3600)
        assert ndfd._is_fresh(str(f)) is True

    def test_old_file_not_fresh(self, tmp_path, monkeypatch):
        f = tmp_path / "f.grib2"
        f.write_bytes(b"x")
        monkeypatch.setattr(ndfd, "NDFD_TTL_SECONDS", 3600)
        old = time.time() - 7200  # 2h ago, TTL 1h
        os.utime(str(f), (old, old))
        assert ndfd._is_fresh(str(f)) is False


class TestMissingRangeNegativeCache:
    """Not every element is published in every time range (e.g. ds.snow.bin
    404s for VP.004-007). A 404 must be negative-cached so it isn't re-probed
    on every request."""

    def test_404_is_negative_cached(self, monkeypatch):
        monkeypatch.setattr(ndfd, "NDFD_MISS_TTL_SECONDS", 60)
        ndfd._MISSING.clear()

        calls = []

        class _Resp:
            ok = False
            status_code = 404
            def __enter__(self): return self
            def __exit__(self, *a): return False

        import types, sys
        fake = types.ModuleType("requests")
        fake.get = lambda url, **kw: (calls.append(url) or _Resp())
        monkeypatch.setitem(sys.modules, "requests", fake)
        # Force the writer-unavailable (inline) path so the stubbed requests.get
        # is exercised and the HTTP status is seen.
        import weather_radar.lib.cachewriter as cw
        monkeypatch.setattr(cw, "ensure_binary",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError()))

        p1 = ndfd._download("snow", "VP.004-007")
        assert p1 is None
        assert len(calls) == 1
        assert ndfd._missing_recent("snow", "VP.004-007") is True

        # Within the miss TTL: no new network probe.
        p2 = ndfd._download("snow", "VP.004-007")
        assert p2 is None
        assert len(calls) == 1  # unchanged

    def test_miss_expires_allows_retry(self, monkeypatch):
        monkeypatch.setattr(ndfd, "NDFD_MISS_TTL_SECONDS", 60)
        ndfd._MISSING.clear()
        ndfd._mark_missing("snow", "VP.004-007")
        assert ndfd._missing_recent("snow", "VP.004-007") is True
        # Simulate the retry window elapsing by back-dating the stored retry-at.
        ndfd._MISSING[("snow", "VP.004-007")] = time.time() - 1
        assert ndfd._missing_recent("snow", "VP.004-007") is False


# --- Source selection (WR_WEATHER_SOURCE) -----------------------------------

class TestSourceSelection:
    def test_default_is_ndfd(self, monkeypatch):
        monkeypatch.delenv("WR_WEATHER_SOURCE", raising=False)
        import weather_radar.lib.models as models
        assert models.weather_source() == "ndfd"
        kinds = models.ensemble_kinds()
        assert kinds["observation"].__name__ == "NDFDObservationEnsemble"
        assert kinds["accumulation"].__name__ == "NDFDAccumulationEnsemble"

    def test_gridpoints_selected(self, monkeypatch):
        monkeypatch.setenv("WR_WEATHER_SOURCE", "gridpoints")
        import weather_radar.lib.models as models
        assert models.weather_source() == "gridpoints"
        kinds = models.ensemble_kinds()
        assert kinds["observation"].__name__ == "ObservationEnsemble"
        assert kinds["accumulation"].__name__ == "AccumulationEnsemble"

    def test_ndfd_selected(self, monkeypatch):
        monkeypatch.setenv("WR_WEATHER_SOURCE", "ndfd")
        import weather_radar.lib.models as models
        assert models.weather_source() == "ndfd"
        kinds = models.ensemble_kinds()
        assert kinds["observation"].__name__ == "NDFDObservationEnsemble"
        assert kinds["accumulation"].__name__ == "NDFDAccumulationEnsemble"

    def test_unknown_source_falls_back_to_ndfd(self, monkeypatch):
        monkeypatch.setenv("WR_WEATHER_SOURCE", "bogus")
        import weather_radar.lib.models as models
        assert models.weather_source() == "ndfd"


# --- Ensemble adapter (ObservationEnsemble contract) ------------------------
# Uses a fake fit (no GRIB / eccodes): two lattice points with simple linear
# splines over a known window, so predictions are exactly checkable.

def _fake_fit(monkeypatch, units="degF", nonneg=False):
    from scipy.interpolate import CubicSpline
    start = int(datetime(2026, 11, 5, 0, 0).timestamp())
    end = start + 6 * 3600
    xs = [start + h * 3600 for h in range(7)]
    # point A: y = hour index (0..6); point B: y = 10 + hour index.
    sa = CubicSpline(xs, list(range(7)))
    sb = CubicSpline(xs, [10 + h for h in range(7)])
    fit = {
        "start": start, "end": end,
        "points": [MapCoordinate(lat=45.6, lon=-118.8),
                   MapCoordinate(lat=45.7, lon=-118.7)],
        "splines": [sa, sb],
        "units": units, "nonneg": nonneg,
    }
    monkeypatch.setattr(ndfd, "fit_area",
                        lambda area, attr, accumulate=False: fit)
    monkeypatch.setattr(ndfd, "grid_mtime", lambda attr: 123.0)
    return start, end


class TestNDFDEnsemble:
    def _ensemble(self, monkeypatch, **kw):
        from weather_radar.lib.models.ndfd_model import NDFDObservationEnsemble
        start, end = _fake_fit(monkeypatch, **kw)
        ens = NDFDObservationEnsemble(_StubArea(), "temperature")
        return ens, start, end

    def test_iter_points_count(self, monkeypatch):
        ens, _, _ = self._ensemble(monkeypatch)
        assert len(list(ens.iter_points())) == 2

    def test_model_tuple_shape(self, monkeypatch):
        ens, start, end = self._ensemble(monkeypatch)
        m = next(iter(ens.iter_points()))
        st, et, center, f, units, expiry = m.model
        assert isinstance(st, datetime) and isinstance(et, datetime)
        # tz-aware (matches the JSON model, else predict() comparison raises).
        assert st.tzinfo is not None
        assert units == "degF"
        # f maps seconds-since-start -> value; point A is y=hour index.
        assert f(0).item() == pytest.approx(0.0, abs=1e-6)
        assert f(3600).item() == pytest.approx(1.0, abs=1e-6)

    def test_predict_within_window(self, monkeypatch):
        ens, start, _ = self._ensemble(monkeypatch)
        m = next(iter(ens.iter_points()))
        st = m.model[0]
        # 2h after start -> point A value 2.0
        assert m.predict(st + timedelta(hours=2)) == pytest.approx(2.0, abs=1e-6)

    def test_predict_out_of_window_raises(self, monkeypatch):
        ens, _, _ = self._ensemble(monkeypatch)
        m = next(iter(ens.iter_points()))
        st = m.model[0]
        with pytest.raises(BoundsError):
            m.predict(st + timedelta(days=30))

    def test_predict_verbose_feature(self, monkeypatch):
        ens, _, _ = self._ensemble(monkeypatch)
        m = next(iter(ens.iter_points()))
        st = m.model[0]
        feat = m.predict(st + timedelta(hours=1), verbose=True)
        assert feat["type"] == "Feature"
        assert feat["properties"]["uom"] == "degF"
        # geometry coords are [lon, lat] (GeoJSON convention; the client's
        # normalizeFrame reads coordinates[0]=lon), consistent with the series
        # + AQI paths.
        assert feat["geometry"]["coordinates"] == [-118.8, 45.6]

    def test_iter_predict_slim_values(self, monkeypatch):
        ens, _, _ = self._ensemble(monkeypatch)
        st = next(iter(ens.iter_points())).model[0]
        vals = list(ens.iter_predict(st + timedelta(hours=2), verbose=False))
        # Two points: A=2.0, B=12.0 at t+2h.
        assert vals == pytest.approx([2.0, 12.0], abs=1e-6)

    def test_iter_predict_out_of_window_empty(self, monkeypatch):
        ens, _, _ = self._ensemble(monkeypatch)
        st = next(iter(ens.iter_points())).model[0]
        assert list(ens.iter_predict(st + timedelta(days=30))) == []

    def test_value_range(self, monkeypatch):
        ens, _, _ = self._ensemble(monkeypatch)
        rng = ens.value_range()
        # Across both points over the window: min 0 (A@start), max 16 (B@end).
        assert rng["min"] == pytest.approx(0.0, abs=1e-6)
        assert rng["max"] == pytest.approx(16.0, abs=1e-6)
        assert rng["uom"] == "degF"
        assert rng["start"] and rng["end"]

    def test_nonneg_clamp_applied(self, monkeypatch):
        # With nonneg=True and an attribute in NON_NEGATIVE_ATTRIBUTES, a
        # negative spline value is clamped at predict().
        from weather_radar.lib.models.ndfd_model import NDFDObservationEnsemble
        from scipy.interpolate import CubicSpline
        start = int(datetime(2026, 11, 5, 0, 0).timestamp())
        xs = [start + h * 3600 for h in range(7)]
        neg = CubicSpline(xs, [-5] * 7)
        fit = {"start": start, "end": start + 6 * 3600,
               "points": [MapCoordinate(lat=45.6, lon=-118.8)],
               "splines": [neg], "units": "mph", "nonneg": True}
        monkeypatch.setattr(ndfd, "fit_area", lambda a, at, accumulate=False: fit)
        monkeypatch.setattr(ndfd, "grid_mtime", lambda at: 1.0)
        ens = NDFDObservationEnsemble(_StubArea(), "wind-speed")
        m = next(iter(ens.iter_points()))
        st = m.model[0]
        assert m.predict(st + timedelta(hours=1)) == 0.0

    def test_no_data_returns_none_range(self, monkeypatch):
        from weather_radar.lib.models.ndfd_model import NDFDObservationEnsemble
        monkeypatch.setattr(ndfd, "fit_area", lambda a, at, accumulate=False: None)
        ens = NDFDObservationEnsemble(_StubArea(), "pressure")
        assert ens.value_range() is None
        assert list(ens.iter_points()) == []


# --- Accumulation integration (cumulative storm total, monotone) ------------
# NDFD publishes qpf/snow/ice as PER-INTERVAL amounts. fit_area(accumulate=True)
# must integrate them into a running total from the window start, fit with a
# monotone interpolator so the total never decreases. These stub _load_grid
# (a real cKDTree + synthetic per-interval frames) to drive fit_area directly.

class TestAccumulationFit:
    def _stub_grid(self, monkeypatch, per_interval_m):
        """Install a _load_grid stub: a single lattice point whose per-interval
        snowfall (metres) over N hourly frames is `per_interval_m`."""
        import numpy as np
        from scipy.spatial import cKDTree
        start = int(datetime(2026, 11, 5, 0, 0).timestamp())
        # One grid cell at the StubArea's first lattice point (45.6, -118.8).
        tree = cKDTree(np.array([[45.6, -118.8]]))
        frames = [(float(start + h * 3600), np.array([per_interval_m[h]],
                                                     dtype=float))
                  for h in range(len(per_interval_m))]
        monkeypatch.setattr(ndfd, "_load_grid",
                            lambda attr: {"tree": tree, "frames": frames})
        return start

    def test_accumulation_is_monotonic_nondecreasing(self, monkeypatch):
        # Per-interval snow (m): rises and falls hour to hour.
        per = [0.0, 0.01, 0.03, 0.0, 0.02, 0.05, 0.0, 0.0]
        start = self._stub_grid(monkeypatch, per)
        area = _StubArea(n=1)
        fit = ndfd.fit_area(area, "snowfall-amount", accumulate=True)
        assert fit is not None
        spline = fit["splines"][0]
        # Sample at 5-min resolution across the window; never decreases.
        import numpy as np
        ts = np.arange(fit["start"], fit["end"], 300, dtype=float)
        ys = spline(ts)
        assert np.all(np.diff(ys) >= -1e-9), "cumulative total must not decrease"

    def test_accumulation_starts_at_zero_and_totals_sum(self, monkeypatch):
        per = [0.0, 0.01, 0.03, 0.0, 0.02, 0.05, 0.0, 0.0]
        start = self._stub_grid(monkeypatch, per)
        fit = ndfd.fit_area(_StubArea(n=1), "snowfall-amount", accumulate=True)
        spline = fit["splines"][0]
        conv = ndfd.ELEMENTS["snowfall-amount"]["convert"]  # m -> in
        # Starts at ~0 at the window start.
        assert float(spline(fit["start"])) == pytest.approx(0.0, abs=1e-6)
        # Ends at the full sum of per-interval amounts (converted to inches).
        expected_total = conv(sum(per))
        assert float(spline(fit["end"])) == pytest.approx(expected_total, abs=1e-4)

    def test_observation_mode_is_per_interval_not_cumulative(self, monkeypatch):
        # Default (accumulate=False) splines the raw per-interval values, so a
        # falling interval DOES produce a lower value (the old behavior, correct
        # for a "rate"/observation layer).
        per = [0.0, 0.05, 0.0, 0.0]
        self._stub_grid(monkeypatch, per)
        fit = ndfd.fit_area(_StubArea(n=1), "snowfall-amount", accumulate=False)
        spline = fit["splines"][0]
        conv = ndfd.ELEMENTS["snowfall-amount"]["convert"]
        # At the 2nd hour the per-interval value is the peak; by the 3rd it
        # drops back toward 0 — i.e. NOT monotonic.
        assert float(spline(fit["start"] + 3600)) == pytest.approx(conv(0.05), abs=1e-4)
        assert float(spline(fit["start"] + 2 * 3600)) < float(spline(fit["start"] + 3600))

    def test_accumulation_ensemble_requests_accumulate(self, monkeypatch):
        # NDFDAccumulationEnsemble must drive fit_area with accumulate=True.
        from weather_radar.lib.models.ndfd_model import NDFDAccumulationEnsemble
        captured = {}

        def fake_fit(area, attr, accumulate=False):
            captured["accumulate"] = accumulate
            return None  # no data -> ensemble stays empty; we only check the flag

        monkeypatch.setattr(ndfd, "fit_area", fake_fit)
        monkeypatch.setattr(ndfd, "grid_mtime", lambda attr: 1.0)
        ens = NDFDAccumulationEnsemble(_StubArea(n=1), "snowfall-amount")
        list(ens.iter_points())  # triggers _ensure_fit
        assert captured.get("accumulate") is True
