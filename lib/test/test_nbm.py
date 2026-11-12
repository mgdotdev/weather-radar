"""Tests for the NBM (National Blend of Models) GRIB weather source
(lib/nbm.py) and its ensemble adapters (lib/models/nbm_model.py), plus the
WR_WEATHER_SOURCE=nbm selection.

These avoid the network / real GRIB: the pure logic (element mapping, unit
conversions, .idx record-range parsing, URL construction, source selection) is
tested directly, and the ensemble adapter is tested against a monkeypatched
grid so the ObservationEnsemble-contract behavior is verified without eccodes.
"""
import numpy as np
import pytest

from weather_radar.lib import nbm
from weather_radar.lib.area import MapCoordinate
from weather_radar.lib.models.utils import BoundsError


class _StubArea:
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
    def test_supported(self):
        for a in ("temperature", "snowfall-amount", "quantitative-precipitation",
                  "wind-speed", "sky-cover", "visibility"):
            assert nbm.supports(a)

    def test_unsupported(self):
        for a in ("heat-index", "pressure", "not-real"):
            assert not nbm.supports(a)

    def test_kelvin_to_f(self):
        spec = nbm.ELEMENTS["temperature"]
        assert spec["convert"](273.15) == pytest.approx(32.0)
        assert spec["units"] == "degF"

    def test_snow_is_accumulation_meters_to_inches(self):
        spec = nbm.ELEMENTS["snowfall-amount"]
        assert spec["accum"] is True
        assert spec["convert"](0.0254) == pytest.approx(1.0, abs=1e-3)  # 1 inch
        assert spec["units"] == "in"

    def test_qpf_is_accumulation(self):
        assert nbm.ELEMENTS["quantitative-precipitation"]["accum"] is True


# --- URL construction -------------------------------------------------------

class TestUrls:
    def test_file_and_idx_url(self):
        u = nbm._file_url("20261111", "12", 24)
        assert u.endswith("blend.20261111/12/core/blend.t12z.core.f024.co.grib2")
        assert nbm._idx_url("20261111", "12", 24) == u + ".idx"

    def test_region_env(self, monkeypatch):
        monkeypatch.setattr(nbm, "NBM_REGION", "ak")
        assert nbm._file_url("20261111", "12", 1).endswith(".ak.grib2")


# --- .idx deterministic record-range parsing (the core new logic) -----------

class TestIdxRecordRange:
    # A realistic snippet: instantaneous TMP at two levels, and ASNOW/APCP with
    # deterministic 1h + 6h block records PLUS probabilistic/percentile records
    # that MUST be skipped.
    IDX = [
        "1:0:d=2026111112:TMP:surface:24 hour fcst:",
        "2:1000:d=2026111112:TMP:2 m above ground:24 hour fcst:",
        "3:2000:d=2026111112:TMP:2 m above ground:24 hour fcst:ens std dev",
        "4:3000:d=2026111112:ASNOW:surface:23-24 hour acc@(fcst,dt=1 hour),missing=0:50% level",
        "5:4000:d=2026111112:ASNOW:surface:23-24 hour acc fcst:",
        "6:5500:d=2026111112:ASNOW:surface:18-24 hour acc fcst:",
        "7:6000:d=2026111112:ASNOW:surface:18-24 hour acc fcst:prob >0.0254",
        "8:7000:d=2026111112:APCP:surface:23-24 hour acc fcst:",
        "9:9000:d=2026111112:DPT:2 m above ground:24 hour fcst:",
    ]

    def test_instantaneous_picks_right_level(self):
        rng = nbm._find_record_range(self.IDX, "TMP", "2 m above ground")
        assert rng == (1000, 1999)

    def test_accumulation_picks_contiguous_block(self):
        # Want the 6-hour block ending at fhour 24 (18-24), NOT the 1-hour
        # (23-24) sliver — the block tiles the window contiguously.
        rng = nbm._find_record_range(self.IDX, "ASNOW", "surface",
                                     want_interval=6, fhour=24)
        assert rng == (5500, 5999)  # record 6, the 18-24 block

    def test_accumulation_skips_prob_and_percentile(self):
        # The selected block must be the deterministic one (record 6), never the
        # percentile (record 4) or prob (record 7).
        rng = nbm._find_record_range(self.IDX, "ASNOW", "surface",
                                     want_interval=6, fhour=24)
        assert rng[0] == 5500

    def test_accumulation_fallback_longest_ending_at_fhour(self):
        # If the exact 6h block is absent but a record ends at fhour, take the
        # longest ending there. Here only the 1h (23-24) deterministic exists
        # for APCP, so it's chosen as the best cover.
        rng = nbm._find_record_range(self.IDX, "APCP", "surface",
                                     want_interval=6, fhour=24)
        assert rng == (7000, 8999)

    def test_missing_var_returns_none(self):
        assert nbm._find_record_range(self.IDX, "GUST",
                                      "10 m above ground") is None

    def test_last_record_open_ended(self):
        rng = nbm._find_record_range(self.IDX, "DPT", "2 m above ground")
        assert rng == (9000, None)


class TestForecastSchedules:
    def test_accumulation_uses_6h_cadence(self):
        hrs = nbm._forecast_hours(accumulate=True)
        # Strict 6-hour contiguous cadence so blocks tile with no gap/overlap.
        assert hrs[0] == 6
        assert all((b - a) == 6 for a, b in zip(hrs, hrs[1:]))

    def test_instantaneous_is_hourly_early(self):
        hrs = nbm._forecast_hours(accumulate=False)
        assert hrs[:3] == [1, 2, 3]


# --- Grid fit + ensemble contract (stubbed grid, no network/eccodes) --------

def _stub_grid(monkeypatch, n=3, n_hours=6, value_fn=None, accum_const=None):
    from scipy.spatial import cKDTree
    base_lat, base_lon = 45.6, -118.8
    cells = [(base_lat + i * 0.1, base_lon + j * 0.1)
             for i in range(n) for j in range(n)]
    tree = cKDTree(np.array(cells, dtype=float))
    import time as _t
    t0 = int(_t.time()) // 3600 * 3600
    if value_fn is None:
        value_fn = lambda c, h: 280.0 + c + h  # Kelvin-ish ramp
    frames = [(t0 + h * 3600,
               np.array([value_fn(c, h) for c in range(n * n)], dtype=float))
              for h in range(n_hours)]
    grid = {"key": ("20261111", "12", "x"), "tree": tree, "frames": frames}
    monkeypatch.setattr(nbm, "_load_grid", lambda attr: grid)
    monkeypatch.setattr(nbm, "grid_mtime", lambda attr: 1.0)
    return t0


class TestFitArea:
    def test_observation_fit(self, monkeypatch):
        _stub_grid(monkeypatch, n=3)
        fit = nbm.fit_area(_StubArea(3), "temperature")
        assert fit is not None
        assert len(fit["points"]) == 9
        assert fit["units"] == "degF"

    def test_accumulation_monotonic(self, monkeypatch):
        per = [0.0, 0.01, 0.03, 0.0, 0.02, 0.0]  # per-interval snow (m)
        _stub_grid(monkeypatch, n=1, n_hours=len(per),
                   value_fn=lambda c, h: per[h])
        fit = nbm.fit_area(_StubArea(1), "snowfall-amount", accumulate=True)
        assert fit is not None
        spline = fit["splines"][0]
        ts = np.arange(fit["start"], fit["end"], 300, dtype=float)
        assert np.all(np.diff(spline(ts)) >= -1e-9)

    def test_unsupported_returns_none(self, monkeypatch):
        _stub_grid(monkeypatch, n=3)
        assert nbm.fit_area(_StubArea(3), "pressure") is None


class TestDenseFit:
    def test_upsample_2(self, monkeypatch):
        _stub_grid(monkeypatch, n=3)
        dense = nbm.fit_area_dense(_StubArea(3), "temperature", upsample=2)
        assert len(dense["points"]) == 25   # 5x5

    def test_env_var(self, monkeypatch):
        _stub_grid(monkeypatch, n=3)
        monkeypatch.setenv("WR_NBM_UPSAMPLE", "2")
        assert len(nbm.fit_area_dense(_StubArea(3), "temperature")["points"]) == 25
        monkeypatch.setenv("WR_NBM_UPSAMPLE", "1")
        assert len(nbm.fit_area_dense(_StubArea(3), "temperature")["points"]) == 9


class TestEnsemble:
    def test_observation_ensemble(self, monkeypatch):
        from weather_radar.lib.models.nbm_model import NBMObservationEnsemble
        _stub_grid(monkeypatch, n=3)
        ens = NBMObservationEnsemble(_StubArea(3), "temperature")
        pts = list(ens.iter_points())
        assert len(pts) == 9
        rng = ens.value_range()
        assert rng and rng["uom"] == "degF"

    def test_accumulation_ensemble_requests_accumulate(self, monkeypatch):
        from weather_radar.lib.models.nbm_model import NBMAccumulationEnsemble
        captured = {}

        def fake(area, attr, accumulate=False, upsample=None):
            captured["accumulate"] = accumulate
            return None

        monkeypatch.setattr(nbm, "fit_area_dense", fake)
        monkeypatch.setattr(nbm, "grid_mtime", lambda attr: 1.0)
        ens = NBMAccumulationEnsemble(_StubArea(1), "snowfall-amount")
        list(ens.iter_points())
        assert captured.get("accumulate") is True


class TestSourceSelection:
    def test_nbm_selected(self, monkeypatch):
        from weather_radar.lib.models import weather_source, ensemble_kinds
        from weather_radar.lib.models.nbm_model import (
            NBMObservationEnsemble, NBMAccumulationEnsemble)
        monkeypatch.setenv("WR_WEATHER_SOURCE", "nbm")
        assert weather_source() == "nbm"
        kinds = ensemble_kinds()
        assert kinds["observation"] is NBMObservationEnsemble
        assert kinds["accumulation"] is NBMAccumulationEnsemble
