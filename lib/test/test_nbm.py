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
    def test_accumulation_mixed_cadence_tiles_window(self):
        hrs = nbm._forecast_hours(accumulate=True)
        # Default (fallback) schedule: 1-hour blocks through ACCUM_HOURLY_MAX,
        # then 6-hour blocks out to NBM's last hour (f264).
        assert hrs[0] == 1
        assert hrs[-1] == 264
        # The per-fhour block lengths must tile [0,264] with NO gap/overlap:
        # each block is (fhour - interval, fhour); consecutive blocks abut.
        blocks = [(fh - nbm._accum_interval_for(fh), fh) for fh in hrs]
        prev = 0
        for a, b in blocks:
            assert a == prev, f"gap/overlap: block {a}-{b} after {prev}"
            assert b > a
            prev = b
        assert prev == 264
        # 1-hour early, 6-hour late, with the switch at ACCUM_HOURLY_MAX.
        assert nbm._accum_interval_for(1) == 1
        assert nbm._accum_interval_for(nbm.ACCUM_HOURLY_MAX) == 1
        assert nbm._accum_interval_for(nbm.ACCUM_HOURLY_MAX + nbm.ACCUM_CADENCE) == 6

    def test_instantaneous_default_is_every_published_hour(self):
        hrs = nbm._forecast_hours(accumulate=False)
        assert hrs[:3] == [1, 2, 3]
        assert len(hrs) == 108 and hrs[-1] == 264
        assert 49 not in hrs and 51 in hrs and 195 not in hrs and 198 in hrs


class TestScheduleFromPublishedHours:
    """The schedules are derived from the hours actually published."""

    def test_instantaneous_takes_every_published_hour(self):
        assert nbm._instant_hours([3, 1, 2, 6, 6, 9]) == [1, 2, 3, 6, 9]

    def test_accumulation_follows_the_hourly_run_then_6h(self):
        # A product that runs hourly only to f36, then 3-hourly.
        hours = list(range(1, 37)) + list(range(39, 91, 3))
        sched = nbm._accum_schedule(hours)
        assert sched[:36] == [(h, 1) for h in range(1, 37)]
        assert sched[36:] == [(h, 6) for h in range(42, 91, 6)]

    def test_accumulation_stops_at_a_gap(self):
        hours = [h for h in nbm._DEFAULT_HOURS if h != 66]
        sched = nbm._accum_schedule(hours)
        assert sched[-1] == (60, 6)          # 60-66 block unavailable -> stop

    def test_accumulation_with_no_hourly_run(self):
        assert nbm._accum_schedule([6, 12, 18, 21]) == [(6, 6), (12, 6), (18, 6)]

    def test_empty(self):
        assert nbm._accum_schedule([]) == [] and nbm._instant_hours([]) == []


def _s3_page(keys, token=None):
    xml = "<ListBucketResult>" + "".join(f"<Contents><Key>{k}</Key></Contents>" for k in keys)
    if token:
        xml += f"<NextContinuationToken>{token}</NextContinuationToken>"
    return xml + "</ListBucketResult>"


class TestPublishedHours:
    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch):
        monkeypatch.setattr(nbm, "_HOURS_CACHE", {})

    def _fake_requests(self, monkeypatch, pages, ok=True, calls=None):
        import sys, types
        mod = types.ModuleType("requests")

        class R:
            def __init__(self, text):
                self.text, self.ok = text, ok

        def get(url, params=None, timeout=None, headers=None):
            if calls is not None:
                calls.append(dict(params or {}))
            tok = (params or {}).get("continuation-token")
            return R(pages[tok])

        mod.get = get
        monkeypatch.setitem(sys.modules, "requests", mod)

    def test_parses_listing_with_pagination_and_region_filter(self, monkeypatch):
        d, c = "20261119", "12"
        k = lambda f, reg="co", ext=".grib2.idx": f"blend.{d}/{c}/core/blend.t{c}z.core.f{f:03d}.{reg}{ext}"
        calls = []
        self._fake_requests(monkeypatch, {
            None: _s3_page([k(1), k(1, ext=".grib2"), k(2, "ak"), k(3)], token="p&amp;2"),
            "p&2": _s3_page([k(6), k(264)]),
        }, calls=calls)
        assert nbm._published_hours(d, c) == [1, 3, 6, 264]
        assert calls[1]["continuation-token"] == "p&2"      # XML-unescaped

    def test_listing_failure_falls_back_to_default(self, monkeypatch):
        self._fake_requests(monkeypatch, {None: ""}, ok=False)
        assert nbm._published_hours("20261119", "12") == nbm._DEFAULT_HOURS

    def test_empty_listing_falls_back_to_default(self, monkeypatch):
        self._fake_requests(monkeypatch, {None: _s3_page([])})
        assert nbm._published_hours("20261119", "12") == nbm._DEFAULT_HOURS

    def test_cached_within_recheck_window(self, monkeypatch):
        calls = []
        key = "blend.20261119/12/core/blend.t12z.core.f001.co.grib2.idx"
        self._fake_requests(monkeypatch, {None: _s3_page([key])}, calls=calls)
        nbm._published_hours("20261119", "12")
        nbm._published_hours("20261119", "12")
        assert len(calls) == 1

    def test_keeps_last_good_listing_when_refresh_fails(self, monkeypatch):
        monkeypatch.setattr(nbm, "_HOURS_CACHE", {("20261119", "12"): (0.0, [1, 2, 3])})
        self._fake_requests(monkeypatch, {None: ""}, ok=False)
        assert nbm._published_hours("20261119", "12") == [1, 2, 3]


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


class TestLoadGridIndexBuiltOnce:
    """The grid index is identical for every record, so _load_grid must ask
    _parse_record to build it for the FIRST parsed record only (it used to
    build a ~3.7M-point KD-tree for every record and keep just the first)."""

    def test_index_requested_only_for_first_record(self, monkeypatch, tmp_path):
        hours = [1, 2, 3, 4]
        files = {}
        for h in hours:
            p = tmp_path / f"rec{h}.grib2"
            p.write_bytes(b"GRIB")
            files[h] = str(p)

        monkeypatch.setattr(nbm, "_GRID_CACHE", {})
        monkeypatch.setattr(nbm, "latest_cycle", lambda force=False: ("20261111", "12"))
        monkeypatch.setattr(nbm, "_published_hours", lambda day, cc: hours)
        monkeypatch.setattr(
            nbm, "_resolve_record_spec",
            lambda day, cc, fh, var, level, want=None: ("fresh", files[fh]))

        sentinel = object()
        calls = []

        def fake_parse(path, want_index=True):
            fh = int(path.split("rec")[1].split(".")[0])
            calls.append((fh, want_index))
            return (sentinel if want_index else None, 1_000_000 + fh * 3600,
                    np.array([float(fh)]))

        monkeypatch.setattr(nbm, "_parse_record", fake_parse)

        grid = nbm._load_grid("temperature")
        assert grid["tree"] is sentinel
        assert [w for _, w in sorted(calls)] == [True, False, False, False]
        assert len(grid["frames"]) == 4


# --- Box extraction (real pyproj grid index on a small synthetic grid) ------

class _BoxArea:
    """An area that only exposes the box (the block path never needs the
    lat/lon lattice)."""
    def __init__(self, s, n, w, e):
        self.requested_bounds = {"s": s, "n": n, "w": w, "e": e}
        self.time_zone = None

    @property
    def map_coordinates(self):
        raise AssertionError("block path must not build the lattice")


def _box_grid(monkeypatch, nx=40, ny=30, n_hours=6, value_fn=None):
    """A real LambertGridIndex over an nx x ny grid + frames whose value is a
    function of the flat cell index."""
    pytest.importorskip("pyproj")
    from weather_radar.lib.gridindex import LambertGridIndex
    index = LambertGridIndex(nx=nx, ny=ny, dx=2539.703, dy=2539.703,
                             lat1=46.0, lon1=-118.0, lad=25.0, lov=-95.0,
                             latin1=25.0, latin2=25.0, radius=6371200.0)
    import time as _t
    t0 = int(_t.time()) // 3600 * 3600
    if value_fn is None:
        value_fn = lambda flat, h: 280.0 + (flat % 97) * 0.1 + h
    flats = np.arange(nx * ny)
    frames = [(t0 + h * 3600, value_fn(flats, h).astype(float))
              for h in range(n_hours)]
    grid = {"key": ("20261111", "12", "x"), "tree": index, "frames": frames}
    monkeypatch.setattr(nbm, "_load_grid", lambda attr: grid)
    monkeypatch.setattr(nbm, "grid_mtime", lambda attr: 1.0)
    return index, grid


def _center_box(index, half_deg=0.1):
    lat, lon = index.latlon_of(index.ny / 2.0, index.nx / 2.0)   # grid center
    lat, lon = float(lat), float(lon)
    return lat - half_deg, lat + half_deg, lon - half_deg, lon + half_deg


class TestFitAreaBox:
    def test_points_are_the_native_cells_in_the_box(self, monkeypatch):
        index, grid = _box_grid(monkeypatch)
        s, n, w, e = _center_box(index)
        fit = nbm.fit_area(_BoxArea(s, n, w, e), "temperature")
        blk = index.box_cells(s, n, w, e)
        assert fit is not None and "block" in fit
        assert len(fit["points"]) == blk.flat.size
        got = np.array([[p.lat, p.lon] for p in fit["points"]])
        assert np.allclose(got[:, 0], blk.lats) and np.allclose(got[:, 1], blk.lons)
        # One point per cell: no two points share a location.
        assert len({(round(p.lat, 9), round(p.lon, 9)) for p in fit["points"]}) \
            == len(fit["points"])

    def test_values_come_from_the_selected_cells(self, monkeypatch):
        index, grid = _box_grid(monkeypatch)
        s, n, w, e = _center_box(index)
        fit = nbm.fit_area(_BoxArea(s, n, w, e), "temperature")
        blk = index.box_cells(s, n, w, e)
        epoch0, vals0 = grid["frames"][0]
        conv = nbm.ELEMENTS["temperature"]["convert"]
        for i in (0, len(fit["splines"]) // 2, len(fit["splines"]) - 1):
            want = conv(vals0[blk.flat[i]])
            assert float(fit["splines"][i](epoch0)) == pytest.approx(want, abs=1e-6)

    def test_box_outside_grid_returns_none(self, monkeypatch):
        _box_grid(monkeypatch)
        assert nbm.fit_area(_BoxArea(-10, -5, 10, 15), "temperature") is None

    def test_dense_fit_densifies_in_grid_space(self, monkeypatch):
        index, grid = _box_grid(monkeypatch)
        s, n, w, e = _center_box(index)
        k = 3
        dense = nbm.fit_area_dense(_BoxArea(s, n, w, e), "temperature", upsample=k)
        n_r, n_c = index.box_cells(s, n, w, e).shape
        assert len(dense["points"]) == ((n_r - 1) * k + 1) * ((n_c - 1) * k + 1)
        assert "block" not in dense

    def test_dense_fit_skips_cells_without_data(self, monkeypatch):
        fill = nbm._FILL
        # Every cell with flat % 7 == 0 is fill (no data) at every hour.
        index, grid = _box_grid(
            monkeypatch,
            value_fn=lambda flat, h: np.where(flat % 7 == 0, fill, 280.0 + h))
        s, n, w, e = _center_box(index)
        fit = nbm.fit_area(_BoxArea(s, n, w, e), "temperature")
        blk = index.box_cells(s, n, w, e)
        assert len(fit["points"]) == int((blk.flat % 7 != 0).sum())
        dense = nbm.fit_area_dense(_BoxArea(s, n, w, e), "temperature", upsample=2)
        n_r, n_c = blk.shape
        assert 0 < len(dense["points"]) < ((n_r - 1) * 2 + 1) * ((n_c - 1) * 2 + 1)

    def test_box_query_error_falls_back_to_lattice(self, monkeypatch):
        index, grid = _box_grid(monkeypatch)

        class Broken:
            def box_cells(self, *a, **kw):
                raise RuntimeError("boom")
            def query(self, pts, k=1):
                return index.query(pts, k=k)

        grid["tree"] = Broken()
        s, n, w, e = _center_box(index)

        class LatticeArea(_BoxArea):
            @property
            def map_coordinates(self):
                lat = (s + n) / 2; lon = (w + e) / 2
                return [MapCoordinate(lat=lat + i * 0.02, lon=lon + j * 0.02)
                        for i in range(3) for j in range(3)]

        fit = nbm.fit_area(LatticeArea(s, n, w, e), "temperature")
        assert fit is not None and "block" not in fit
        assert len(fit["points"]) == 9

    def test_fit_does_not_stack_whole_frames(self, monkeypatch):
        """Only the selected cells are gathered from each frame."""
        index, grid = _box_grid(monkeypatch)
        stacked_shapes = []
        real_vstack = np.vstack

        def spy(arrays, *a, **kw):
            out = real_vstack(arrays, *a, **kw)
            stacked_shapes.append(out.shape)
            return out

        monkeypatch.setattr(np, "vstack", spy)
        s, n, w, e = _center_box(index)
        nbm.fit_area(_BoxArea(s, n, w, e), "temperature")
        n_cells = index.box_cells(s, n, w, e).flat.size
        assert stacked_shapes and all(sh[1] == n_cells for sh in stacked_shapes)


# --- Decoding: alternating-row scanning --------------------------------------

def _fake_eccodes_module(stored, keys, lats=None, lons=None, n_messages=1):
    """A stand-in for the eccodes module serving `n_messages` identical
    messages with the given stored (scan-ordered) values and keys."""
    import types
    mod = types.ModuleType("eccodes")
    state = {"left": n_messages}

    def new(f):
        if state["left"] <= 0:
            return None
        state["left"] -= 1
        return object()

    mod.codes_grib_new_from_file = new
    mod.codes_get = lambda gid, k: keys[k]
    mod.codes_get_values = lambda gid: np.array(stored, dtype=float)
    mod.codes_get_array = lambda gid, k: np.array(lats if k == "latitudes" else lons)
    mod.codes_release = lambda gid: None
    return mod


NBM_SCAN_KEYS = {"alternativeRowScanning": 1, "Nx": 4, "Ny": 3,
                 "jPointsAreConsecutive": 0,
                 "validityDate": 20261121, "validityTime": 1800}


class TestParseRecordScanOrder:
    def test_values_put_in_uniform_layout(self, monkeypatch, tmp_path):
        import sys
        truth = np.arange(12, dtype=float).reshape(3, 4)
        stored = truth.copy(); stored[1::2] = stored[1::2, ::-1]
        monkeypatch.setitem(sys.modules, "eccodes",
                            _fake_eccodes_module(stored.ravel(), NBM_SCAN_KEYS))
        p = tmp_path / "rec.grib2"; p.write_bytes(b"GRIB")
        _, epoch, vals = nbm._parse_record(str(p), want_index=False)
        assert np.array_equal(vals, truth.ravel())
        assert epoch == 1795284000      # 2026-11-21 18:00Z

    def test_flag_off_values_unchanged(self, monkeypatch, tmp_path):
        import sys
        stored = np.arange(12, dtype=float)
        keys = {**NBM_SCAN_KEYS, "alternativeRowScanning": 0}
        monkeypatch.setitem(sys.modules, "eccodes", _fake_eccodes_module(stored, keys))
        p = tmp_path / "rec.grib2"; p.write_bytes(b"GRIB")
        _, _, vals = nbm._parse_record(str(p), want_index=False)
        assert np.array_equal(vals, stored)


# --- Fetch plan + knots -------------------------------------------------------

class TestLoadGridPlan:
    """_load_grid fetches every published hour (instantaneous) or the finest
    gap-free block tiling (accumulation), and rebuilds when more hours land."""

    def _setup(self, monkeypatch, tmp_path, published):
        files = {}
        requested = []

        def resolve(day, cc, fh, var, level, want=None):
            requested.append((fh, want))
            p = tmp_path / f"rec{fh}.grib2"
            p.write_bytes(b"GRIB")
            files[str(p)] = fh
            return ("fresh", str(p))

        def parse(path, want_index=True):
            fh = files[path]
            return (object() if want_index else None,
                    1_000_000 + fh * 3600, np.array([float(fh)]))

        state = {"hours": published}
        monkeypatch.setattr(nbm, "_GRID_CACHE", {})
        monkeypatch.setattr(nbm, "latest_cycle", lambda force=False: ("20261111", "12"))
        monkeypatch.setattr(nbm, "_published_hours", lambda day, cc: state["hours"])
        monkeypatch.setattr(nbm, "_resolve_record_spec", resolve)
        monkeypatch.setattr(nbm, "_parse_record", parse)
        return requested, state

    def test_instantaneous_requests_every_published_hour(self, monkeypatch, tmp_path):
        req, _ = self._setup(monkeypatch, tmp_path, nbm._DEFAULT_HOURS)
        grid = nbm._load_grid("temperature")
        assert sorted(fh for fh, _ in req) == nbm._DEFAULT_HOURS
        assert all(w is None for _, w in req)
        assert len(grid["frames"]) == 108

    def test_accumulation_requests_block_tiling(self, monkeypatch, tmp_path):
        req, _ = self._setup(monkeypatch, tmp_path, nbm._DEFAULT_HOURS)
        nbm._load_grid("snowfall-amount")
        assert sorted(req) == nbm._accum_schedule(nbm._DEFAULT_HOURS)
        assert max(fh for fh, _ in req) == 264

    def test_rebuilds_when_more_hours_are_published(self, monkeypatch, tmp_path):
        req, state = self._setup(monkeypatch, tmp_path, list(range(1, 25)))
        first = nbm._load_grid("temperature")
        assert len(first["frames"]) == 24
        assert nbm._load_grid("temperature") is first            # cached
        state["hours"] = list(range(1, 49))                      # upload progressed
        second = nbm._load_grid("temperature")
        assert second is not first and len(second["frames"]) == 48

    def test_grid_records_cycle_init(self, monkeypatch, tmp_path):
        from datetime import datetime, timezone
        self._setup(monkeypatch, tmp_path, [1, 2, 3])
        grid = nbm._load_grid("temperature")
        want = int(datetime(2026, 11, 11, 12, tzinfo=timezone.utc).timestamp())
        assert grid["init"] == want


class TestKnots:
    def test_observation_knots_are_the_frame_epochs(self, monkeypatch):
        t0 = _stub_grid(monkeypatch, n=3, n_hours=5)
        fit = nbm.fit_area(_StubArea(3), "temperature")
        assert list(fit["knots"]) == [t0 + h * 3600 for h in range(5)]

    def test_accumulation_anchors_at_cycle_init(self, monkeypatch):
        t0 = _stub_grid(monkeypatch, n=1, n_hours=4, value_fn=lambda c, h: 0.01)
        grid = nbm._load_grid("snowfall-amount")
        grid["init"] = t0 - 5 * 3600                    # init 5 h before 1st frame
        fit = nbm.fit_area(_StubArea(1), "snowfall-amount", accumulate=True)
        assert fit["start"] == t0 - 5 * 3600
        assert fit["knots"][0] == t0 - 5 * 3600
        assert float(fit["splines"][0](fit["start"])) == pytest.approx(0.0)

    def test_dense_refit_uses_native_knots_not_a_capped_grid(self, monkeypatch):
        # 300 hourly frames: the old evenly spaced grid would cap at 200 knots.
        # A 4x4 grid keeps the spatial step on the (exact-at-nodes) cubic path.
        n_hours, n = 300, 4
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 3, n_hours)
        _stub_grid(monkeypatch, n=n, n_hours=n_hours,
                   value_fn=lambda c, h: 280.0 + c + noise[h])
        fit = nbm.fit_area(_StubArea(n), "temperature")
        dense = nbm.fit_area_dense(_StubArea(n), "temperature", upsample=2)
        # A dense point that coincides with a native cell reproduces the native
        # series at EVERY forecast hour (exact at all 300 knots).
        native = {(round(p.lat, 6), round(p.lon, 6)): s
                  for p, s in zip(fit["points"], fit["splines"])}
        hit = 0
        for p, s in zip(dense["points"], dense["splines"]):
            key = (round(p.lat, 6), round(p.lon, 6))
            if key in native:
                hit += 1
                k = fit["knots"]
                assert np.allclose(s(k), native[key](k), atol=1e-6)
        assert hit == n * n


# --- Transient vs genuine failures -------------------------------------------

class _FakeResp:
    def __init__(self, status, content=b""):
        self.status_code, self.content = status, content


class _FakeSession:
    """Serves a scripted sequence: an int status, bytes (200), or an exception."""
    def __init__(self, script):
        self.script, self.calls = list(script), 0

    def get(self, url, headers=None, timeout=None):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, bytes):
            return _FakeResp(200, item)
        return _FakeResp(item)


class TestHttpFetch:
    @pytest.fixture(autouse=True)
    def _no_sleep(self, monkeypatch):
        monkeypatch.setattr(nbm.time, "sleep", lambda s: None)

    def _use(self, monkeypatch, script):
        s = _FakeSession(script)
        monkeypatch.setattr(nbm, "_session", lambda: s)
        return s

    def test_retries_transient_errors_then_succeeds(self, monkeypatch):
        s = self._use(monkeypatch, [ConnectionError("dns"), ConnectionError("dns"), b"ok"])
        assert nbm._http_fetch("u") == ("ok", b"ok")
        assert s.calls == 3

    def test_gives_up_after_attempts(self, monkeypatch):
        s = self._use(monkeypatch, [ConnectionError("dns")] * nbm._HTTP_ATTEMPTS)
        assert nbm._http_fetch("u") == ("error", None)
        assert s.calls == nbm._HTTP_ATTEMPTS

    def test_not_found_is_missing_without_retry(self, monkeypatch):
        s = self._use(monkeypatch, [404])
        assert nbm._http_fetch("u") == ("missing", None)
        assert s.calls == 1

    def test_server_error_is_retried(self, monkeypatch):
        s = self._use(monkeypatch, [503, b"x"])
        assert nbm._http_fetch("u") == ("ok", b"x")
        assert s.calls == 2

    def test_other_client_error_not_retried(self, monkeypatch):
        s = self._use(monkeypatch, [416])
        assert nbm._http_fetch("u") == ("error", None)
        assert s.calls == 1

    def test_session_is_shared(self, monkeypatch):
        monkeypatch.setattr(nbm, "_HTTP_SESSION", None)
        assert nbm._session() is nbm._session()


class TestTransientIdxNotNegativeCached:
    @pytest.fixture(autouse=True)
    def _isolate(self, monkeypatch, tmp_path):
        monkeypatch.setattr(nbm, "_MISSING", {})
        monkeypatch.setattr(nbm, "_IDX_CACHE", {})
        monkeypatch.setattr(nbm, "NBM_DIR", str(tmp_path))

    def test_transient_idx_failure_is_retry_and_not_marked_missing(self, monkeypatch):
        monkeypatch.setattr(nbm, "_http_fetch", lambda url, rng=None, timeout=60: ("error", None))
        kind, _ = nbm._resolve_record_spec("20261119", "12", 90, "TMP", "2 m above ground")
        assert kind == "retry"
        assert not nbm._missing_recent(nbm._miss_key("20261119", "12", 90, "TMP"))

    def test_missing_idx_is_miss_and_marked_missing(self, monkeypatch):
        monkeypatch.setattr(nbm, "_http_fetch", lambda url, rng=None, timeout=60: ("missing", None))
        kind, _ = nbm._resolve_record_spec("20261119", "12", 90, "TMP", "2 m above ground")
        assert kind == "miss"
        assert nbm._missing_recent(nbm._miss_key("20261119", "12", 90, "TMP"))

    def test_transient_idx_failure_is_not_cached(self, monkeypatch):
        calls = []

        def fetch(url, rng=None, timeout=60):
            calls.append(url)
            return ("error", None) if len(calls) == 1 else ("ok", b"1:0:d=x:TMP:2 m above ground:90 hour fcst:\n")

        monkeypatch.setattr(nbm, "_http_fetch", fetch)
        assert nbm._fetch_idx_lines("20261119", "12", 90) is nbm._TRANSIENT
        assert nbm._fetch_idx_lines("20261119", "12", 90)[0].startswith("1:0:")


class TestIncompleteGridRetried:
    def _setup(self, monkeypatch, tmp_path, transient_hours):
        files = {}

        def resolve(day, cc, fh, var, level, want=None):
            if fh in state["transient"]:
                return ("retry", None)
            p = tmp_path / f"rec{fh}.grib2"
            p.write_bytes(b"GRIB")
            files[str(p)] = fh
            return ("fresh", str(p))

        def parse(path, want_index=True):
            fh = files[path]
            return (object() if want_index else None, 1_000_000 + fh * 3600,
                    np.array([float(fh)]))

        state = {"transient": set(transient_hours)}
        monkeypatch.setattr(nbm, "_GRID_CACHE", {})
        monkeypatch.setattr(nbm, "latest_cycle", lambda force=False: ("20261111", "12"))
        monkeypatch.setattr(nbm, "_published_hours", lambda day, cc: [1, 2, 3, 4])
        monkeypatch.setattr(nbm, "_resolve_record_spec", resolve)
        monkeypatch.setattr(nbm, "_parse_record", parse)
        return state

    def test_incomplete_grid_rebuilt_after_retry_window(self, monkeypatch, tmp_path):
        state = self._setup(monkeypatch, tmp_path, transient_hours={3})
        first = nbm._load_grid("temperature")
        assert len(first["frames"]) == 3 and first["retry_at"] is not None
        assert nbm._load_grid("temperature") is first        # within the window
        state["transient"].clear()                             # network recovered
        first["retry_at"] = 0.0                                # window elapsed
        second = nbm._load_grid("temperature")
        assert second is not first and len(second["frames"]) == 4
        assert second["retry_at"] is None
        assert nbm._load_grid("temperature") is second       # complete: cached

    def test_complete_grid_has_no_retry(self, monkeypatch, tmp_path):
        self._setup(monkeypatch, tmp_path, transient_hours=set())
        assert nbm._load_grid("temperature")["retry_at"] is None
