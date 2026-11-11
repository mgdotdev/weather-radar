"""Tests for the NAQFC (AQI) NowCast variant.

NowCast is the EPA-weighted trailing-hour average that AirNow / Apple Weather
report, as opposed to the raw instantaneous hourly AQI. These cover:
  * pollutant-id parsing ("pm25-nowcast" -> base + flag),
  * the NowCast concentration algorithm (weights, data-sufficiency rule),
  * prior-cycle resolution + archive URL layout (for trailing-hour backfill),
  * fit_area applying NowCast end-to-end against a stubbed grid, including the
    served window being the CURRENT cycle only (prior hours feed the average,
    they are not served as forecast knots).
"""
import numpy as np

from weather_radar.lib import aqi


# --- pollutant id parsing ---------------------------------------------------

class TestParsePollutant:
    def test_base(self):
        assert aqi._parse_pollutant("pm25") == ("pm25", False)
        assert aqi._parse_pollutant("ozone") == ("ozone", False)

    def test_nowcast(self):
        assert aqi._parse_pollutant("pm25-nowcast") == ("pm25", True)
        assert aqi._parse_pollutant("ozone-nowcast") == ("ozone", True)

    def test_case_and_whitespace(self):
        assert aqi._parse_pollutant("  PM25-NowCast ") == ("pm25", True)

    def test_unknown_passthrough(self):
        assert aqi._parse_pollutant("bogus") == ("bogus", False)


# --- NowCast concentration algorithm ----------------------------------------

class TestNowCastConcentration:
    def test_constant_series_returns_that_value(self):
        # No variability -> weight clamps to 1 -> plain mean of equal values.
        nc = aqi._nowcast_concentration([10.0] * 12)
        assert abs(nc - 10.0) < 1e-9

    def test_recent_hours_weighted_more(self):
        # Most-recent-first: a high recent value with low older values should
        # pull NowCast ABOVE the simple mean (recent hours weigh more).
        series = [20.0] + [5.0] * 11
        nc = aqi._nowcast_concentration(series)
        simple_mean = (20.0 + 5.0 * 11) / 12
        assert nc > simple_mean

    def test_damps_relative_to_instantaneous_spike(self):
        # The whole point: NowCast of a recent spike is LOWER than the raw
        # most-recent hour (it's averaged with the calmer trailing hours).
        series = [50.0] + [5.0] * 11
        nc = aqi._nowcast_concentration(series)
        assert nc < 50.0

    def test_insufficient_recent_data_returns_none(self):
        # Fewer than 2 of the 3 most-recent hours present -> undefined.
        assert aqi._nowcast_concentration([None, None, 5.0, 5.0]) is None

    def test_missing_hours_skipped(self):
        # Gaps in older hours are fine as long as the recency rule is met.
        nc = aqi._nowcast_concentration([10.0, 10.0, None, 10.0, None])
        assert nc is not None and nc > 0

    def test_fill_values_ignored(self):
        # 9999 is the NAQFC fill/no-data sentinel; treated as missing.
        assert aqi._nowcast_concentration([9999.0, 9999.0, 5.0]) is None


# --- prior-cycle resolution + archive layout --------------------------------

class TestPrevCycle:
    def test_12z_predecessor_is_same_day_06z(self):
        assert aqi._prev_cycle("20261105", "12") == ("20261105", "06")

    def test_06z_predecessor_is_prior_day_12z(self):
        assert aqi._prev_cycle("20261105", "06") == ("20261104", "12")

    def test_month_boundary(self):
        assert aqi._prev_cycle("20261101", "06") == ("20261031", "12")


class TestArchiveUrl:
    def test_url_layout(self, monkeypatch):
        monkeypatch.setattr(aqi, "NAQFC_ARCHIVE_BASE", "https://x")
        monkeypatch.setattr(aqi, "NAQFC_ARCHIVE_VERSION", "AQMv7")
        monkeypatch.setattr(aqi, "NAQFC_ARCHIVE_SECTOR", "CS")
        url = aqi._archive_url("20261105", "06", "pm25")
        assert url == (
            "https://x/AQMv7/CS/20261105/06/"
            "aqm.t06z.ave_1hr_pm25_bc.20261105.227.grib2")


# --- fit_area end-to-end with a stubbed grid --------------------------------

def _stub_grid(monkeypatch, cur_frames, prior_frames=None):
    """Install a fake current grid (and optional prior grid) so fit_area runs
    without eccodes/network. A trivial KDTree stub maps any query to cell 0."""
    class _Tree:
        def query(self, pts, k=1):
            n = len(pts)
            return np.zeros(n), np.zeros(n, dtype=int)

    cur = {"size": 1, "tree": _Tree(), "frames": cur_frames}
    monkeypatch.setattr(aqi, "ensure_grid", lambda pollutant="pm25": cur)
    monkeypatch.setattr(
        aqi, "_prior_frames",
        lambda pollutant: ({"size": 1, "tree": _Tree(), "frames": prior_frames}
                           if prior_frames else None))
    return cur


class _Coord:
    def __init__(self, lat, lon):
        self.lat, self.lon = lat, lon


class _Area:
    def __init__(self):
        self.map_coordinates = [_Coord(45.0, -118.0)]


class TestFitAreaNowCast:
    def test_window_is_current_cycle_only(self, monkeypatch):
        # Prior cycle hours precede the current window; they must not widen it.
        h = 3600
        t0 = 1_700_000_000
        prior = [(t0 - (12 - i) * h, np.array([5.0])) for i in range(12)]
        cur = [(t0 + i * h, np.array([5.0])) for i in range(24)]
        _stub_grid(monkeypatch, cur, prior)

        fit = aqi.fit_area(_Area(), "pm25-nowcast")
        assert fit is not None
        assert fit["start"] == t0
        assert fit["end"] == t0 + 23 * h

    def test_nowcast_lower_than_raw_on_spike(self, monkeypatch):
        # Current cycle: a sharp spike at the first served hour, calm before it
        # (supplied by the prior cycle). Raw AQI uses the spike directly;
        # NowCast averages it down. Compare the fitted value AT t0.
        h = 3600
        t0 = 1_700_000_000
        prior = [(t0 - (12 - i) * h, np.array([3.0])) for i in range(12)]
        # First current hour spikes to a high PM2.5, then returns to calm.
        cur_vals = [120.0] + [3.0] * 23
        cur = [(t0 + i * h, np.array([v])) for i, v in enumerate(cur_vals)]
        _stub_grid(monkeypatch, cur, prior)

        raw_fit = aqi.fit_area(_Area(), "pm25")
        nc_fit = aqi.fit_area(_Area(), "pm25-nowcast")
        raw_at_t0 = float(raw_fit["splines"][0](t0))
        nc_at_t0 = float(nc_fit["splines"][0](t0))
        # NowCast damps the spike -> strictly lower AQI at the spike hour.
        assert nc_at_t0 < raw_at_t0

    def test_unknown_base_returns_none(self, monkeypatch):
        _stub_grid(monkeypatch, [(1, np.array([5.0]))])
        assert aqi.fit_area(_Area(), "bogus-nowcast") is None


# --- Dense (spatial) AQI fitting: fit_area_dense + shared interp -------------

class _GridArea:
    """A regular n x n lattice area (matches a stubbed multi-cell grid)."""
    def __init__(self, n=3, base_lat=45.0, base_lon=-118.0, step=0.1):
        self._n, self._bl, self._bo, self._st = n, base_lat, base_lon, step

    @property
    def map_coordinates(self):
        return [_Coord(self._bl + i * self._st, self._bo + j * self._st)
                for i in range(self._n) for j in range(self._n)]


def _stub_grid_nxn(monkeypatch, n, n_hours, value_fn):
    """Stub a NAQFC grid over an n x n lattice. The KDTree returns each query
    point's own index (points queried in map_coordinates order == cell order).
    value_fn(cell, h) -> concentration."""
    class _Tree:
        def query(self, pts, k=1):
            m = len(pts)
            return np.zeros(m), np.arange(m, dtype=int)
    t0 = 1_700_000_000
    frames = [(t0 + h * 3600,
               np.array([value_fn(c, h) for c in range(n * n)], dtype=float))
              for h in range(n_hours)]
    cur = {"size": n * n, "tree": _Tree(), "frames": frames}
    monkeypatch.setattr(aqi, "ensure_grid", lambda pollutant="pm25": cur)
    monkeypatch.setattr(aqi, "_prior_frames", lambda pollutant: None)
    return t0


class TestAqiDenseFit:
    def test_upsample_1_passthrough(self, monkeypatch):
        _stub_grid_nxn(monkeypatch, 3, 6, lambda c, h: 10.0 + c + h)
        area = _GridArea(3)
        plain = aqi.fit_area(area, "pm25")
        dense = aqi.fit_area_dense(area, "pm25", upsample=1)
        assert len(dense["points"]) == len(plain["points"]) == 9

    def test_upsample_increases_points(self, monkeypatch):
        _stub_grid_nxn(monkeypatch, 3, 6, lambda c, h: 10.0 + c + h)
        dense = aqi.fit_area_dense(_GridArea(3), "pm25", upsample=2)
        # 3x3 native -> 5x5 dense ((n-1)*k+1 per axis).
        assert len(dense["points"]) == 25

    def test_dense_points_are_lon_lat_pairs(self, monkeypatch):
        _stub_grid_nxn(monkeypatch, 3, 6, lambda c, h: 10.0 + c + h)
        dense = aqi.fit_area_dense(_GridArea(3), "pm25", upsample=2)
        p = dense["points"][0]
        assert isinstance(p, list) and len(p) == 2  # [lon, lat]

    def test_env_var_drives_upsample(self, monkeypatch):
        _stub_grid_nxn(monkeypatch, 3, 6, lambda c, h: 10.0 + c + h)
        monkeypatch.setenv("WR_AQI_UPSAMPLE", "2")
        assert len(aqi.fit_area_dense(_GridArea(3), "pm25")["points"]) == 25
        monkeypatch.setenv("WR_AQI_UPSAMPLE", "1")
        assert len(aqi.fit_area_dense(_GridArea(3), "pm25")["points"]) == 9


class TestInterpDensify:
    """Direct tests of the shared stage-2 helper (no source coupling)."""

    def test_densify_axis(self):
        from weather_radar.lib import interp
        ax = np.array([0.0, 1.0, 2.0])
        out = interp.densify_axis(ax, 2)
        # (n-1)*k + 1 = 5 points, endpoint-inclusive, strictly increasing.
        assert out.tolist() == [0.0, 0.5, 1.0, 1.5, 2.0]
        assert interp.densify_axis(ax, 1).tolist() == ax.tolist()

    def test_exact_at_native_nodes_linear_field(self):
        # A spatially-linear field is reproduced exactly by cubic interp at the
        # native nodes, so a dense re-fit matches the native spline there.
        from weather_radar.lib import interp
        from scipy.interpolate import CubicSpline
        lats = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        lons = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
        epochs = np.array([0.0, 3600.0, 7200.0])
        # value = 10 + lat + 2*lon, constant in time -> per-point splines.
        native = []
        for la, lo in zip(lats, lons):
            v = 10.0 + la + 2.0 * lo
            native.append(CubicSpline(epochs, np.array([v, v, v])))

        def refit(vals, ep):
            good = np.isfinite(vals)
            return CubicSpline(ep[good], vals[good]) if good.sum() >= 2 else None

        d_lats, d_lons, d_spl = interp.densify_fit(lats, lons, epochs, native,
                                                   refit, 2)
        # Find the dense point at native node (1,1); its value must be 10+1+2=13.
        for la, lo, sp in zip(d_lats, d_lons, d_spl):
            if abs(la - 1.0) < 1e-9 and abs(lo - 1.0) < 1e-9:
                assert float(sp(3600.0)) == __import__("pytest").approx(13.0, abs=1e-6)
                break
        else:
            raise AssertionError("native node (1,1) not in dense lattice")

    def test_holes_are_filled(self):
        # A None native spline (hole) must not leave a dense hole.
        from weather_radar.lib import interp
        from scipy.interpolate import CubicSpline
        lats = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        lons = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
        epochs = np.array([0.0, 3600.0, 7200.0])
        native = []
        for k, (la, lo) in enumerate(zip(lats, lons)):
            if k == 4:              # drop the center point
                native.append(None)
                continue
            v = 10.0 + la + lo
            native.append(CubicSpline(epochs, np.array([v, v, v])))

        def refit(vals, ep):
            good = np.isfinite(vals)
            return CubicSpline(ep[good], vals[good]) if good.sum() >= 2 else None

        d_lats, d_lons, d_spl = interp.densify_fit(lats, lons, epochs, native,
                                                   refit, 2)
        assert len(d_spl) == 25
        assert all(s is not None for s in d_spl)   # no dense holes
