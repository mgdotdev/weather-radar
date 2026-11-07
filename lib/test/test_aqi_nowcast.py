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
