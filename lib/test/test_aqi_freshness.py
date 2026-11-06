"""Tests for NAQFC (AQI) GRIB freshness / re-fetch behavior.

Regression: NAQFC publishes only twice daily (06z/12z) but its HTTP Expires is
only ~4h out. In the multi-hour window where Expires has passed but the SAME
cycle is still newest, ensure_grid must NOT re-download the identical ~68MB
GRIB on every request — it should extend the local freshness stamp and reuse
the file, only downloading when a genuinely newer cycle appears.
"""
import os
import time
import tempfile

from weather_radar.lib import aqi


class TestEnsureGridNoRefetch:
    def _setup(self, monkeypatch):
        d = tempfile.mkdtemp()
        monkeypatch.setattr(aqi, "NAQFC_DIR", d)
        monkeypatch.setattr(aqi, "NAQFC_RECHECK_SECONDS", 1800)
        aqi._CURRENT_PATH.clear()
        return d

    def test_same_cycle_expired_does_not_redownload(self, monkeypatch):
        self._setup(monkeypatch)
        day, cyc, pol = "20261105", "12", "pm25"
        path = aqi._grib_path(day, cyc, pol)
        with open(path, "wb") as f:
            f.write(b"fake grib")
        # Expires stamped 3h in the past (mtime past) -> not fresh.
        past = time.time() - 3 * 3600
        os.utime(path, (past, past))

        downloads = {"n": 0}
        monkeypatch.setattr(aqi, "_latest_cycle", lambda: (day, cyc))
        monkeypatch.setattr(aqi, "_download_grib",
                            lambda *a, **k: downloads.__setitem__("n", downloads["n"] + 1) or path)
        monkeypatch.setattr(aqi, "_load_grid",
                            lambda p, pol: {"tree": object(), "frames": [1],
                                            "mtime": os.path.getmtime(p)})

        aqi.ensure_grid(pol)
        # Same cycle already on disk -> no re-download.
        assert downloads["n"] == 0
        # Freshness extended into the future.
        assert os.path.getmtime(path) > time.time()

    def test_second_call_uses_fast_path_no_reprobe(self, monkeypatch):
        self._setup(monkeypatch)
        day, cyc, pol = "20261105", "12", "pm25"
        path = aqi._grib_path(day, cyc, pol)
        with open(path, "wb") as f:
            f.write(b"fake grib")
        os.utime(path, (time.time() - 3 * 3600,) * 2)

        monkeypatch.setattr(aqi, "_latest_cycle", lambda: (day, cyc))
        monkeypatch.setattr(aqi, "_download_grib", lambda *a, **k: path)
        monkeypatch.setattr(aqi, "_load_grid",
                            lambda p, pol: {"tree": object(), "frames": [1],
                                            "mtime": os.path.getmtime(p)})
        aqi.ensure_grid(pol)  # first call extends freshness

        # Second call must hit the fast path (fresh) and NOT re-probe NOMADS.
        def _boom():
            raise AssertionError("re-probed NOMADS on a fresh file")
        monkeypatch.setattr(aqi, "_latest_cycle", _boom)
        aqi.ensure_grid(pol)  # should not raise

    def test_newer_cycle_downloads(self, monkeypatch):
        self._setup(monkeypatch)
        pol = "pm25"
        # We have yesterday's 12z on disk (expired)...
        old = aqi._grib_path("20261104", "12", pol)
        with open(old, "wb") as f:
            f.write(b"old")
        os.utime(old, (time.time() - 3 * 3600,) * 2)
        aqi._CURRENT_PATH[pol] = old

        # ...and NOMADS now advertises today's 12z (a DIFFERENT path).
        downloads = {"n": 0}
        new_path = aqi._grib_path("20261105", "12", pol)
        monkeypatch.setattr(aqi, "_latest_cycle", lambda: ("20261105", "12"))

        def fake_dl(day, c, pollutant, force=False):
            downloads["n"] += 1
            with open(new_path, "wb") as f:
                f.write(b"new")
            return new_path
        monkeypatch.setattr(aqi, "_download_grib", fake_dl)
        monkeypatch.setattr(aqi, "_load_grid",
                            lambda p, pol: {"tree": object(), "frames": [1],
                                            "mtime": os.path.getmtime(p)})

        aqi.ensure_grid(pol)
        # A newer cycle = a new path not on disk -> downloads.
        assert downloads["n"] == 1
        assert aqi._CURRENT_PATH[pol] == new_path
