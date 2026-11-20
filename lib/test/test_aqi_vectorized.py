"""The vectorized AQI conversion / NowCast must agree with the scalar reference
implementations (which are the EPA algorithm as originally written)."""
import math

import numpy as np
import pytest

from weather_radar.lib import aqi


def _scalar(fn, c):
    out = []
    for x in np.asarray(c, dtype=float).ravel():
        try:
            r = fn(float(x))
        except (ValueError, OverflowError):      # e.g. floor(nan)
            r = None
        out.append(np.nan if r is None else float(r))
    return np.array(out).reshape(np.shape(c))


def _same(a, b):
    return np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(
        a[~np.isnan(a)], b[~np.isnan(b)], rtol=0, atol=0)


class TestConverters:
    PM_EDGES = [0, 0.04, 0.05, 8.95, 9.0, 9.04, 9.05, 9.1, 35.4, 35.45, 35.5,
                55.4, 55.5, 125.4, 125.5, 225.4, 225.5, 325.4, 325.5, 900.0,
                9998.9, 9999, 12345, -0.1, -5]
    OZ_EDGES = [0, 0.9, 1, 53.99, 54, 54.5, 55, 70, 70.9, 71, 85, 86, 105, 105.9,
                106, 200, 200.5, 250, 9998, 9999, -1]

    def test_pm25_boundaries_and_rounding(self):
        c = np.array(self.PM_EDGES)
        assert _same(aqi._aqi_from_pm25_array(c), _scalar(aqi._aqi_from_pm25, c))

    def test_ozone_boundaries(self):
        c = np.array(self.OZ_EDGES)
        assert _same(aqi._aqi_from_ozone_array(c), _scalar(aqi._aqi_from_ozone, c))

    @pytest.mark.parametrize("name,hi", [("pm25", 400.0), ("ozone", 260.0)])
    def test_random_values_match_exactly(self, name, hi):
        rng = np.random.default_rng(7)
        c = np.concatenate([rng.uniform(-2, hi, 40_000), rng.uniform(0, 12, 20_000)])
        vec = aqi._CONVERTER_ARRAY[name](c)
        assert _same(vec, _scalar(aqi._CONVERTER[name], c))

    def test_nan_and_inf_are_missing_not_errors(self):
        out = aqi._aqi_from_pm25_array(np.array([np.nan, np.inf, -np.inf, 10.0]))
        assert list(np.isnan(out)) == [True, True, True, False]

    def test_shape_is_preserved(self):
        assert aqi._aqi_from_pm25_array(np.zeros((4, 7))).shape == (4, 7)


def _scalar_nowcast(col):
    """Reference: the scalar algorithm applied per hour, like the old fit loop."""
    out = []
    for i in range(len(col)):
        lo = max(0, i - aqi.NOWCAST_HOURS + 1)
        trailing = [None if not np.isfinite(col[j]) else float(col[j])
                    for j in range(i, lo - 1, -1)]
        nc = aqi._nowcast_concentration(trailing)
        out.append(np.nan if nc is None else nc)
    return np.array(out)


class TestNowCastMatrix:
    def _check(self, m, rtol=1e-12):
        vec = aqi._nowcast_matrix(m)
        ref = np.column_stack([_scalar_nowcast(m[:, j]) for j in range(m.shape[1])])
        assert np.array_equal(np.isnan(vec), np.isnan(ref))
        np.testing.assert_allclose(vec[~np.isnan(vec)], ref[~np.isnan(ref)],
                                   rtol=rtol, atol=1e-12)

    def test_random_smooth_series(self):
        rng = np.random.default_rng(1)
        base = rng.uniform(1, 80, (1, 40))
        m = base * (1 + 0.3 * np.sin(np.arange(60))[:, None]) + rng.uniform(0, 5, (60, 40))
        self._check(m)

    def test_missing_and_invalid_hours(self):
        rng = np.random.default_rng(2)
        m = rng.uniform(0, 60, (50, 64))
        m[rng.random(m.shape) < 0.25] = np.nan
        m[rng.random(m.shape) < 0.03] = -3.0          # invalid -> treated as missing
        m[rng.random(m.shape) < 0.02] = 10_000.0      # >= 9999 -> missing
        self._check(m)

    def test_constant_zero_and_spiky_series(self):
        m = np.zeros((30, 3))
        m[:, 1] = 12.0
        m[:, 2] = 3.0
        m[15, 2] = 200.0                               # spike is damped
        self._check(m)
        vec = aqi._nowcast_matrix(m)
        assert vec[20, 0] == 0.0 and abs(vec[20, 1] - 12.0) < 1e-12
        assert vec[15, 2] < 200.0

    def test_first_hours_use_only_available_history(self):
        m = np.array([[5.0], [7.0], [6.0], [20.0]])
        self._check(m)
        assert not np.isnan(aqi._nowcast_matrix(m)[1, 0])   # 2 hours is enough

    def test_data_sufficiency_rule(self):
        # hour 3: only 1 of the 3 most recent hours present -> undefined
        m = np.array([[5.0], [5.0], [np.nan], [np.nan], [5.0], [np.nan], [np.nan], [np.nan]])
        self._check(m)
        assert np.isnan(aqi._nowcast_matrix(m)[3, 0])

    def test_chunking_does_not_change_results(self):
        rng = np.random.default_rng(3)
        m = rng.uniform(0, 50, (40, 700))
        np.testing.assert_array_equal(aqi._nowcast_matrix(m, chunk=64),
                                      aqi._nowcast_matrix(m, chunk=4096))


class TestFitEquivalence:
    """fit_area end-to-end equals the old scalar per-point computation."""

    def test_splines_match_scalar_pipeline(self, monkeypatch):
        from scipy.interpolate import CubicSpline
        T0, H = 1_700_000_000, 3600
        rng = np.random.default_rng(4)
        n_pts, hours = 6, 30
        frames = [(T0 + i * H, rng.uniform(0, 70, 8)) for i in range(hours)]

        class Tree:
            def query(self, pts, k=1):
                return np.zeros(len(pts)), np.arange(len(pts)) % 8
        g = {"size": 1, "tree": Tree(), "frames": frames}
        monkeypatch.setattr(aqi, "ensure_grid", lambda p="pm25": g)
        monkeypatch.setattr(aqi, "_prior_frames", lambda p: None)

        class C:
            def __init__(s, i): s.lat, s.lon = 45 + i * .01, -118.0
        class A:
            map_coordinates = [C(i) for i in range(n_pts)]

        for name in ("pm25", "pm25-nowcast"):
            fit = aqi.fit_area(A(), name)
            base, nowcast = aqi._parse_pollutant(name)
            epochs = np.array([e for e, _ in frames], dtype=float)
            idx = np.arange(n_pts) % 8
            series = np.vstack([v[idx] for _, v in frames])
            xs = np.linspace(epochs[0], epochs[-1], 97)
            for col in range(n_pts):
                raw = series[:, col]
                # Reference = the original scalar per-point pipeline: None for
                # "no value", and those hours are simply not spline knots.
                conc = _scalar_nowcast(raw) if nowcast else raw
                ys = [None if np.isnan(v) else aqi._CONVERTER[base](float(v))
                      for v in conc]
                ok = np.array([y is not None for y in ys])
                ref = CubicSpline(epochs[ok], np.array([y for y in ys if y is not None]))
                np.testing.assert_allclose(fit["splines"][col](xs), ref(xs), atol=1e-9)


def _scalar_predict(fit, epochs):
    """Reference: the original per-spline, per-frame loop."""
    lo, hi = fit["start"], fit["end"]
    for t in epochs:
        vals, any_val = [], False
        for s in fit["splines"]:
            if s is None or not (lo <= t <= hi):
                vals.append(None)
                continue
            y = float(s(t))
            if y < 0:
                y = 0.0
            vals.append(round(y))
            any_val = True
        if any_val:
            yield {"epoch": int(t), "vals": vals}


class TestPredictAreaVectorized:
    def _fit(self, seed=0, n=300):
        from scipy.interpolate import CubicSpline
        rng = np.random.default_rng(seed)
        x_full = 1_700_000_000.0 + 3600.0 * np.arange(30)
        splines = []
        for i in range(n):
            r = rng.random()
            if r < 0.05:
                splines.append(None)                              # no data
            elif r < 0.15:                                        # a different knot set
                keep = np.sort(rng.choice(30, 12, replace=False))
                splines.append(CubicSpline(x_full[keep], rng.uniform(-30, 180, 12)))
            else:
                # includes negative excursions (clamped) and sharp changes
                splines.append(CubicSpline(x_full, rng.uniform(-40, 300, 30)))
        pts = [[-117.0 + i * 0.01, 47.0] for i in range(n)]
        return {"start": int(x_full[0]), "end": int(x_full[-1]),
                "points": pts, "splines": splines}

    def test_bit_identical_to_the_per_spline_loop(self):
        fit = self._fit()
        lo, hi = fit["start"], fit["end"]
        epochs = ([lo, hi, lo - 1, hi + 1]                            # ends + outside
                  + [int(lo + 3600 * k) for k in range(30)]           # exactly at knots
                  + [int(lo + 1234 * k) for k in range(1, 500, 7)]    # between knots
                  + [lo + 1800 + 3600 * k for k in range(29)])        # mid-interval
        got = list(aqi.predict_area(fit, epochs, fixed_grid=True))
        want = list(_scalar_predict(fit, epochs))
        assert [g["epoch"] for g in got] == [w["epoch"] for w in want]
        assert [g["vals"] for g in got] == [w["vals"] for w in want]

    def test_none_splines_stay_none_and_values_are_ints(self):
        fit = self._fit(seed=3)
        fr = next(iter(aqi.predict_area(fit, [fit["start"] + 5000], fixed_grid=True)))
        for s, v in zip(fit["splines"], fr["vals"]):
            assert (v is None) == (s is None)
            assert v is None or (isinstance(v, int) and v >= 0)

    def test_frame_outside_window_or_with_no_splines_is_skipped(self):
        fit = self._fit()
        assert list(aqi.predict_area(fit, [fit["end"] + 10], fixed_grid=True)) == []
        empty = {"start": 0, "end": 10, "points": [[0, 0]], "splines": [None]}
        assert list(aqi.predict_area(empty, [5], fixed_grid=True)) == []

    def test_negative_values_clamp_to_zero(self):
        from scipy.interpolate import CubicSpline
        xs = np.array([0.0, 3600.0, 7200.0])
        fit = {"start": 0, "end": 7200, "points": [[0, 0]],
               "splines": [CubicSpline(xs, [-5.0, -5.0, -5.0])]}
        assert next(iter(aqi.predict_area(fit, [3600], fixed_grid=True)))["vals"] == [0]

    def test_half_values_round_half_even_like_python(self):
        from scipy.interpolate import CubicSpline
        xs = np.array([0.0, 1.0, 2.0, 3.0])
        sp = [CubicSpline(xs, [v] * 4) for v in (0.5, 1.5, 2.5, 3.5)]
        fit = {"start": 0, "end": 3, "points": [[0, 0]] * 4, "splines": sp}
        got = next(iter(aqi.predict_area(fit, [1], fixed_grid=True)))["vals"]
        assert got == [round(0.5), round(1.5), round(2.5), round(3.5)] == [0, 2, 2, 4]

    def test_vector_evaluator_is_built_once_per_fit(self):
        fit = self._fit(n=20)
        list(aqi.predict_area(fit, [fit["start"] + 10], fixed_grid=True))
        first = fit["_vec"]
        list(aqi.predict_area(fit, [fit["start"] + 20], fixed_grid=True))
        assert fit["_vec"] is first
