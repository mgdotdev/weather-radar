from datetime import datetime, timedelta, timezone

import pytest

from weather_radar.lib.area import MapCoordinate, CoordinateArea, NOAAGridpoint
from weather_radar.lib.models.accumulation import AccumulationEnsemble, AccumulationModel


class TestAccumulationModel:
    def test_precipitation_ensemble(self):
        coord = MapCoordinate(47.6763, -116.7798)
        area = CoordinateArea(coord, 40, 10)
        grid = AccumulationEnsemble(area, "quantitative-precipitation").predict(datetime.now(), verbose=True)
        assert grid

    def test_precipitation_model(self):
        coord = MapCoordinate(47.6763, -116.7798)
        model = AccumulationModel.from_map_coordinate(coord, "quantitative-precipitation")
        resp = model.predict(datetime.now(), verbose=True)
        assert resp


class TestAccumulationMonotonic:
    """A cumulative accumulation curve must be NON-DECREASING: dx/dt >= 0 at
    all times (snowfall/precip can't un-fall). A plain cubic spline overshoots
    and dips between knots on step-like data; the model must use a monotone
    (PCHIP) fit so the graph never goes down. Regression for the reported
    'accumulation graph decreasing' bug."""

    def _doc(self, per_interval_mm, start):
        """A synthetic /gridpoints doc: hourly snowfall amounts (mm) that are
        flat, then a burst, then flat — the shape that makes a cubic spline
        overshoot."""
        values = []
        t = start
        for v in per_interval_mm:
            values.append({"validTime": t.isoformat() + "/PT1H", "value": v})
            t += timedelta(hours=1)
        return {
            "geometry": {"type": "Polygon",
                         "coordinates": [[[-116.6, 48.3], [-116.5, 48.3],
                                          [-116.5, 48.4], [-116.6, 48.4],
                                          [-116.6, 48.3]]]},
            "properties": {
                "snowfallAmount": {"uom": "wmoUnit:mm", "values": values},
            },
        }

    def test_fit_is_non_decreasing(self, monkeypatch):
        start = datetime(2026, 11, 13, 0, 0, 0, tzinfo=timezone.utc)
        # Flat, a burst, then flat again (mm). Cumulative is a monotone step;
        # a cubic interpolant would dip between the step and the plateau.
        per_interval = [0, 0, 0, 0, 50, 75, 25, 0, 0, 0, 0, 0, 0]
        doc = self._doc(per_interval, start)

        monkeypatch.setattr(
            "weather_radar.lib.models.accumulation.NOAAConnection.get_with_expiry",
            lambda self, path: (datetime(2099, 1, 1), doc),
            raising=True,
        )

        m = AccumulationModel(NOAAGridpoint("OTX", 1, 1), "snowfall-amount")
        start_time, end_time, _center, f, units, _expiry = m.model
        assert units == "in"  # mm -> in conversion applied

        # Sample densely across the whole window; the fitted curve must be
        # monotone non-decreasing (allowing a tiny numerical epsilon).
        import numpy as np
        xs = np.linspace(0.0, (end_time - start_time).total_seconds(), 5000)
        ys = f(xs)
        dy = np.diff(ys)
        assert dy.min() >= -1e-9, f"curve decreases (min step {dy.min():.3e})"
        assert ys.min() >= -1e-9, f"curve goes negative (min {ys.min():.4f})"
        # Sanity: total equals sum of intervals converted to inches.
        expected_total = sum(per_interval) / 25.4
        assert ys[-1] == pytest.approx(expected_total, rel=1e-6)

    def test_predict_window_non_decreasing(self, monkeypatch):
        start = datetime(2026, 11, 13, 0, 0, 0, tzinfo=timezone.utc)
        per_interval = [0, 0, 10, 40, 60, 0, 0, 0]
        doc = self._doc(per_interval, start)
        monkeypatch.setattr(
            "weather_radar.lib.models.accumulation.NOAAConnection.get_with_expiry",
            lambda self, path: (datetime(2099, 1, 1), doc),
            raising=True,
        )
        m = AccumulationModel(NOAAGridpoint("OTX", 1, 1), "snowfall-amount")
        s, e, *_ = m.model
        # Step hourly through predict() (the graph path) and check monotonicity.
        prev = None
        t = s
        while t <= e:
            y = m.predict(t)
            assert y >= -1e-9
            if prev is not None:
                assert y >= prev - 1e-9, f"predict decreased: {prev} -> {y}"
            prev = y
            t += timedelta(minutes=15)

