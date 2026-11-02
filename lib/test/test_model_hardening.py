"""Regression tests for model parsing/robustness hardening:

  * accumulation treats null interval values as 0 (no accumulation) instead
    of raising TypeError and dropping the whole gridpoint;
  * observation drops null entries (a null reading is "no data", not 0);
  * both models raise BoundsError when asked for a time outside their forecast
    horizon (below start OR above end), rather than extrapolating.

The gridpoint fetch is mocked so these run offline and deterministically.
"""

from datetime import datetime, timedelta, timezone

import pytest

from weather_radar.lib.area import NOAAGridpoint
from weather_radar.lib.models.accumulation import AccumulationModel
from weather_radar.lib.models.observation import ObservationModel
from weather_radar.lib.models.utils import BoundsError


# A fixed base time so start/end are predictable in assertions.
BASE = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
EXPIRY = datetime(2100, 1, 1)  # far future so the fitted model never expires


def _gridpoint_json(attribute_camel, values, uom="wmoUnit:mm"):
    """Build a minimal NOAA gridpoint GeoJSON body with one property series.

    `values` is a list of (hour_offset, value) pairs; value may be None.
    """
    return {
        "geometry": {
            "coordinates": [[
                [-116.80, 47.70], [-116.76, 47.70],
                [-116.76, 47.66], [-116.80, 47.66],
            ]],
        },
        "properties": {
            attribute_camel: {
                # A None uom means "dimensionless" -> omit the key entirely,
                # matching layers like probabilityOfThunder.
                **({"uom": uom} if uom is not None else {}),
                "values": [
                    {
                        "validTime": (BASE + timedelta(hours=h)).isoformat() + "/PT1H",
                        "value": v,
                    }
                    for h, v in values
                ],
            },
        },
    }


@pytest.fixture
def mock_gridpoint(monkeypatch):
    """Patch NOAAConnection.get_with_expiry to return supplied JSON."""
    def _install(body):
        def fake(self, path):
            return EXPIRY, body
        monkeypatch.setattr(
            "weather_radar.lib.connection.NOAAConnection.get_with_expiry",
            fake,
            raising=True,
        )
    return _install


GP = NOAAGridpoint("OTX", 50, 100)


class TestAccumulationNulls:
    def test_null_values_treated_as_zero(self, mock_gridpoint):
        # Hourly amounts with a null gap; cumulative total must ignore the null
        # (not crash) and equal the sum of the real values.
        body = _gridpoint_json(
            "snowfallAmount",
            [(0, 0.0), (1, 2.0), (2, None), (3, 3.0), (4, None), (5, 1.0)],
        )
        mock_gridpoint(body)
        model = AccumulationModel(GP, "snowfall-amount")
        # Cumulative raw total is 0+2+0+3+0+1 = 6 mm, reported in inches.
        total = model.predict(BASE + timedelta(hours=5))
        assert total == pytest.approx(6.0 / 25.4, abs=1e-6)

    def test_all_null_is_zero_not_crash(self, mock_gridpoint):
        body = _gridpoint_json(
            "snowfallAmount",
            [(0, None), (1, None), (2, None), (3, None)],
        )
        mock_gridpoint(body)
        model = AccumulationModel(GP, "snowfall-amount")
        assert model.predict(BASE + timedelta(hours=3)) == pytest.approx(0.0, abs=1e-6)


class TestAccumulationUnits:
    def test_mm_converted_to_inches(self, mock_gridpoint):
        # 25.4 mm cumulative should read as exactly 1.0 inch, with uom "in".
        body = _gridpoint_json(
            "snowfallAmount",
            [(0, 0.0), (1, 10.0), (2, 10.0), (3, 5.4)],  # cumulative 25.4 mm
        )
        mock_gridpoint(body)
        model = AccumulationModel(GP, "snowfall-amount")
        feature = model.predict(BASE + timedelta(hours=3), verbose=True)
        assert feature["properties"]["snowfall-amount"] == pytest.approx(1.0, abs=1e-6)
        assert feature["properties"]["uom"] == "in"


class TestMissingUom:
    def test_observation_without_uom(self, mock_gridpoint):
        # A dimensionless layer (no uom, e.g. probabilityOfThunder) must not
        # crash; units come back empty.
        body = _gridpoint_json(
            "probabilityOfThunder",
            [(0, 5.0), (1, 20.0), (2, 60.0), (3, 10.0)],
            uom=None,
        )
        mock_gridpoint(body)
        model = ObservationModel(GP, "probability-of-thunder")
        feature = model.predict(BASE + timedelta(hours=2), verbose=True)
        assert feature["properties"]["uom"] == ""
        assert feature["properties"]["probability-of-thunder"] == pytest.approx(60.0, abs=1e-6)

    def test_accumulation_without_uom(self, mock_gridpoint):
        # No uom -> no conversion factor; values pass through unchanged.
        body = _gridpoint_json(
            "someAccumulation",
            [(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0)],
            uom=None,
        )
        mock_gridpoint(body)
        model = AccumulationModel(GP, "some-accumulation")
        feature = model.predict(BASE + timedelta(hours=3), verbose=True)
        assert feature["properties"]["uom"] == ""
        assert feature["properties"]["some-accumulation"] == pytest.approx(4.0, abs=1e-6)


class TestObservationNulls:
    def test_null_entries_are_dropped(self, mock_gridpoint):
        # A null temperature must be omitted (gap interpolated), not read as 0.
        body = _gridpoint_json(
            "temperature",
            [(0, 10.0), (1, None), (2, 12.0), (3, None), (4, 14.0)],
            uom="wmoUnit:degC",
        )
        mock_gridpoint(body)
        model = ObservationModel(GP, "temperature")
        # Spline passes through the kept (non-null) points; sampling at a kept
        # time returns that value (converted degC->degF: 12C -> 53.6F).
        f = model.predict(BASE + timedelta(hours=2))
        assert f == pytest.approx(12.0 * 9 / 5 + 32, abs=1e-6)
        # And the value is nowhere near what a 0-substitution would produce
        # (0C -> 32F) at the null slot's neighborhood.
        assert f > 50.0


class TestBounds:
    def test_accumulation_below_start_raises(self, mock_gridpoint):
        body = _gridpoint_json("snowfallAmount", [(0, 1.0), (1, 2.0), (2, 3.0), (3, 4.0)])
        mock_gridpoint(body)
        model = AccumulationModel(GP, "snowfall-amount")
        with pytest.raises(BoundsError):
            model.predict(BASE - timedelta(hours=1))

    def test_accumulation_above_end_raises(self, mock_gridpoint):
        body = _gridpoint_json("snowfallAmount", [(0, 1.0), (1, 2.0), (2, 3.0), (3, 4.0)])
        mock_gridpoint(body)
        model = AccumulationModel(GP, "snowfall-amount")
        with pytest.raises(BoundsError):
            model.predict(BASE + timedelta(hours=10))

    def test_observation_below_start_raises(self, mock_gridpoint):
        body = _gridpoint_json(
            "temperature",
            [(0, 10.0), (1, 11.0), (2, 12.0), (3, 13.0)],
            uom="wmoUnit:degC",
        )
        mock_gridpoint(body)
        model = ObservationModel(GP, "temperature")
        with pytest.raises(BoundsError):
            model.predict(BASE - timedelta(hours=1))

    def test_observation_above_end_raises(self, mock_gridpoint):
        body = _gridpoint_json(
            "temperature",
            [(0, 10.0), (1, 11.0), (2, 12.0), (3, 13.0)],
            uom="wmoUnit:degC",
        )
        mock_gridpoint(body)
        model = ObservationModel(GP, "temperature")
        with pytest.raises(BoundsError):
            model.predict(BASE + timedelta(hours=10))
