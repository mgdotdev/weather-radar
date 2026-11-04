"""Tests for non-negative clamping of physically non-negative model outputs.

A CubicSpline fit through snow/precip values can dip below zero between knots
(Runge-type oscillation), producing spurious negatives (e.g. "negative
snowfall"). clamp_value (and the model predict paths) must clamp those to 0
for non-negative attributes, while leaving legitimately-negative attributes
(temperature) untouched.
"""
from weather_radar.lib.models.utils import clamp_value, NON_NEGATIVE_ATTRIBUTES


class TestClampValue:
    def test_non_negative_attributes_clamped(self):
        for attr in ("snowfall-amount", "quantitative-precipitation",
                     "ice-accumulation"):
            assert clamp_value(attr, -0.3) == 0.0
            assert clamp_value(attr, 0.0) == 0.0
            assert clamp_value(attr, 2.5) == 2.5

    def test_temperature_not_clamped(self):
        # Temperature is legitimately negative; must pass through unchanged.
        assert clamp_value("temperature", -12.0) == -12.0
        assert clamp_value("temperature", 20.0) == 20.0

    def test_none_passthrough(self):
        assert clamp_value("snowfall-amount", None) is None

    def test_snowfall_is_registered_non_negative(self):
        # Guard against someone dropping snowfall from the set (the reported
        # bug was negative snowfall on the heatmap).
        assert "snowfall-amount" in NON_NEGATIVE_ATTRIBUTES
