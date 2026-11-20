"""Shared pytest fixtures."""
import pytest


@pytest.fixture(autouse=True)
def _clear_aqi_fit_memo():
    """aqi.fit_area / fit_area_dense memoize by (cycle, area); tests build many
    fake grids, so start each one with an empty memo."""
    from weather_radar.lib import aqi
    aqi._FIT_CACHE.clear()
    aqi._FIT_KEY_LOCKS.clear()
    yield
