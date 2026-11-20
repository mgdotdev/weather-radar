"""Point-series cache wiring + pre-warm targets (server/pointseries.py)."""
import threading
from urllib.parse import quote

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("jinja2")        # weather_radar.server.app imports it

from weather_radar.lib import seriescache as sc          # noqa: E402

RESULT = {"type": "FeatureCollection",
          "features": [{"type": "Feature",
                        "properties": {"temperature": 40.0, "time": 1, "uom": "degF"},
                        "geometry": {"type": "Point", "coordinates": [-116.7, 47.7]}}]}

# The user's real watched map URL (6 pins).
MAP_URL = ("https://weather.vancleave.io/map/models/snowfall-amount/"
           "?lat=48.145506&lon=-116.391332&width=201.90&height=183.69&dt=900&zoom=8.24"
           "&speed=50.00&pins=47.77877%2C-116.70459%2CHome%3B48.65516%2C-116.65461%2C"
           "Beehive%2520Lake%3B47.43378%2C-115.75997%2CStevens%2520Lake%3B"
           "48.18882%2C-116.08184%2CScotchman%27s%3B47.67359%2C-116.78221%2C"
           "Coeur%2520d%27Alene%3B48.27823%2C-116.55037%2CSandpoint&pinSize=2")


@pytest.fixture
def ps(tmp_path, monkeypatch):
    import weather_radar.server.pointseries as ps
    monkeypatch.setattr(sc, "SERIES_ROOT", str(tmp_path))
    sc._READ_CACHE.clear()
    state = {"token": 100.0, "computes": []}
    monkeypatch.setattr(ps, "_token", lambda kind, param: state["token"])
    monkeypatch.setattr(ps, "_grib_source", lambda: True)

    def compute(kind, param, lat, lon, dt):
        state["computes"].append((kind, param, round(lat, 4), round(lon, 4), dt))
        return RESULT
    monkeypatch.setattr(ps, "_compute", compute)
    ps.state = state
    return ps


def _join_saves():
    for t in threading.enumerate():
        if t.name == "series-cache-save":
            t.join(10)


class TestEndpointCache:
    def test_first_request_computes_second_is_a_read(self, ps):
        a = ps.point_series("observation", "temperature", 47.7, -116.7, 450)
        _join_saves()
        b = ps.point_series("observation", "temperature", 47.7, -116.7, 450)
        assert a == b == RESULT and len(ps.state["computes"]) == 1

    def test_new_data_version_recomputes_once(self, ps):
        ps.point_series("observation", "temperature", 47.7, -116.7, 450); _join_saves()
        ps.state["token"] = 200.0
        ps.point_series("observation", "temperature", 47.7, -116.7, 450); _join_saves()
        ps.point_series("observation", "temperature", 47.7, -116.7, 450)
        assert len(ps.state["computes"]) == 2

    def test_dt_zero_shares_the_hourly_entry(self, ps):
        ps.point_series("observation", "temperature", 47.7, -116.7, 3600); _join_saves()
        ps.point_series("observation", "temperature", 47.7, -116.7, 0)
        assert len(ps.state["computes"]) == 1

    def test_other_point_or_attribute_is_separate(self, ps):
        ps.point_series("observation", "temperature", 47.7, -116.7, 450); _join_saves()
        ps.point_series("observation", "dewpoint", 47.7, -116.7, 450)
        ps.point_series("observation", "temperature", 47.8, -116.7, 450)
        assert len(ps.state["computes"]) == 3

    def test_empty_results_are_not_cached(self, ps, monkeypatch):
        monkeypatch.setattr(ps, "_compute", lambda *a: {"type": "FeatureCollection", "features": []})
        ps.point_series("observation", "temperature", 47.7, -116.7, 450); _join_saves()
        assert not sc.read(sc.path_for("observation", "temperature", 47.7, -116.7, 450))

    def test_unknown_token_bypasses_the_cache(self, ps):
        ps.state["token"] = None
        ps.point_series("observation", "temperature", 47.7, -116.7, 450); _join_saves()
        ps.point_series("observation", "temperature", 47.7, -116.7, 450)
        assert len(ps.state["computes"]) == 2

    def test_graph_endpoint_uses_it(self, ps, monkeypatch):
        import weather_radar.server.api.models as m
        monkeypatch.setattr(m, "weather_source", lambda: "nbm")
        a = m.models("observation", "temperature", 47.7, -116.7, dt=450); _join_saves()
        b = m.models("observation", "temperature", 47.7, -116.7, dt=450)
        assert a == b == RESULT and len(ps.state["computes"]) == 1


class TestTargets:
    def test_map_url_warms_each_pin_for_the_maps_layer_only(self, ps):
        t = ps.targets_for_url(MAP_URL)
        assert len(t) == 6 * 2                                   # 6 pins x (graph, forecast) steps
        assert {(k, a) for k, a, *_ in t} == {("accumulation", "snowfall-amount")}
        assert {d for *_, d in t} == {450, 3600}
        assert (("accumulation", "snowfall-amount", 47.77877, -116.70459, 450) in t)

    def test_map_center_is_not_a_target(self, ps):
        assert not [x for x in ps.targets_for_url(MAP_URL)
                    if abs(x[2] - 48.145506) < 1e-6 and abs(x[3] + 116.391332) < 1e-6]

    def test_map_without_pins_has_no_targets(self, ps):
        assert ps.targets_for_url("/map/models/temperature/?lat=1&lon=2&width=3&height=4") == []

    def test_malformed_pins_are_skipped(self, ps):
        u = "/map/models/temperature/?lat=1&lon=2&pins=" + quote("x,y,Bad;;47.5,-116.5,Ok;9")
        assert [(x[2], x[3]) for x in ps.targets_for_url(u)] == [(47.5, -116.5)] * 2

    def test_aqi_map_pins_are_targets_for_the_pollutant(self, ps):
        t = ps.targets_for_url("/map/models/air-quality-pm25/?lat=1&lon=2&pins=1,2,A")
        assert sorted(t) == [("aqi", "pm25", 1.0, 2.0, 450), ("aqi", "pm25", 1.0, 2.0, 3600)]

    def test_graph_url_warms_exactly_what_it_names(self, ps):
        models = quote('{"observation":["temperature","dewpoint"],"accumulation":["snowfall-amount"]}')
        t = ps.targets_for_url(f"/graph/?models={models}&lat=47.7&lon=-116.7&dt=900")
        assert sorted((k, a, d) for k, a, _la, _lo, d in t) == [
            ("accumulation", "snowfall-amount", 900),
            ("observation", "dewpoint", 900), ("observation", "temperature", 900)]
        assert all((la, lo) == (47.7, -116.7) for _k, _a, la, lo, _d in t)

    def test_graph_url_default_dt_matches_the_graph_route(self, ps):
        t = ps.targets_for_url('/graph/?models=' + quote('{"observation":["temperature"]}') + "&lat=1&lon=2")
        assert [d for *_, d in t] == [450]

    def test_bad_graph_urls_yield_nothing(self, ps):
        assert ps.targets_for_url("/graph/?models=notjson&lat=1&lon=2") == []
        assert ps.targets_for_url('/graph/?models=' + quote('{"observation":["t"]}')) == []   # no point

    def test_other_urls_yield_nothing(self, ps):
        assert ps.targets_for_url("/forecast/?model=temperature&lat=1&lon=2") == []
        assert ps.targets_for_url("/api/models/x") == []

    def test_duplicate_pins_are_deduped(self, ps):
        u = "/map/models/temperature/?lat=1&lon=2&pins=" + quote("5,6,A;5.0000001,6,B")
        assert len(ps.targets_for_url(u)) == 2


class TestWarm:
    def test_builds_then_is_fresh(self, ps):
        assert ps.warm("observation", "temperature", 47.7, -116.7, 450) == "built"
        assert ps.warm("observation", "temperature", 47.7, -116.7, 450) == "fresh"
        assert len(ps.state["computes"]) == 1
        assert sc.read(sc.path_for("observation", "temperature", 47.7, -116.7, 450))["managed"]

    def test_request_cached_entry_is_promoted_without_recompute(self, ps):
        ps.point_series("observation", "temperature", 47.7, -116.7, 450); _join_saves()
        assert ps.warm("observation", "temperature", 47.7, -116.7, 450) == "marked"
        assert len(ps.state["computes"]) == 1
        assert ps.warm("observation", "temperature", 47.7, -116.7, 450) == "fresh"

    def test_new_cycle_rebuilds_and_old_one_serves_meanwhile(self, ps):
        ps.warm("observation", "temperature", 47.7, -116.7, 450)
        ps.state["token"] = 200.0
        path = sc.path_for("observation", "temperature", 47.7, -116.7, 450)
        assert sc.get(path, 200.0) == RESULT                     # stale-but-managed served
        assert ps.warm("observation", "temperature", 47.7, -116.7, 450) == "built"
        assert sc.read(path)["token"] == 200.0

    def test_skips_non_grib_sources_and_unknown_tokens(self, ps, monkeypatch):
        monkeypatch.setattr(ps, "_grib_source", lambda: False)
        assert ps.warm("observation", "temperature", 1, 2, 450) == "skipped"
        monkeypatch.setattr(ps, "_grib_source", lambda: True)
        ps.state["token"] = None
        assert ps.warm("observation", "temperature", 1, 2, 450) == "skipped"

    def test_empty_result_is_not_stored(self, ps, monkeypatch):
        monkeypatch.setattr(ps, "_compute", lambda *a: {"features": []})
        assert ps.warm("observation", "temperature", 1, 2, 450) == "empty"

    def test_warm_url_covers_the_whole_map_url(self, ps):
        counts = ps.warm_url(MAP_URL)
        assert counts == {"built": 12}
        assert ps.warm_url(MAP_URL) == {"fresh": 12}             # steady state: no work
        assert len(ps.state["computes"]) == 12

    def test_one_failure_does_not_stop_the_rest(self, ps, monkeypatch):
        calls = []
        def flaky(kind, param, lat, lon, dt):
            calls.append(lat)
            if len(calls) == 1:
                raise RuntimeError("boom")
            return RESULT
        monkeypatch.setattr(ps, "_compute", flaky)
        assert ps.warm_url(MAP_URL) == {"error": 1, "built": 11}


class TestBuilderHook:
    def test_each_builder_pass_warms_every_watched_url(self, ps, monkeypatch):
        import weather_radar.server.bundle as b
        from weather_radar.lib import hotcache
        seen = []
        monkeypatch.setattr(hotcache, "list_watched", lambda: {"urls": [MAP_URL, "/graph/?x=1"]})
        monkeypatch.setattr(b, "build_for_url", lambda url: True)
        monkeypatch.setattr(ps, "warm_url", lambda url: seen.append(url))
        b._build_pass()
        assert seen == [MAP_URL, "/graph/?x=1"]

    def test_a_failing_warm_does_not_break_the_pass(self, ps, monkeypatch):
        import weather_radar.server.bundle as b
        from weather_radar.lib import hotcache
        monkeypatch.setattr(hotcache, "list_watched", lambda: {"urls": ["a", "b"]})
        built = []
        monkeypatch.setattr(b, "build_for_url", lambda url: built.append(url) or True)
        monkeypatch.setattr(ps, "warm_url", lambda url: (_ for _ in ()).throw(RuntimeError("x")))
        b._build_pass()
        assert built == ["a", "b"]


# --- AQI point series (the pin on an air-quality map) -----------------------

class _Spline:
    """Stands in for a CubicSpline: value(t) = base + slope * (t - T0)."""
    def __init__(self, base, slope=0.0):
        self.base, self.slope = base, slope

    def __call__(self, t):
        return self.base + self.slope * (t - T0)


T0 = 1_800_000_000 - 1_800_000_000 % 3600
LAT, LON = 48.18882, -116.08184


@pytest.fixture
def aqi_fit(monkeypatch):
    from weather_radar.lib import aqi
    fit = {"start": T0, "end": T0 + 6 * 3600,
           "points": [[LON, LAT], [LON + 0.05, LAT], [LON, LAT + 0.05], [LON + 5, LAT + 5]],
           "splines": [_Spline(40.0), _Spline(60.0), None, _Spline(500.0)]}
    monkeypatch.setattr(aqi, "fit_area_dense", lambda area, pollutant: fit)
    return fit


class TestAqiPointSeries:
    def test_window_step_and_shape(self, aqi_fit):
        import weather_radar.server.api.models as m
        fc = m._aqi_point_series("pm25", LAT, LON, 3600)
        assert [f["properties"]["time"] for f in fc["features"]] == \
            [(T0 + h * 3600) * 1000 for h in range(7)]
        f = fc["features"][0]
        assert set(f["properties"]) == {"pm25", "time", "uom"} and f["properties"]["uom"] == "AQI"
        assert f["geometry"]["coordinates"] == [LON, LAT]

    def test_graph_default_step(self, aqi_fit):
        import weather_radar.server.api.models as m
        assert len(m._aqi_point_series("pm25", LAT, LON, 450)["features"]) == 6 * 8 + 1

    def test_value_is_the_idw_blend_of_nearby_points(self, aqi_fit):
        import weather_radar.server.api.models as m
        eps = m._IDW_EPS
        d = [0.0, 0.05 ** 2, 5.0 ** 2 * 2]                  # the three usable points
        w = [1 / (x + eps) for x in d]
        want = (w[0] * 40 + w[1] * 60 + w[2] * 500) / sum(w)
        got = m._aqi_point_series("pm25", LAT, LON, 3600)["features"][0]["properties"]["pm25"]
        assert got == round(want, 1)                        # one decimal, not a whole number
        assert 40 <= got <= 60                              # dominated by the pin's own cell

    def test_a_slow_rise_is_smooth_not_a_staircase(self, aqi_fit):
        import weather_radar.server.api.models as m
        # +0.135 AQI per 450 s step: whole-number rounding would give ~7 plateaus
        aqi_fit["splines"] = [_Spline(10.0, 0.0003)] * 4
        vals = [f["properties"]["pm25"] for f in
                m._aqi_point_series("pm25", LAT, LON, 450)["features"]]
        assert len(vals) == 49
        assert len(set(vals)) >= 40                         # was ~8 with integer rounding
        assert all(b >= a for a, b in zip(vals, vals[1:]))  # still monotone
        assert max(abs(b - a) for a, b in zip(vals, vals[1:])) < 0.3

    def test_points_without_a_spline_are_ignored(self, aqi_fit):
        import weather_radar.server.api.models as m
        aqi_fit["splines"] = [None, None, None, None]
        assert m._aqi_point_series("pm25", LAT, LON, 3600)["features"] == []

    def test_negative_spline_values_clamp_to_zero(self, aqi_fit):
        import weather_radar.server.api.models as m
        aqi_fit["splines"] = [_Spline(-30.0)] * 4
        assert {f["properties"]["pm25"] for f in
                m._aqi_point_series("pm25", LAT, LON, 3600)["features"]} == {0}

    def test_values_follow_the_forecast_over_time(self, aqi_fit):
        import weather_radar.server.api.models as m
        aqi_fit["splines"] = [_Spline(10.0, 0.001)] * 4
        vals = [f["properties"]["pm25"] for f in m._aqi_point_series("pm25", LAT, LON, 3600)["features"]]
        assert vals == sorted(vals) and vals[0] == 10 and vals[-1] > vals[0]

    def test_unavailable_grid_or_pollutant_gives_an_empty_series_not_an_error(self, monkeypatch):
        import weather_radar.server.api.models as m
        from weather_radar.lib import aqi
        monkeypatch.setattr(aqi, "fit_area_dense", lambda area, pollutant: None)
        assert m._aqi_point_series("bogus", LAT, LON, 3600) == {"type": "FeatureCollection", "features": []}


class TestAqiGraphEndpoint:
    def test_works_whatever_the_weather_source(self, ps, aqi_fit, monkeypatch):
        import weather_radar.server.api.models as m
        monkeypatch.setattr(m, "weather_source", lambda: "gridpoints")
        monkeypatch.setattr(ps, "_compute", lambda kind, param, lat, lon, dt:
                            m._point_series(kind, param, lat, lon, dt))
        fc = m.models("aqi", "pm25", LAT, LON, dt=450)
        assert len(fc["features"]) == 49 and fc["features"][0]["properties"]["uom"] == "AQI"

    def test_second_request_is_served_from_the_cache(self, ps, aqi_fit, monkeypatch):
        import weather_radar.server.api.models as m
        monkeypatch.setattr(m, "weather_source", lambda: "gridpoints")
        calls = []
        real = m._point_series
        monkeypatch.setattr(ps, "_compute", lambda *a: (calls.append(a), real(*a))[1])
        a = m.models("aqi", "pm25", LAT, LON, dt=450); _join_saves()
        b = m.models("aqi", "pm25", LAT, LON, dt=450)
        assert a == b and len(calls) == 1

    def test_first_ever_request_is_cached_even_though_no_grid_was_on_disk_before(self, ps, aqi_fit, monkeypatch):
        # token unknown BEFORE the compute (nothing downloaded yet), known after
        import weather_radar.server.api.models as m
        monkeypatch.setattr(m, "weather_source", lambda: "gridpoints")
        seq = iter([None, 100.0, 100.0, 100.0])
        monkeypatch.setattr(ps, "_token", lambda kind, param: next(seq))
        a = m.models("aqi", "pm25", LAT, LON, dt=450); _join_saves()
        assert sc.read(sc.path_for("aqi", "pm25", LAT, LON, 450)) is not None

    def test_warm_works_for_aqi_even_when_the_weather_source_is_not_gridded(self, ps, monkeypatch):
        monkeypatch.setattr(ps, "_grib_source", lambda: False)
        assert ps.warm("aqi", "pm25", LAT, LON, 450) == "built"
        assert ps.warm("observation", "temperature", LAT, LON, 450) == "skipped"
