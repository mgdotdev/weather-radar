"""Sample data for the preview pages (server/fixtures.py): the right wire formats,
deterministic, fast, and only ever served when asked for (and behind the token)."""
import asyncio
import gzip
import json
import re
import struct
import time

import pytest

pytest.importorskip("fastapi")

from weather_radar.lib import framing as F                 # noqa: E402
from weather_radar.server import fixtures as fx             # noqa: E402

Q = {"lat": "48.1", "lon": "-116.4", "width": "200", "height": "180", "dt": "3600"}


@pytest.fixture(autouse=True)
def _fixed_clock(monkeypatch):
    """Pin 'now' so the 72 h window can't shift mid-test."""
    monkeypatch.setattr(fx.time, "time", lambda: 1_800_000_000 + 0.0)


def _decode_stream(body):
    raw = gzip.decompress(body)
    recs = list(F.iter_records(raw))
    hdr = json.loads(recs[0].decode())
    frames = {}
    for r in recs[1:]:
        n = (len(r) - 8) // 2
        ints = struct.unpack_from("<%dh" % n, r, 8)
        frames[F.decode_frame_epoch(r)] = [None if i == hdr["sentinel"] else i / hdr["scale"] + hdr["offset"] for i in ints]
    return hdr, frames


START = 1_800_000_000 // 3600 * 3600


class TestEndpoints:
    def test_prefetch_progress_is_ndjson_the_client_can_read(self):
        status, _, body, media = fx.respond("/api/models/prefetch/stream", Q)
        assert status == 200 and json.loads(body.decode().strip())["done"] == 1

    def test_plain_prefetch_is_204(self):
        assert fx.respond("/api/models/prefetch", Q)[0] == 204

    @pytest.mark.parametrize("attr,uom,accum", [("temperature", "degF", False), ("snowfall-amount", "in", True),
                                                ("relative-humidity", "percent", False), ("wind-speed", "mph", False)])
    def test_range_has_what_applyRange_needs(self, attr, uom, accum):
        status, _, body, media = fx.respond(f"/api/models/{attr}/type/range", {**Q, "kind": "accumulation" if accum else "observation"})
        r = json.loads(body)
        assert status == 200 and media == "application/json"
        assert r["uom"] == uom and r["max"] > r["min"]
        assert r["start"].endswith("+00:00") and r["end"] > r["start"]
        ex = r["extent"]
        assert ex["w"] < ex["e"] and ex["s"] < ex["n"] and ex["w"] < -116.4 < ex["e"] and ex["s"] < 48.1 < ex["n"]

    def test_aqi_range_is_the_fixed_display_scale(self):
        r = json.loads(fx.respond("/api/models/aqi/range", {**Q, "pollutant": "pm25"})[2])
        assert (r["min"], r["max"], r["uom"]) == (0, 300, "AQI")

    def test_series_stream_matches_the_wire_format_and_the_requested_epochs(self):
        status, headers, body, media = fx.respond("/api/models/temperature/type/series/stream",
                                                  {**Q, "kind": "observation", "start": str(START), "end": str(START + 6 * 3600)})
        assert status == 200 and headers == {"Content-Encoding": "gzip"} and media == "application/octet-stream"
        hdr, frames = _decode_stream(body)
        assert hdr["type"] == "grid" and hdr["uom"] == "degF" and len(hdr["coords"]) == fx.LATTICE ** 2
        assert sorted(frames) == [START + h * 3600 for h in range(7)]
        assert all(len(v) == len(hdr["coords"]) for v in frames.values())

    def test_coords_are_lon_lat_and_cover_the_requested_area(self):
        hdr, _ = _decode_stream(fx.respond("/api/models/temperature/type/series/stream", {**Q, "kind": "observation", "start": str(START), "end": str(START)})[2])
        lons = [c[0] for c in hdr["coords"]]; lats = [c[1] for c in hdr["coords"]]
        assert min(lons) < -116.4 < max(lons) and min(lats) < 48.1 < max(lats)
        assert (max(lats) - min(lats)) * 111.32 == pytest.approx(180, rel=0.01)       # height in km

    def test_times_batches_and_compact_runs_are_honoured_and_out_of_window_is_dropped(self):
        run = f"{START + 3 * 3600}:{START + 1 * 3600}:3600"                 # descending run, as the client sends
        _, frames = _decode_stream(fx.respond("/api/models/temperature/type/series/stream",
                                              {**Q, "kind": "observation", "times": run + f",{START + 99 * 3600},{START - 3600}"})[2])
        assert sorted(frames) == [START + h * 3600 for h in (1, 2, 3)]       # 99h and -1h are outside the window

    def test_accumulations_rise_monotonically_and_never_go_negative(self):
        _, frames = _decode_stream(fx.respond("/api/models/snowfall-amount/type/series/stream",
                                              {**Q, "kind": "accumulation", "start": str(START), "end": str(START + 72 * 3600), "dt": "21600"})[2])
        series = [frames[t][37] for t in sorted(frames)]
        assert series == sorted(series) and series[0] == 0 and series[-1] > 0

    def test_values_stay_inside_the_layers_own_range(self):
        for attr, (_u, lo, hi, _a) in fx._SPECS.items():
            if attr in ("pm25", "ozone"):
                lo, hi = 0, 300
            kind = "accumulation" if fx._SPECS[attr][3] else "observation"
            _, frames = _decode_stream(fx.respond(f"/api/models/{attr}/type/series/stream",
                                                  {**Q, "kind": kind, "start": str(START), "end": str(START + 72 * 3600), "dt": "10800"})[2])
            vals = [v for f in frames.values() for v in f if v is not None]
            assert min(vals) >= lo - 0.2 and max(vals) <= hi + 0.2, attr

    def test_aqi_stream_uses_the_pollutant_and_aqi_units(self):
        hdr, frames = _decode_stream(fx.respond("/api/models/aqi/stream", {**Q, "pollutant": "pm25-nowcast", "start": str(START), "end": str(START + 7200)})[2])
        assert hdr["uom"] == "AQI" and len(frames) == 3

    def test_unknown_attributes_still_get_usable_data(self):
        hdr, frames = _decode_stream(fx.respond("/api/models/made-up/type/series/stream", {**Q, "kind": "observation", "start": str(START), "end": str(START)})[2])
        assert len(frames) == 1 and all(v is not None for v in frames[START])

    def test_graph_series_is_a_feature_collection_the_graph_page_plots(self):
        r = json.loads(fx.respond("/api/models/observation/params/temperature", {"lat": "47.7", "lon": "-116.7", "dt": "450"})[2])
        f = r["features"]
        assert r["type"] == "FeatureCollection" and len(f) == 72 * 8 + 1
        assert set(f[0]["properties"]) == {"temperature", "time", "uom"}
        assert f[1]["properties"]["time"] - f[0]["properties"]["time"] == 450 * 1000
        assert f[0]["geometry"]["coordinates"] == [-116.7, 47.7]

    def test_graph_series_default_step_is_hourly_and_points_differ_in_space(self):
        a = json.loads(fx.respond("/api/models/observation/params/temperature", {"lat": "47.7", "lon": "-116.7", "dt": "0"})[2])["features"]
        b = json.loads(fx.respond("/api/models/observation/params/temperature", {"lat": "46.0", "lon": "-110.0"})[2])["features"]
        assert len(a) == 73 and [x["properties"]["temperature"] for x in a] != [x["properties"]["temperature"] for x in b]

    def test_deterministic(self):
        args = ("/api/models/temperature/type/series/stream", {**Q, "kind": "observation", "start": str(START), "end": str(START + 7200)})
        assert gzip.decompress(fx.respond(*args)[2]) == gzip.decompress(fx.respond(*args)[2])

    def test_it_is_instant(self):
        t = time.perf_counter()
        fx.respond("/api/models/temperature/type/series/stream", {**Q, "kind": "observation", "start": str(START), "end": str(START + 72 * 3600), "dt": "900"})
        assert time.perf_counter() - t < 1.5                    # 289 frames x 400 points; the real pipeline takes seconds

    @pytest.mark.parametrize("path", ["/api/geocode/", "/api/hotcache", "/map/models/temperature/", "/api/models/point",
                                      "/api/models/x/type/ensemble"])
    def test_other_paths_fall_through_to_the_real_handlers(self, path):
        assert fx.respond(path, Q) is None


# ---- the middleware ----------------------------------------------------------

def _call(app, path, query=""):
    """Minimal ASGI GET; returns (status, headers, body). (No httpx needed.)"""
    sent = {"status": None, "headers": {}, "body": b""}

    async def run():
        scope = {"type": "http", "method": "GET", "path": path, "raw_path": path.encode(), "query_string": query.encode(),
                 "headers": [], "server": ("t", 80), "scheme": "http", "client": ("c", 1), "http_version": "1.1",
                 "root_path": "", "app": app}
        async def receive(): return {"type": "http.request", "body": b"", "more_body": False}
        async def send(m):
            if m["type"] == "http.response.start":
                sent["status"] = m["status"]; sent["headers"] = {k.decode().lower(): v.decode() for k, v in m["headers"]}
            elif m["type"] == "http.response.body":
                sent["body"] += m.get("body", b"")
        await app(scope, receive, send)
    asyncio.run(run())
    return sent["status"], sent["headers"], sent["body"]


@pytest.fixture
def app():
    from fastapi import FastAPI
    a = FastAPI()
    calls = []

    @a.get("/api/models/{model}/type/range")
    def real(model: str):
        calls.append(model)
        return {"real": True}

    @a.get("/other")
    def other():
        return {"other": True}
    fx.install(a)
    a.calls = calls
    return a


class TestMiddleware:
    def test_fixture_flag_answers_from_sample_data_without_touching_the_real_handler(self, app):
        status, headers, body = _call(app, "/api/models/temperature/type/range", "lat=1&lon=2&fixture=1&kind=observation")
        assert status == 200 and json.loads(body)["uom"] == "degF" and app.calls == []

    def test_without_the_flag_the_real_handler_runs(self, app):
        status, _, body = _call(app, "/api/models/temperature/type/range", "lat=1&lon=2")
        assert json.loads(body) == {"real": True} and app.calls == ["temperature"]

    @pytest.mark.parametrize("q", ["fixture=0", "fixture=", "fixture=true", "fixtures=1"])
    def test_only_fixture_equals_1_counts(self, app, q):
        assert json.loads(_call(app, "/api/models/temperature/type/range", q)[2]) == {"real": True}

    def test_non_api_paths_are_never_intercepted(self, app):
        assert json.loads(_call(app, "/other", "fixture=1")[2]) == {"other": True}

    def test_unsupported_api_paths_with_the_flag_fall_through(self, app):
        status, _, _ = _call(app, "/api/models/x/y/z", "fixture=1")
        assert status == 404                                    # the real router's answer, not ours

    def test_can_be_switched_off(self, app, monkeypatch):
        monkeypatch.setenv("WR_PREVIEW_FIXTURES", "0")
        assert json.loads(_call(app, "/api/models/temperature/type/range", "fixture=1")[2]) == {"real": True}

    def test_gzip_header_matches_the_real_stream_endpoints(self, app):
        status, headers, body = _call(app, "/api/models/temperature/type/series/stream", f"fixture=1&kind=observation&lat=1&lon=2&width=10&height=10&start={START}&end={START}")
        assert status == 200 and headers.get("content-encoding") == "gzip"
        hdr, frames = _decode_stream(body)                       # i.e. a valid gzip of a valid record stream
        assert hdr["type"] == "grid" and list(frames) == [START]


class TestServerWiring:
    def test_sample_data_is_installed_before_access_control_so_the_token_wraps_it(self):
        """The LAST middleware added is the outermost, so installing sample data
        first means the access check runs first: no token, no sample data."""
        import os
        src = open(os.path.join(os.path.dirname(__file__), "..", "..", "server", "src", "weather_radar",
                                "server", "__main__.py"), encoding="utf-8").read()
        body = src[src.index("def server():"):]
        assert body.index("fixtures.install(server)") < body.index("_install_access_control(server)")
