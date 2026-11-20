"""Tests for the GRID wire format shared by the animated layers.

The streams and bundles send the fixed coordinate lattice ONCE as a header
line ({"type":"grid","uom","coords"}) and then values-only frames
({"time","vals"}) whose `vals` align by index to the header coords (null where
a point has no value). This removes the per-frame coordinate repetition that
dominated large-area animation payloads.

Covered here (no FastAPI needed):
  * aqi.predict_area(fixed_grid=True) + grid_points — values-only frames with
    fixed-length vals and nulls; legacy self-contained mode still works.
  * bundle grid header write + serve_bundle_epochs(emit_header=...) round-trip.
"""
import json
import os

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from weather_radar.lib import aqi


def _fit(start=1_793_883_600, hours=6):
    xs = [start + h * 3600 for h in range(hours)]

    def spline(base):
        return CubicSpline(xs, [base + h for h in range(hours)])

    # 4 points; index 1 has NO spline (no coverage) -> null in every frame.
    return {
        "start": start,
        "end": start + (hours - 1) * 3600,
        "points": [[-116.9, 47.2], [-116.8, 47.2], [-116.7, 47.2], [-116.6, 47.2]],
        "splines": [spline(10), None, spline(50), spline(90)],
    }


class TestAqiGridFormat:
    def test_grid_points_are_lon_lat(self):
        fit = _fit()
        pts = aqi.grid_points(fit)
        assert pts == fit["points"]
        # coords[0] is lon (matches the client's normalizeFrame co[0]=lon).
        assert pts[0][0] == -116.9

    def test_fixed_grid_frames_are_values_only_fixed_length(self):
        fit = _fit()
        start = fit["start"]
        frames = list(aqi.predict_area(fit, [start, start + 3600],
                                       fixed_grid=True))
        assert len(frames) == 2
        for fr in frames:
            assert "coords" not in fr          # values-only
            assert set(fr.keys()) == {"epoch", "vals"}
            assert len(fr["vals"]) == 4        # fixed to the point count
            assert fr["vals"][1] is None       # the None-spline point

    def test_fixed_grid_values_track_splines(self):
        fit = _fit()
        start = fit["start"]
        fr = next(iter(aqi.predict_area(fit, [start], fixed_grid=True)))
        # base values at t=start: 10, (null), 50, 90.
        assert fr["vals"] == [10, None, 50, 90]

    def test_fixed_grid_out_of_window_drops_frame(self):
        fit = _fit()
        # Way past the window -> no frame emitted.
        frames = list(aqi.predict_area(fit, [fit["end"] + 10_000],
                                       fixed_grid=True))
        assert frames == []

    def test_legacy_self_contained_still_works(self):
        fit = _fit()
        fr = next(iter(aqi.predict_area(fit, [fit["start"]], fixed_grid=False)))
        # Legacy mode: coords included, None-spline point DROPPED (3 points).
        assert "coords" in fr
        assert len(fr["coords"]) == 3
        assert len(fr["vals"]) == 3

    def test_nonneg_clamp(self):
        # A spline dipping negative is clamped to 0 in both modes.
        xs = [0, 3600, 7200]
        fit = {"start": 0, "end": 7200,
               "points": [[-116.9, 47.2]],
               "splines": [CubicSpline(xs, [-5, -5, -5])]}
        fr = next(iter(aqi.predict_area(fit, [3600], fixed_grid=True)))
        assert fr["vals"] == [0]


class TestBundleGridFormat:
    def _mk(self, tmp_path, monkeypatch):
        import weather_radar.server.bundle as b
        monkeypatch.setattr(b, "BUNDLE_DIR", str(tmp_path))
        # No writer process in unit tests: _store_layer falls back to writing
        # inline (the same save_layer the writer runs).
        return b

    PARAMS = ("pm25", 47.2, -116.9, 10, 10, "aqi")

    def _header(self, coords=None, token=123.0, rng=True):
        h = {"type": "range", "uom": "AQI",
             "coords": coords or [[-116.9, 47.2], [-116.8, 47.2]],
             "meta": {"source_mtime": token}}
        if rng:
            h["range"] = {"min": 0, "max": 300, "uom": "AQI",
                          "start": "s", "end": "e"}
        return h

    def test_write_and_serve_round_trip(self, tmp_path, monkeypatch):
        b = self._mk(tmp_path, monkeypatch)
        d = b._bundle_dir(*self.PARAMS)
        frames = [{"time": 1000, "vals": [10, 20]},
                  {"time": 2000, "vals": [11, None]},
                  {"time": 3000, "vals": [12, 22]}]
        assert b._write_bundle(d, self._header(), iter(frames)) == 3
        # read_bundle_range returns the stored range (prefetch short-circuit).
        monkeypatch.setattr(b, "_source_token", lambda m, k: 123.0)
        rng = b.read_bundle_range(*self.PARAMS)
        assert rng["uom"] == "AQI" and rng["max"] == 300

    def test_serve_emits_binary_header_then_frames(self, tmp_path, monkeypatch):
        from weather_radar.lib import framing as F
        b = self._mk(tmp_path, monkeypatch)
        monkeypatch.setattr(b, "_source_token", lambda m, k: 123.0)
        b._write_bundle(b._bundle_dir(*self.PARAMS), self._header(rng=False),
                        iter([{"time": 1000, "vals": [10, 20]},
                              {"time": 2000, "vals": [11, None]},
                              {"time": 3000, "vals": [12, 22]}]))
        covered = set()
        out = list(b.serve_bundle_epochs(*self.PARAMS, [1000, 3000], covered,
                                         emit_header=True))
        # Each yielded item is one length-prefixed record. First = grid header,
        # then the two matching frames.
        assert len(out) == 3
        hdr = json.loads(next(F.iter_records(out[0])).decode("utf-8"))
        assert hdr["type"] == "grid"
        assert hdr["coords"] == [[-116.9, 47.2], [-116.8, 47.2]]
        assert hdr["enc"] == "i16"
        got = {}
        import struct
        for rec in out[1:]:
            pl = next(F.iter_records(rec))
            n = (len(pl) - 8) // 2
            ints = struct.unpack_from("<%dh" % n, pl, 8)
            got[F.decode_frame_epoch(pl)] = [
                None if iv == hdr["sentinel"] else iv / hdr["scale"] + hdr["offset"]
                for iv in ints]
        assert got == {1000: [10.0, 20.0], 3000: [12.0, 22.0]}
        assert covered == {1000, 3000}

    def test_serve_without_header(self, tmp_path, monkeypatch):
        from weather_radar.lib import framing as F
        b = self._mk(tmp_path, monkeypatch)
        monkeypatch.setattr(b, "_source_token", lambda m, k: 123.0)
        b._write_bundle(b._bundle_dir(*self.PARAMS),
                        self._header(coords=[[-116.9, 47.2]], rng=False),
                        iter([{"time": 2000, "vals": [11]}]))
        covered = set()
        out = list(b.serve_bundle_epochs(*self.PARAMS, [2000], covered,
                                         emit_header=False))
        assert len(out) == 1
        assert F.decode_frame_epoch(next(F.iter_records(out[0]))) == 2000
        assert covered == {2000}

    def test_serve_missing_bundle_yields_nothing(self, tmp_path, monkeypatch):
        b = self._mk(tmp_path, monkeypatch)
        covered = set()
        out = list(b.serve_bundle_epochs("pm25", 1, 2, 10, 10, "aqi",
                                         [1000], covered, emit_header=True))
        assert out == [] and covered == set()


class TestBundleLayersThroughServer:
    """The design: each request stores only what was missing; reads merge."""

    PARAMS = ("pm25", 47.2, -116.9, 10, 10, "aqi")

    @pytest.fixture
    def b(self, tmp_path, monkeypatch):
        import weather_radar.server.bundle as b
        monkeypatch.setattr(b, "BUNDLE_DIR", str(tmp_path))
        monkeypatch.setattr(b, "_source_token", lambda m, k: 500.0)
        return b

    def _build(self, b, epochs, base=0, token=500.0):
        header = {"type": "range", "uom": "AQI", "coords": [[-116.9, 47.2]],
                  "range": {"min": 0, "max": 300, "uom": "AQI"},
                  "meta": {"source_mtime": token}}
        return b._write_bundle(b._bundle_dir(*self.PARAMS), header,
                               iter([{"time": e, "vals": [base + e % 100]}
                                     for e in epochs]))

    def _serve(self, b, wanted):
        from weather_radar.lib import framing as F
        covered = set()
        recs = list(b.serve_bundle_epochs(*self.PARAMS, wanted, covered))
        return [F.decode_frame_epoch(next(F.iter_records(r))) for r in recs], covered

    def test_key_has_no_dt_and_one_bundle_per_area(self, b):
        import inspect
        assert "dt" not in inspect.signature(b._bundle_key).parameters
        assert b._bundle_dir(*self.PARAMS) == b._bundle_dir(*self.PARAMS)

    def test_three_requests_three_layers_one_merged_read(self, b):
        from weather_radar.lib import bundlestore
        self._build(b, [0, 1800, 3600])                    # dt=30 min
        self._build(b, [600, 1200, 2400, 3000])            # dt=10 min: only the new ones
        self._build(b, [300, 900, 1500, 2100, 2700, 3300]) # dt=5 min:  only the new ones
        d = b._bundle_dir(*self.PARAMS)
        assert len(bundlestore.layer_paths(d)) == 3
        got, covered = self._serve(b, list(range(0, 3601, 300)))
        assert got == list(range(0, 3601, 300))      # every 5 min, ascending
        assert covered == set(got)

    def test_dt_30_reads_only_the_first_layer(self, b):
        from weather_radar.lib import bundlestore
        self._build(b, [0, 1800, 3600])
        self._build(b, [600, 1200, 2400])
        ls = bundlestore.usable_layers(b._bundle_dir(*self.PARAMS), 500.0)
        assert len(bundlestore.plan(ls, [0, 1800, 3600])) == 1

    def test_missing_epochs_left_for_live_path(self, b):
        self._build(b, [0, 1800])
        got, covered = self._serve(b, [0, 900, 1800])
        assert got == [0, 1800] and covered == {0, 1800}

    def test_new_data_version_replaces_all_layers(self, b):
        from weather_radar.lib import bundlestore
        self._build(b, [0, 1800])
        self._build(b, [600])
        self._build(b, [0, 1800], token=900.0)       # newer cycle
        ls = bundlestore.layers(b._bundle_dir(*self.PARAMS))
        assert len(ls) == 1 and bundlestore.token_of(ls[0][2]) == 900.0

    def test_request_layer_saved_for_computed_frames(self, b):
        from weather_radar.lib import framing as F, bundlestore
        scale, offset = F.choose_scale("AQI")
        items = [(e, F.encode_frame(e, [e % 100], scale, offset))
                 for e in range(0, 20 * 600, 600)]
        assert b.save_request_layer(*self.PARAMS[:5], "aqi", "AQI",
                                    [[-116.9, 47.2]], items, 500.0,
                                    rng={"min": 0, "max": 300, "uom": "AQI"})
        import threading
        for t in threading.enumerate():
            if t.name == "bundle-layer-save":
                t.join(5)
        got, _ = self._serve(b, [0, 600, 1200])
        assert got == [0, 600, 1200]
        # range comes from the request layer too -> prefetch short-circuit works
        assert b.read_bundle_range(*self.PARAMS)["max"] == 300

    def test_small_requests_do_not_write_layers(self, b):
        from weather_radar.lib import framing as F
        scale, offset = F.choose_scale("AQI")
        few = [(e, F.encode_frame(e, [1], scale, offset)) for e in range(3)]
        assert b.save_request_layer(*self.PARAMS[:5], "aqi", "AQI",
                                    [[-116.9, 47.2]], few, 500.0) is False

    def test_no_token_means_no_request_layer(self, b):
        from weather_radar.lib import framing as F
        scale, offset = F.choose_scale("AQI")
        many = [(e, F.encode_frame(e, [1], scale, offset)) for e in range(40)]
        assert b.save_request_layer(*self.PARAMS[:5], "aqi", "AQI",
                                    [[-116.9, 47.2]], many, None) is False

    def test_writes_go_through_the_writer_when_available(self, b, monkeypatch):
        calls = []
        import weather_radar.lib.cachewriter as cw
        monkeypatch.setattr(cw, "write_bundle_layer",
                            lambda *a: (calls.append(a), "ok")[1])
        assert self._build(b, [0, 1800]) == 2
        assert len(calls) == 1 and calls[0][0] == b._bundle_dir(*self.PARAMS)

    def test_params_for_url_ignores_dt_in_identity(self):
        try:
            import weather_radar.server.bundle as bb
            u = "/map/models/air-quality-pm25/?lat=48.1&lon=-116.4&width=201.9&height=183.7"
            a = bb._params_for_url(u + "&dt=900")
            c = bb._params_for_url(u + "&dt=3600")
        except Exception:
            pytest.skip("server deps unavailable")
        keys = lambda p: bb._bundle_key(p["model"], p["lat"], p["lon"],
                                        p["width"], p["height"], p["kind"])
        assert keys(a) == keys(c)
        assert a["dt"] == 900 and c["dt"] == 3600      # builder still builds at dt


class TestGzipStream:
    def test_gzip_stream_round_trips_and_is_incremental(self):
        import gzip as _gzip
        import io
        import pytest
        try:
            from weather_radar.server.api.models import _gzip_stream
        except Exception:
            pytest.skip("server deps (fastapi) unavailable")
        # _gzip_stream now compresses BYTE chunks (binary sidecar records).
        chunks = [b"\x00\x01\x02\x03", b"\x04\x05\x06", b"\x07"]
        out = b"".join(_gzip_stream(iter(chunks)))
        dec = _gzip.GzipFile(fileobj=io.BytesIO(out)).read()
        assert dec == b"".join(chunks)
