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
        return b

    def test_write_and_serve_round_trip(self, tmp_path, monkeypatch):
        b = self._mk(tmp_path, monkeypatch)
        model, lat, lon, w, h, dt, kind = "pm25", 47.2, -116.9, 10, 10, 3600, "aqi"
        path = b._bundle_path(model, lat, lon, w, h, dt, kind)
        header = {
            "type": "range",
            "range": {"min": 0, "max": 300, "uom": "AQI",
                      "start": "s", "end": "e"},
            "uom": "AQI",
            "coords": [[-116.9, 47.2], [-116.8, 47.2]],
            "meta": {"source_mtime": 123.0},
        }
        frames = [
            {"time": 1000, "vals": [10, 20]},
            {"time": 2000, "vals": [11, None]},
            {"time": 3000, "vals": [12, 22]},
        ]
        n = b._write_bundle(path, header, iter(frames))
        assert n == 3
        assert os.path.exists(path)

        # read_bundle_range returns the stored range (prefetch short-circuit).
        rng = b.read_bundle_range(model, lat, lon, w, h, dt, kind)
        assert rng["uom"] == "AQI" and rng["max"] == 300

    def test_serve_emits_binary_header_then_frames(self, tmp_path, monkeypatch):
        from weather_radar.lib import framing as F
        b = self._mk(tmp_path, monkeypatch)
        model, lat, lon, w, h, dt, kind = "pm25", 47.2, -116.9, 10, 10, 3600, "aqi"
        path = b._bundle_path(model, lat, lon, w, h, dt, kind)
        header = {"type": "range", "range": {"uom": "AQI"},
                  "uom": "AQI", "coords": [[-116.9, 47.2], [-116.8, 47.2]],
                  "meta": {"source_mtime": 1.0}}
        frames = [{"time": 1000, "vals": [10, 20]},
                  {"time": 2000, "vals": [11, None]},
                  {"time": 3000, "vals": [12, 22]}]
        b._write_bundle(path, header, iter(frames))

        covered = set()
        out = list(b.serve_bundle_epochs(model, lat, lon, w, h, dt, kind,
                                         [1000, 3000], covered, emit_header=True))
        # Each yielded item is one length-prefixed record. First = grid header,
        # then the two matching frames.
        assert len(out) == 3
        hdr_payload = next(F.iter_records(out[0]))
        hdr = json.loads(hdr_payload.decode("utf-8"))
        assert hdr["type"] == "grid"
        assert hdr["coords"] == [[-116.9, 47.2], [-116.8, 47.2]]
        assert hdr["enc"] == "i16"
        # Decode the frame records back to values.
        got = {}
        for rec in out[1:]:
            pl = next(F.iter_records(rec))
            epoch = F.decode_frame_epoch(pl)
            import struct
            n = (len(pl) - 8) // 2
            ints = struct.unpack_from("<%dh" % n, pl, 8)
            got[epoch] = [None if iv == hdr["sentinel"]
                          else iv / hdr["scale"] + hdr["offset"] for iv in ints]
        assert sorted(got) == [1000, 3000]
        assert got[1000] == [10.0, 20.0]
        assert got[3000] == [12.0, 22.0]
        assert covered == {1000, 3000}

    def test_serve_without_header(self, tmp_path, monkeypatch):
        from weather_radar.lib import framing as F
        b = self._mk(tmp_path, monkeypatch)
        model, lat, lon, w, h, dt, kind = "pm25", 47.2, -116.9, 10, 10, 3600, "aqi"
        path = b._bundle_path(model, lat, lon, w, h, dt, kind)
        header = {"type": "range", "range": {}, "uom": "AQI",
                  "coords": [[-116.9, 47.2]], "meta": {"source_mtime": 1.0}}
        b._write_bundle(path, header, iter([{"time": 2000, "vals": [11]}]))
        covered = set()
        out = list(b.serve_bundle_epochs(model, lat, lon, w, h, dt, kind,
                                         [2000], covered, emit_header=False))
        # No header record; just the one frame.
        assert len(out) == 1
        pl = next(F.iter_records(out[0]))
        assert F.decode_frame_epoch(pl) == 2000
        assert covered == {2000}

    def test_serve_missing_bundle_yields_nothing(self, tmp_path, monkeypatch):
        b = self._mk(tmp_path, monkeypatch)
        covered = set()
        out = list(b.serve_bundle_epochs("pm25", 1, 2, 10, 10, 3600, "aqi",
                                         [1000], covered, emit_header=True))
        assert out == []
        assert covered == set()


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
