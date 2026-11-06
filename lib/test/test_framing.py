"""Tests for the binary-sidecar framing codec (weather_radar.lib.framing)."""
import json
import struct

from weather_radar.lib import framing as F


class TestHeader:
    def test_header_roundtrip(self):
        coords = [[-116.9, 47.2], [-116.8, 47.2]]
        scale, offset = F.choose_scale("degF")
        hdr = json.loads(F.encode_header("degF", coords, scale, offset).decode())
        assert hdr["type"] == "grid"
        assert hdr["enc"] == "i16"
        assert hdr["coords"] == coords
        assert hdr["scale"] == 10.0 and hdr["offset"] == 0.0
        assert hdr["sentinel"] == F.SENTINEL

    def test_choose_scale(self):
        assert F.choose_scale("in") == (100.0, 0.0)
        assert F.choose_scale("inches") == (100.0, 0.0)
        assert F.choose_scale("mm") == (100.0, 0.0)
        assert F.choose_scale("degF") == (10.0, 0.0)
        assert F.choose_scale("AQI") == (10.0, 0.0)
        assert F.choose_scale("") == (10.0, 0.0)


class TestFrame:
    def _decode(self, payload, scale, offset, sentinel=F.SENTINEL):
        epoch = F.decode_frame_epoch(payload)
        n = (len(payload) - 8) // 2
        ints = struct.unpack_from("<%dh" % n, payload, 8)
        vals = [None if iv == sentinel else iv / scale + offset for iv in ints]
        return epoch, vals

    def test_frame_roundtrip_with_null(self):
        scale, offset = 10.0, 0.0
        payload = F.encode_frame(1700000000, [12.3, None, -5.0], scale, offset)
        assert len(payload) == F.frame_payload_len(3)
        epoch, vals = self._decode(payload, scale, offset)
        assert epoch == 1700000000
        assert vals[0] == 12.3
        assert vals[1] is None
        assert vals[2] == -5.0

    def test_precip_scale_precision(self):
        scale, offset = F.choose_scale("in")
        payload = F.encode_frame(1, [0.07, 1.25], scale, offset)
        _, vals = self._decode(payload, scale, offset)
        assert vals == [0.07, 1.25]

    def test_clamp_to_int16(self):
        # A value beyond the int16*scale range clamps, not overflows.
        scale, offset = 10.0, 0.0
        payload = F.encode_frame(1, [1e9], scale, offset)
        _, vals = self._decode(payload, scale, offset)
        assert vals[0] == 32767 / scale  # clamped to I16_MAX

    def test_sentinel_collision_nudged(self):
        # A real value that scales exactly onto the sentinel is nudged off it,
        # so it's not misread as "missing".
        scale, offset = 10.0, 0.0
        payload = F.encode_frame(1, [-3276.8], scale, offset)  # *10 == -32768
        epoch, vals = self._decode(payload, scale, offset)
        assert vals[0] is not None  # NOT read as missing


class TestRecords:
    def test_record_prefix_and_iter(self):
        a = b"hello"
        b = b"\x00\x01\x02\x03"
        blob = F.record(a) + F.record(b)
        recs = list(F.iter_records(blob))
        assert recs == [a, b]

    def test_iter_records_ignores_trailing_partial(self):
        # A truncated final record is skipped, not errored.
        blob = F.record(b"ok") + F._LEN.pack(100) + b"short"
        recs = list(F.iter_records(blob))
        assert recs == [b"ok"]

    def test_full_stream_shape(self):
        # Header record + 2 frame records, de-framed in order.
        scale, offset = F.choose_scale("AQI")
        coords = [[-116.9, 47.2], [-116.8, 47.2]]
        stream = (F.record(F.encode_header("AQI", coords, scale, offset))
                  + F.record(F.encode_frame(1000, [10, 20], scale, offset))
                  + F.record(F.encode_frame(2000, [None, 30], scale, offset)))
        recs = list(F.iter_records(stream))
        assert len(recs) == 3
        hdr = json.loads(recs[0].decode())
        assert hdr["type"] == "grid"
        assert F.decode_frame_epoch(recs[1]) == 1000
        assert F.decode_frame_epoch(recs[2]) == 2000
