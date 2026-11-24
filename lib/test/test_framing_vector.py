"""encode_frame was a Python loop per value; it is vectorized now. The BYTES must not change
(stored bundles and the client decoder depend on them)."""
import struct

import numpy as np
import pytest

from weather_radar.lib import framing

I16_MIN, I16_MAX, SENT = -32768, 32767, framing.SENTINEL


def reference(epoch, vals, scale, offset, sentinel=SENT):
    """The previous implementation, verbatim."""
    buf = bytearray(8 + len(vals) * 2)
    struct.pack_into(">q", buf, 0, int(epoch))
    off = 8
    for v in vals:
        if v is None:
            iv = sentinel
        else:
            iv = round((v - offset) * scale)
            if iv < I16_MIN:
                iv = I16_MIN
            elif iv > I16_MAX:
                iv = I16_MAX
            if iv == sentinel:
                iv = sentinel + 1
        struct.pack_into("<h", buf, off, iv)
        off += 2
    return bytes(buf)


EDGE = [0.0, -0.0, 0.25, 0.35, 0.45, 2.5, 3.5, -2.5, -3.5, 12.349999, -3276.8, -3276.7, 3276.7, 3276.8, 1e6, -1e6,
        -3276.75, 3276.75, 0.05, 0.15]


@pytest.mark.parametrize("scale,offset", [(10.0, 0.0), (100.0, 0.0), (10.0, 5.0), (1.0, -3.0)])
def test_edge_values_encode_to_the_same_bytes(scale, offset):
    vals = EDGE + [None, None] + EDGE[::-1]
    assert framing.encode_frame(1_700_000_000, vals, scale, offset) == reference(1_700_000_000, vals, scale, offset)


def test_random_frames_encode_to_the_same_bytes():
    rng = np.random.default_rng(3)
    for scale in (10.0, 100.0):
        vals = rng.normal(40, 60, 5000).tolist()
        for i in rng.integers(0, 5000, 300):
            vals[i] = None
        assert framing.encode_frame(1_700_000_123, vals, scale, 0.0) == reference(1_700_000_123, vals, scale, 0.0)


def test_a_float_array_with_nan_means_the_same_as_a_list_with_none():
    vals = [1.5, None, -4.25, 99.0, None]
    arr = np.array([1.5, np.nan, -4.25, 99.0, np.nan])
    assert framing.encode_frame(5, arr, 10.0, 0.0) == framing.encode_frame(5, vals, 10.0, 0.0) == reference(5, vals, 10.0, 0.0)


def test_ints_and_empty_frames():
    assert framing.encode_frame(9, [1, 2, 300, None], 10.0, 0.0) == reference(9, [1, 2, 300, None], 10.0, 0.0)
    assert framing.encode_frame(9, [], 10.0, 0.0) == reference(9, [], 10.0, 0.0)


def test_no_real_value_ever_decodes_as_the_sentinel():
    out = framing.encode_frame(1, [-3276.8, -1e9, 5.0], 10.0, 0.0)
    got = struct.unpack("<3h", out[8:])
    assert got[0] != SENT and got[1] != SENT and got[2] == 50


def test_a_large_frame_is_fast():
    import time
    vals = np.random.default_rng(0).normal(40, 20, 44_440)
    t0 = time.perf_counter()
    for _ in range(20):
        framing.encode_frame(1, vals, 10.0, 0.0)
    assert (time.perf_counter() - t0) / 20 < 0.01       # the loop took tens of ms for this size
