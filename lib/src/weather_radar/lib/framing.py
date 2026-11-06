"""Binary sidecar framing for animated frame streams + bundles.

The animated layers stream a FIXED coordinate lattice once (a header) then one
value array per frame. Encoding those value arrays as JSON (`[12,13,null,...]`)
costs ~3-5 bytes/value and a full JSON.parse of an N-element array per frame on
the client. This module replaces the value arrays with a compact binary
encoding: fixed 2 bytes/value (int16), no punctuation, decoded on the client as
a typed-array view (microseconds, no per-value JS object allocation).

Stream = a sequence of length-prefixed RECORDS, gzipped as a whole:

    record := [4-byte big-endian uint32 length][payload of that length]

  * record 0 is the HEADER: UTF-8 JSON
        {"type":"grid","uom":str,"coords":[[lon,lat],...],
         "enc":"i16","scale":float,"offset":float,"sentinel":int}
    `coords` is the fixed lattice (sent once); N = len(coords). Each frame's
    value array has exactly N entries in this order.
  * records 1..M are FRAMES: a binary blob
        [8-byte big-endian int64 epoch seconds][N × int16 little-endian values]
    Each stored value is round((v - offset) * scale), clamped to int16; a
    missing value (out of horizon / no data, i.e. JSON null) is `sentinel`.

Why int16 + scale/offset: AQI (0..500), temperature degF (~-60..130),
humidity/sky %, wind mph, precip/snow inches all fit int16 with a per-layer
scale (e.g. scale=10 keeps 0.1 precision; scale=100 keeps 0.01" precip). The
header carries enc/scale/offset/sentinel so the client decodes generically.

Both the live stream endpoints and the on-disk bundles use this framing, so a
bundle's bytes can be streamed to the client with only a recompress (same as
the JSON bundles were).
"""

import json
import struct

# int16 range; one value reserved as the "no data" sentinel (JSON null).
_I16_MIN = -32768
_I16_MAX = 32767
SENTINEL = _I16_MIN            # -32768 => missing point

ENC_I16 = "i16"

_LEN = struct.Struct(">I")     # 4-byte big-endian record length prefix
_EPOCH = struct.Struct(">q")   # 8-byte big-endian int64 epoch


def choose_scale(uom):
    """Pick a fixed-point scale for a unit of measure so int16 keeps useful
    precision without overflowing. Returns (scale, offset).

    The stored int is round((v - offset) * scale); value = stored/scale+offset.
    Chosen conservatively so the scaled range stays within +-32767:
      * precip/snow/ice inches: scale=100 (0.01" precision; up to ~327").
      * everything else (temp, AQI, %, wind, pressure, visibility): scale=10
        (0.1 precision; range +-3276), which covers all those comfortably.
    offset is 0 for all current layers (none need a bias to fit int16).
    """
    u = (uom or "").lower()
    if u in ("in", "inch", "inches", "mm"):
        return 100.0, 0.0
    return 10.0, 0.0


def encode_header(uom, coords, scale, offset, sentinel=SENTINEL):
    """Serialize the grid header record payload (bytes, no length prefix)."""
    obj = {
        "type": "grid",
        "uom": uom or "",
        "coords": coords,
        "enc": ENC_I16,
        "scale": scale,
        "offset": offset,
        "sentinel": sentinel,
    }
    return json.dumps(obj, separators=(",", ":")).encode("utf-8")


def encode_frame(epoch, vals, scale, offset, sentinel=SENTINEL):
    """Serialize one frame record payload (bytes, no length prefix):
    [8-byte epoch][N × int16 LE]. `vals` is a list aligned to the header coords;
    None entries become the sentinel. Values are scaled/offset and clamped to
    the int16 range."""
    n = len(vals)
    buf = bytearray(8 + n * 2)
    _EPOCH.pack_into(buf, 0, int(epoch))
    off = 8
    for v in vals:
        if v is None:
            iv = sentinel
        else:
            iv = round((v - offset) * scale)
            if iv < _I16_MIN:
                iv = _I16_MIN
            elif iv > _I16_MAX:
                iv = _I16_MAX
            # Never let a real value collide with the sentinel; nudge by 1.
            if iv == sentinel:
                iv = sentinel + 1
        # little-endian int16
        struct.pack_into("<h", buf, off, iv)
        off += 2
    return bytes(buf)


def frame_payload_len(n_points):
    """Byte length of a frame payload for a grid of `n_points`."""
    return 8 + n_points * 2


def record(payload):
    """Prefix a payload with its 4-byte big-endian length -> a full record."""
    return _LEN.pack(len(payload)) + payload


def decode_frame_epoch(payload):
    """Read just the epoch from a frame payload (first 8 bytes)."""
    return _EPOCH.unpack_from(payload, 0)[0]


def iter_records(blob):
    """Yield each record payload (bytes) from a length-prefixed blob. Used to
    read a bundle back off disk."""
    i = 0
    n = len(blob)
    while i + 4 <= n:
        (ln,) = _LEN.unpack_from(blob, i)
        i += 4
        if i + ln > n:
            break
        yield blob[i:i + ln]
        i += ln
