"""Canned SAMPLE DATA for the map and graph pages, served instead of the real
pipeline when a request carries ``fixture=1``.

Why: the Home Assistant preview pages (/ha-preview/, /graph-preview/) show the
real pages in an iframe, and re-render them as you change settings. Going through
the real pipeline for that (NBM/NAQFC grid parse, area fits, spline evaluation —
seconds to tens of seconds) is far too slow for tweaking layout. With
``fixture=1`` the same endpoints answer instantly with smooth, deterministic
synthetic fields in the exact wire formats the pages consume, so only the page
itself is being exercised.

Scope and safety: only ``/api/models/...`` GETs the pages actually make; no real
data is read or written; it sits INSIDE the access-control middleware (a token is
still required); and it can be switched off with WR_PREVIEW_FIXTURES=0.

The data is synthetic — a sensible-looking field per layer (units/ranges follow
the real layers), a 72 h window starting at the top of the current hour, values
that vary smoothly in space and time (accumulations rise monotonically). It is
NOT a forecast.
"""

import gzip
import json
import math
import os
import time
from datetime import datetime, timezone

from weather_radar.lib import framing
from weather_radar.lib.frametimes import frame_times

PARAM = "fixture"
HOURS = 72
LATTICE = 20          # points per side of the synthetic lattice

# attribute -> (units, low, high, accumulates)
_SPECS = {
    "temperature": ("degF", -5.0, 70.0, False),
    "dewpoint": ("degF", -10.0, 55.0, False),
    "apparent-temperature": ("degF", -15.0, 70.0, False),
    "relative-humidity": ("percent", 15.0, 100.0, False),
    "sky-cover": ("percent", 0.0, 100.0, False),
    "probability-of-precipitation": ("percent", 0.0, 100.0, False),
    "wind-speed": ("mph", 0.0, 30.0, False),
    "wind-gust": ("mph", 0.0, 45.0, False),
    "wind-direction": ("degrees", 0.0, 360.0, False),
    "visibility": ("mi", 0.0, 10.0, False),
    "quantitative-precipitation": ("in", 0.0, 1.5, True),
    "snowfall-amount": ("in", 0.0, 12.0, True),
    "ice-accumulation": ("in", 0.0, 0.2, True),
    "pm25": ("AQI", 15.0, 140.0, False),
    "ozone": ("AQI", 15.0, 120.0, False),
}


def enabled():
    return os.environ.get("WR_PREVIEW_FIXTURES", "1") != "0"


def _spec(attribute, kind=None):
    a = (attribute or "").lower()
    base = a[: -len("-nowcast")] if a.endswith("-nowcast") else a
    if base in _SPECS:
        return _SPECS[base]
    return ("", 0.0, 100.0, kind == "accumulation")


def _window():
    start = int(time.time()) // 3600 * 3600
    return start, start + HOURS * 3600


def _iso(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def _value(spec, lat, lon, epoch, start):
    """A smooth synthetic value: varies in space (a fixed pattern) and in time."""
    _uom, lo, hi, accum = spec
    spatial = 0.5 + 0.5 * math.sin(lon * 1.7 + lat * 2.3)
    hours = max(0.0, (epoch - start) / 3600.0)
    if accum:                                          # rises monotonically through the window
        return round(hi * spatial * (hours / HOURS) ** 1.3, 3)
    temporal = 0.5 + 0.5 * math.sin(hours / 9.0 + lon * 0.9 - lat * 0.6)
    return lo + (hi - lo) * (0.55 * temporal + 0.45 * spatial)


def lattice(lat, lon, width_km, height_km, n=LATTICE):
    """n x n [lon, lat] points covering the requested area."""
    dlat = height_km / 111.32
    dlon = width_km / (111.32 * max(0.2, math.cos(math.radians(lat))))
    return [[lon - dlon / 2 + dlon * i / (n - 1), lat - dlat / 2 + dlat * j / (n - 1)]
            for j in range(n) for i in range(n)]


def _num(q, key, default):
    try:
        return float(q[key])
    except (KeyError, TypeError, ValueError):
        return default


def _json(obj, status=200):
    return status, {}, json.dumps(obj).encode(), "application/json"


def _range(attribute, kind, q):
    spec = _spec(attribute, kind)
    uom, lo, hi, accum = spec
    start, end = _window()
    pts = lattice(_num(q, "lat", 0), _num(q, "lon", 0), _num(q, "width", 1), _num(q, "height", 1))
    lons, lats = [p[0] for p in pts], [p[1] for p in pts]
    padx, pady = (max(lons) - min(lons)) * 0.05, (max(lats) - min(lats)) * 0.05
    if uom == "AQI":
        lo, hi = 0.0, 300.0                               # the fixed AQI display scale
    return _json({"min": lo, "max": hi, "uom": uom, "start": _iso(start), "end": _iso(end),
                  "timeZone": None,
                  "extent": {"w": min(lons) - padx, "e": max(lons) + padx,
                             "s": min(lats) - pady, "n": max(lats) + pady}})


def _stream(attribute, kind, q):
    """The binary-sidecar frame stream (see weather_radar.lib.framing), gzipped
    exactly as the real endpoints send it."""
    spec = _spec(attribute, kind)
    uom = spec[0]
    start, end = _window()
    lat, lon = _num(q, "lat", 0), _num(q, "lon", 0)
    coords = lattice(lat, lon, _num(q, "width", 1), _num(q, "height", 1))
    def opt(k):
        return float(q[k]) if q.get(k) not in (None, "") else None
    epochs = [e for e in frame_times(opt("start"), opt("end"), int(_num(q, "dt", 0)), q.get("times"))
              if start <= e <= end]
    scale, offset = framing.choose_scale(uom)
    body = framing.record(framing.encode_header(uom, coords, scale, offset))
    for e in epochs:
        vals = [_value(spec, c[1], c[0], e, start) for c in coords]
        body += framing.record(framing.encode_frame(e, vals, scale, offset))
    return 200, {"Content-Encoding": "gzip"}, gzip.compress(body, compresslevel=3), "application/octet-stream"


def _graph_series(kind, param, q):
    spec = _spec(param, kind)
    start, end = _window()
    step = int(_num(q, "dt", 0))
    step = step if step > 0 else 3600
    lat, lon = _num(q, "lat", 0), _num(q, "lon", 0)
    feats = [{"type": "Feature",
              "properties": {param: round(_value(spec, lat, lon, t, start), 1), "time": t * 1000, "uom": spec[0]},
              "geometry": {"type": "Point", "coordinates": [lon, lat]}}
             for t in range(start, end + 1, step)]
    return _json({"type": "FeatureCollection", "features": feats})


def respond(path, query):
    """Sample-data answer for an API request, as (status, headers, body, media_type);
    None when the path isn't one the pages use for data (so it falls through to
    the real handler). `query` is a plain dict of the request's query parameters."""
    parts = [p for p in path.split("/") if p]
    if len(parts) < 3 or parts[0] != "api" or parts[1] != "models":
        return None
    rest = parts[2:]
    if rest == ["prefetch", "stream"]:
        return 200, {}, (json.dumps({"phase": "gridpoints", "done": 1, "total": 1}) + "\n").encode(), "application/x-ndjson"
    if rest == ["prefetch"]:
        return 204, {}, b"", "application/json"
    if rest == ["aqi", "range"]:
        return _range(query.get("pollutant", "pm25"), "aqi", query)
    if rest == ["aqi", "stream"]:
        return _stream(query.get("pollutant", "pm25"), "aqi", query)
    if len(rest) == 3 and rest[1:] == ["type", "range"]:
        return _range(rest[0], query.get("kind"), query)
    if len(rest) == 4 and rest[1:3] == ["type", "series"] and rest[3] == "stream":
        return _stream(rest[0], query.get("kind"), query)
    if len(rest) == 3 and rest[1] == "params":
        return _graph_series(rest[0], rest[2], query)
    return None


def install(server):
    """Answer ``fixture=1`` API requests from sample data. Must be installed
    BEFORE the access-control middleware so that one wraps it (the last middleware
    added is the outermost): sample data is still behind the token."""
    from starlette.responses import Response

    @server.middleware("http")
    async def _fixtures(request, call_next):
        if (request.query_params.get(PARAM) == "1" and request.url.path.startswith("/api/")
                and enabled()):
            r = respond(request.url.path, dict(request.query_params))
            if r is not None:
                status, headers, body, media = r
                return Response(content=body, status_code=status, headers=headers, media_type=media)
        return await call_next(request)
