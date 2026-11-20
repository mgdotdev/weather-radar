from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
import json
import logging
import os
import time
import zlib

from weather_radar.server.api.models.utils import datetime_from_time, time_t

from weather_radar.lib.models import model_from_type, ensemble_kinds, weather_source
from weather_radar.lib.models.accumulation import AccumulationEnsemble
from weather_radar.lib.models.observation import ObservationEnsemble
from weather_radar.lib.area import MapCoordinate, CoordinateArea
from weather_radar.lib.models.utils import BoundsError, clamp_value
from weather_radar.lib import framing


def _gzip_stream(chunk_iter):
    """Wrap a byte-chunk generator in a streaming gzip compressor.

    Yields gzip-compressed bytes incrementally: each source chunk is compressed
    with Z_SYNC_FLUSH so the client receives (and can decode) data as it
    arrives — preserving the per-frame streaming / first-frame latency the
    animation pipeline depends on.

    `chunk_iter` yields `bytes` (the binary-sidecar length-prefixed records; see
    weather_radar.lib.framing). The browser's fetch() transparently inflates
    `Content-Encoding: gzip`, so the client only has to de-frame the records."""
    # wbits=31 -> gzip wrapper (not raw deflate / zlib).
    compressor = zlib.compressobj(6, zlib.DEFLATED, 31)
    for chunk in chunk_iter:
        if not chunk:
            continue
        out = compressor.compress(chunk)
        out += compressor.flush(zlib.Z_SYNC_FLUSH)
        if out:
            yield out
    tail = compressor.flush(zlib.Z_FINISH)
    if tail:
        yield tail


# Standard headers for a gzipped binary frame stream (binary sidecar framing).
_GZIP_STREAM_HEADERS = {"Content-Encoding": "gzip"}
_STREAM_MEDIA_TYPE = "application/octet-stream"


logger = logging.getLogger(__name__)

# When set (any truthy value), emit a per-request timing summary for the
# series-stream sweep at WARNING — useful for diagnosing first-frame latency
# (NOAA warm vs. spline fit vs. evaluation). Off by default to avoid per-request
# log noise in normal operation.
_TIMING = bool(os.environ.get("WR_STREAM_TIMING"))

router = APIRouter(prefix="/models")

# Maps the public `kind` query value to the ensemble implementation. The
# attribute each ensemble predicts is supplied separately (the `{model}` path
# segment), so e.g. kind="observation" + model="temperature" produces a
# temperature heatmap, while kind="accumulation" + model="quantitative-precipitation"
# produces a precipitation heatmap.
#
# The CONCRETE classes depend on the configured weather source
# (WR_WEATHER_SOURCE): NBM GRIB (default), NDFD GRIB, or gridpoints JSON.
# ENSEMBLE_KINDS is a function (not a static dict) so a source change is picked
# up without a reimport. Callers resolve with ENSEMBLE_KINDS().get(kind, <default>).
def ENSEMBLE_KINDS():
    return ensemble_kinds()


# True when the active weather source samples a GRIB grid directly (NDFD or
# NBM) rather than the NOAA /gridpoints JSON. These ensembles sample the area's
# lattice points directly (no WFO gridpoint resolution), so the hot series path
# branches on this to iterate the ensemble's points instead of
# area.iter_gridpoints().
def _is_ndfd():
    return weather_source() in ("ndfd", "nbm")


@router.get("/{model}/params/{param}", response_class=JSONResponse)
def models(model: str, param: str, lat: float, lon: float, time: time_t=None, dt:int=0, counts: int=-1):
    """Single-point time series for the forecast graph. `model` here is the
    KIND ("observation"/"accumulation"); `param` is the attribute. Returns a
    GeoJSON FeatureCollection (verbose predict) so the graph can plot value vs.
    time.

    SOURCE-AWARE: this must use the SAME data source as the map, else the graph
    disagrees with the map (e.g. under WR_WEATHER_SOURCE=ndfd the map shows
    NDFD snowfall while the JSON single-point model reads a different —
    possibly empty — /gridpoints series). For NDFD we fit a minimal area at the
    point through the source's ensemble (so accumulation gets the cumulative
    integration too) and evaluate its nearest lattice point."""
    dtime = datetime_from_time(time, dt=dt, counts=counts)
    coord = MapCoordinate(lat, lon)
    from datetime import timedelta

    if model == "aqi" or weather_source() in ("ndfd", "nbm"):
        # AQI (NAQFC) is its own gridded source whatever the weather source is.
        # NDFD/NBM are area/ensemble-based. Sample the point the SAME way the map
        # does: an inverse-distance-weighted blend of the area's lattice points
        # (mirroring the client's buildField IDW), NOT just the single nearest
        # lattice point — the nearest cell may be a no-data (fill) cell that the
        # map's IDW skips, which is why a single-cell graph read 0 while the map
        # showed the blended neighbors' snow. Graph the WHOLE forecast window.
        # Answered from the point-series cache when it holds this data (a file
        # read — no grid parse), else computed and cached for the next request.
        from weather_radar.server.pointseries import point_series
        return point_series(model, param, lat, lon, dt)

    model_class = model_from_type(model)
    m = model_class.from_map_coordinate(coord, param)
    fc = m.predict(dtime, verbose=True)
    # Normalize feature timestamps to UTC epoch-ms so the graph renders them in
    # the USER'S browser-local zone (matching the NDFD/NBM path above and the
    # map). The gridpoints models emit `time` as an ISO string (aware local or
    # naive); convert to an absolute-instant epoch-ms here.
    return _features_time_to_epoch_ms(fc)


def _features_time_to_epoch_ms(fc):
    """Rewrite each feature's properties.time (an ISO-8601 string) to UTC epoch
    milliseconds, in place. A naive string (no offset) is assumed to be in the
    server's local zone (that's how the model built it via .astimezone()), so
    it's made aware before converting. Leaves non-string/unparseable values
    untouched. Returns the same `fc` for convenience."""
    from datetime import datetime
    try:
        feats = fc.get("features") or []
    except AttributeError:
        return fc
    for ft in feats:
        props = ft.get("properties") if isinstance(ft, dict) else None
        if not props or "time" not in props:
            continue
        tv = props["time"]
        if not isinstance(tv, str):
            continue  # already epoch-ms (or unexpected) — leave as is
        try:
            d = datetime.fromisoformat(tv)
            if d.tzinfo is None:
                d = d.astimezone()  # assume server-local for a naive stamp
            props["time"] = int(d.timestamp() * 1000)
        except Exception:
            pass
    return fc


# IDW params mirrored from the client raster (map.html buildField), so the
# graph's blended point value matches what the heatmap paints there.
_IDW_K = 12
_IDW_EPS = 0.015 * 0.015


def _point_series(kind, param, lat, lon, dt):
    """Compute the graph series for one point from whichever gridded source
    backs `kind`: the NAQFC air-quality grid for "aqi", else the NBM/NDFD GRIB."""
    if kind == "aqi":
        return _aqi_point_series(param, lat, lon, dt)
    return _grib_point_series(kind, param, lat, lon, dt)


def _aqi_point_series(pollutant, lat, lon, dt):
    """A graph FeatureCollection for one point from the NAQFC air-quality grid:
    the whole forecast window, `dt` apart, each value an inverse-distance blend
    of the nearest lattice points (the SAME blend the map's raster paints, so
    graph and heatmap agree at the pin), clamped >= 0. Values keep one decimal:
    rounding to whole AQI turned the smooth spline into a staircase (a 72 h
    series had 43 distinct values in 569 samples), unlike the temperature
    graph, which plots floats. Empty features when the grid or pollutant is
    unavailable."""
    from weather_radar.lib import aqi
    empty = {"type": "FeatureCollection", "features": []}
    fit = aqi.fit_area_dense(CoordinateArea(MapCoordinate(lat, lon), 1, 1), pollutant)
    if not fit:
        return empty
    near = sorted(
        ((((p[1] - lat) ** 2 + (p[0] - lon) ** 2), sp)
         for p, sp in zip(fit["points"], fit["splines"]) if sp is not None),
        key=lambda t: t[0])[:_IDW_K]
    if not near:
        return empty
    weights = [1.0 / (d2 + _IDW_EPS) for d2, _ in near]
    wsum = sum(weights)
    step = int(dt) if dt and int(dt) > 0 else 3600
    features = []
    t = int(fit["start"])
    while t <= int(fit["end"]):
        num = 0.0
        for w, (_d2, sp) in zip(weights, near):
            y = float(sp(t))
            num += w * (y if y > 0 else 0.0)
        features.append({
            "type": "Feature",
            "properties": {pollutant: round(num / wsum, 1),
                           "time": t * 1000, "uom": "AQI"},
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
        })
        t += step
    return {"type": "FeatureCollection", "features": features}


def _grib_point_series(kind, param, lat, lon, dt):
    """Compute the graph series for one point from the GRIB source (NBM/NDFD):
    the single-point FeatureCollection the graph and forecast pages plot."""
    coord = MapCoordinate(lat, lon)
    ensemble_class = ENSEMBLE_KINDS().get(kind, AccumulationEnsemble)
    area = CoordinateArea(coord, 1, 1)
    ensemble = ensemble_class(area, param)
    return _ndfd_point_series(ensemble, coord, param, dt)


def _ndfd_point_series(ensemble, coord, param, dt):
    """A verbose FeatureCollection time series at `coord`, IDW-blended from the
    NDFD ensemble's lattice points over the forecast window (stepped by dt,
    fallback 1h). Matches the map's interpolation so the graph agrees with the
    heatmap at the pinned point."""
    from datetime import datetime, timedelta

    pts = []  # [(center, f, start_dt, end_dt, units)]
    units = ""
    start_dt = end_dt = None
    try:
        models = list(ensemble.iter_points())
    except Exception:
        models = []
    for m in models:
        try:
            st, et, center, f, uom, _x = m.model
        except Exception:
            continue
        pts.append((center, f, st, et))
        units = units or uom
        # The lattice shares one window; take the widest seen.
        if start_dt is None or st < start_dt:
            start_dt = st
        if end_dt is None or et > end_dt:
            end_dt = et
    if not pts or start_dt is None:
        return {"type": "FeatureCollection", "features": []}

    # Nearest-K points to the pin by squared degree distance (lon/lat), with
    # their normalized inverse-distance (power 2) weights — constant across the
    # whole window (positions don't move), so compute once.
    scored = sorted(
        ((((c.lat - coord.lat) ** 2 + (c.lon - coord.lon) ** 2), c, f, st, et)
         for (c, f, st, et) in pts),
        key=lambda t: t[0],
    )[:_IDW_K]

    step = int(dt) if dt and int(dt) > 0 else 3600
    features = []
    t = start_dt
    guard = 0
    while t <= end_dt and guard < 100000:
        num = 0.0
        den = 0.0
        for d2, c, f, st, et in scored:
            if t < st or t > et:
                continue  # this point out of its window at t
            secs = (t - st).total_seconds()
            try:
                y = f(secs).item()
            except Exception:
                continue
            if y < 0:
                y = 0.0
            w = 1.0 / (d2 + _IDW_EPS)
            num += w * y
            den += w
        if den > 0:
            val = clamp_value(param, num / den)
            # Emit the timestamp as UTC (epoch ms) so the client renders it in
            # the USER'S browser-local zone. `t` is a server-local aware
            # datetime; .timestamp() gives the absolute instant regardless of
            # the server's zone, and *1000 -> ms for a JS Date.
            t_ms = int(t.timestamp() * 1000)
            features.append({
                "type": "Feature",
                "properties": {param: val, "time": t_ms, "uom": units},
                "geometry": {"type": "Point",
                             "coordinates": [coord.lon, coord.lat]},
            })
        t = t + timedelta(seconds=step)
        guard += 1
    return {"type": "FeatureCollection", "features": features}


# --- Point summary (/api/models/point) --------------------------------------
# A compact, source-agnostic "what's the weather at this lat/lon" endpoint,
# meant for machine consumers (Home Assistant REST sensors, scripts) rather
# than the browser map. Unlike /{model}/params/{param} (one attribute, verbose
# GeoJSON FeatureCollection), this returns ALL the common attributes in one
# call as a flat JSON object of {key: {value, uom}}, with an optional `hourly`
# array. It works across every weather source (gridpoints / ndfd / nbm) by
# reusing the same per-source sampling the graph uses.

# Public key -> (kind, kebab-case attribute). The key is the stable name a
# consumer templates against; the attribute is the internal model param. Keys
# are grouped by the ensemble kind that predicts them (observation = snapshot
# values; accumulation = per-interval totals).
_POINT_ATTRIBUTES = {
    "temperature":               ("observation", "temperature"),
    "dewpoint":                  ("observation", "dewpoint"),
    "relative_humidity":         ("observation", "relative-humidity"),
    "sky_cover":                 ("observation", "sky-cover"),
    "wind_speed":                ("observation", "wind-speed"),
    "wind_gust":                 ("observation", "wind-gust"),
    "probability_of_precipitation": ("observation", "probability-of-precipitation"),
    "quantitative_precipitation": ("accumulation", "quantitative-precipitation"),
    "snowfall_amount":           ("accumulation", "snowfall-amount"),
    "ice_accumulation":          ("accumulation", "ice-accumulation"),
}


def _point_sampler(kind, param, coord):
    """Return (sample_fn, uom, start_dt, end_dt) for one (kind, param) at
    `coord`, or None if the active source has no data for it.

    `sample_fn(when: datetime) -> float|None` evaluates the attribute at a
    time, source-aware: gridpoints uses the single resolved gridpoint's fitted
    spline; ndfd/nbm uses the IDW blend of the area's lattice points (matching
    the map / graph). Clamps to the attribute's physical range and returns None
    when `when` is outside the forecast window or the point can't be sampled.
    """
    from datetime import datetime

    if weather_source() in ("ndfd", "nbm"):
        ensemble_class = ENSEMBLE_KINDS().get(kind, AccumulationEnsemble)
        area = CoordinateArea(coord, 1, 1)
        ensemble = ensemble_class(area, param)
        try:
            models = list(ensemble.iter_points())
        except Exception:
            return None
        pts = []  # (d2, f, start_dt, end_dt)
        uom = ""
        start_dt = end_dt = None
        for m in models:
            try:
                st, et, center, f, u, _x = m.model
            except Exception:
                continue
            d2 = (center.lat - coord.lat) ** 2 + (center.lon - coord.lon) ** 2
            pts.append((d2, f, st, et))
            uom = uom or u
            if start_dt is None or st < start_dt:
                start_dt = st
            if end_dt is None or et > end_dt:
                end_dt = et
        if not pts or start_dt is None:
            return None
        scored = sorted(pts, key=lambda t: t[0])[:_IDW_K]

        def sample(when):
            if when < start_dt or when > end_dt:
                return None
            num = den = 0.0
            for d2, f, st, et in scored:
                if when < st or when > et:
                    continue
                try:
                    y = f((when - st).total_seconds()).item()
                except Exception:
                    continue
                if y < 0:
                    y = 0.0
                w = 1.0 / (d2 + _IDW_EPS)
                num += w * y
                den += w
            if den <= 0:
                return None
            return clamp_value(param, num / den)

        return sample, uom, start_dt, end_dt

    # gridpoints (default): resolve the single gridpoint and fit its spline.
    model_class = model_from_type(kind)
    try:
        m = model_class.from_map_coordinate(coord, param)
        start_dt, end_dt, _center, f, uom, _expiry = m.model
    except Exception:
        return None

    def sample(when):
        when_local = when.astimezone()
        if when_local < start_dt or when_local > end_dt:
            return None
        try:
            y = f((when_local - start_dt).total_seconds()).item()
        except Exception:
            return None
        return clamp_value(param, y)

    return sample, uom, start_dt, end_dt


@router.get("/point", response_class=JSONResponse)
def point(lat: float, lon: float, hours: int = 0, step: int = 3600,
          attrs: str = ""):
    """Compact current-conditions (and optional hourly) summary at a point.

    Query params:
      lat, lon  : required coordinate.
      hours     : if >0, also include an `hourly` array covering the next
                  `hours` hours (capped at 168 / 7 days).
      step      : hourly sampling step in seconds (default 3600; min 600).
      attrs     : optional comma-separated subset of attribute keys (e.g.
                  "temperature,probability_of_precipitation"); default is all.

    Returns:
      {
        "meta": {"lat","lon","source","generated","window":{"start","end"}},
        "current": {"temperature": {"value": 54.2, "uom": "degF"}, ...},
        "hourly":  [ {"time": "...", "values": {"temperature": {...}, ...}}, ... ]
      }
    Attributes the active source can't provide are present with a null value
    (graceful no-op), so the response shape is stable. Values are numbers in
    the source's native unit (`uom`); temperature is degF (the model's
    built-in degC->degF wrapper), others are NOAA/GRIB native (e.g. PoP %,
    QPF mm). A point outside NOAA coverage yields all-null values with a 200
    (not a 404) so a consumer can distinguish "no coverage" from an error.
    """
    from datetime import datetime, timedelta

    coord = MapCoordinate(lat, lon)
    now = datetime.now()

    # Resolve the requested attribute subset (default: all), preserving a
    # stable key order. Unknown keys are ignored.
    if attrs.strip():
        requested = [k.strip() for k in attrs.split(",") if k.strip()]
        keys = [k for k in _POINT_ATTRIBUTES if k in requested]
    else:
        keys = list(_POINT_ATTRIBUTES)

    step = max(int(step or 3600), 600)
    hours = max(0, min(int(hours or 0), 168))

    # Build each attribute's sampler once, then evaluate at the needed times.
    samplers = {}           # key -> (sample_fn, uom)
    win_start = win_end = None
    for key in keys:
        kind, param = _POINT_ATTRIBUTES[key]
        got = _point_sampler(kind, param, coord)
        if got is None:
            samplers[key] = None
            continue
        sample_fn, uom, s_dt, e_dt = got
        samplers[key] = (sample_fn, uom)
        # Track the intersection-ish overall window for meta (widest union).
        if win_start is None or s_dt < win_start:
            win_start = s_dt
        if win_end is None or e_dt > win_end:
            win_end = e_dt

    def _eval_at(when):
        out = {}
        for key in keys:
            s = samplers.get(key)
            if s is None:
                out[key] = {"value": None, "uom": None}
                continue
            sample_fn, uom = s
            try:
                val = sample_fn(when)
            except Exception:
                val = None
            out[key] = {"value": val, "uom": uom or None}
        return out

    current = _eval_at(now)

    hourly = []
    if hours:
        t = now
        end = now + timedelta(hours=hours)
        guard = 0
        while t <= end and guard < 10000:
            hourly.append({"time": t.astimezone().isoformat(),
                           "values": _eval_at(t)})
            t = t + timedelta(seconds=step)
            guard += 1

    return {
        "meta": {
            "lat": lat,
            "lon": lon,
            "source": weather_source(),
            "generated": now.astimezone().isoformat(),
            "window": {
                "start": win_start.astimezone().isoformat() if win_start else None,
                "end": win_end.astimezone().isoformat() if win_end else None,
            },
        },
        "current": current,
        "hourly": hourly,
    }


@router.get("/prefetch", status_code=204)
def prefetch(lat: float, lon: float, width: float|None=None, height: float|None=None):
    coord = MapCoordinate(lat, lon)
    if width or height:
        area = CoordinateArea(coord, width, height)
        area.cache()


@router.get("/prefetch/stream")
def prefetch_stream(lat: float, lon: float, width: float=1, height: float=1,
                    model: str=None, dt: int=0, kind: str="accumulation"):
    """Stream prefetch progress as NDJSON (one JSON object per line) while the
    area's data is fetched from NOAA. Each line is
    {"phase": "points"|"gridpoints", "done": N, "total": M} so the client can
    show a live completed/total fraction. Phases run in order; the final line
    of each phase has done == total.

    BUNDLE SHORT-CIRCUIT: if a precomputed bundle exists for (model, area,
    kind) — a WATCHED url — the area's NOAA data need not be resolved/fetched
    at all (every frame is already computed in the bundle). In that case we
    emit a single {"phase":"bundle","range":{...}} line carrying the bundle's
    color range + forecast window and return immediately — the client then
    skips /type/range and streams the precomputed frames. This is what makes a
    watched url jump straight to animation (no resolve, no fetch, no range
    round-trip)."""
    coord = MapCoordinate(lat, lon)
    area = CoordinateArea(coord, width, height)

    def generate():
        # Watched-url fast path: a matching bundle means no resolve/fetch.
        if model is not None:
            try:
                from weather_radar.server.bundle import read_bundle_range
                rng = read_bundle_range(model, lat, lon, width, height, kind)
                if rng is not None:
                    yield json.dumps({"phase": "bundle", "range": rng}) + "\n"
                    return
            except Exception:
                pass  # fall through to the normal resolve/fetch

        # GRIB grid source (NDFD or NBM): there are no NOAA /points or
        # /gridpoints to resolve — the national element GRIB is fetched
        # (writer-owned) and sampled in the fit. Warm the CORRECT source's grid
        # here so the subsequent /type/range + series stream are fast, and
        # report a single-phase completion so the client's prefetch progress
        # bar still advances. (Dispatch by weather_source(): using the wrong
        # module warms the wrong backend — e.g. under NBM it would 404-fetch
        # NDFD's day-4-7 ds.snow.bin, which NDFD doesn't publish.)
        if _is_ndfd() and model is not None:
            try:
                if weather_source() == "nbm":
                    from weather_radar.lib import nbm as _grid_src
                else:
                    from weather_radar.lib import ndfd as _grid_src
                _grid_src._load_grid(model)  # download (writer) + parse + cache
            except Exception:
                pass
            yield json.dumps({"phase": "gridpoints", "done": 1, "total": 1}) + "\n"
            return

        for progress in area.iter_cache():
            yield json.dumps(progress) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@router.get("/{model}/type/ensemble", response_class=JSONResponse)
def ensemble(
    model: str,
    lat: float,
    lon: float,
    time: time_t=None,
    dt: int=0,
    width: float=1,
    height: float=1,
    kind: str="accumulation",
):
    dtime = datetime_from_time(time)
    coord = MapCoordinate(lat, lon)
    area = CoordinateArea(coord, width, height)
    ensemble_class = ENSEMBLE_KINDS().get(kind, AccumulationEnsemble)
    # Compute from the (NOAA-cached) gridpoint data. The interpolation itself
    # isn't cached server-side — the browser caches results (apiClient), and
    # the NOAA source data is kept warm by the writer/hotcache.
    return ensemble_class(area, model).predict(dtime, dt=dt, verbose=True)


# --- Air Quality (NAQFC) ----------------------------------------------------
# AQI is a separate data path from the NOAA weather layers: a gridded, hourly
# NAQFC forecast sampled directly (no /points gridpoint resolution, no spline
# fitting). It still emits the SAME wire shapes the client consumes — a range
# payload and slim {time,uom,coords,vals} frame lines — so it reuses the
# client's range/fill/scrubber machinery unchanged.

# Fixed AQI scale: the EPA index is 0..500 with standard category thresholds,
# so (unlike temperature) the color range is constant, not auto-scaled.
AQI_MIN, AQI_MAX = 0, 300  # cap the displayed scale at 300 (Very Unhealthy);
                           # values above still compute, color clamps at top.


@router.get("/aqi/range", response_class=JSONResponse)
def aqi_range(lat: float, lon: float, width: float = 1, height: float = 1,
              pollutant: str = "pm25"):
    """Fixed color range + forecast window for the AQI (NAQFC) layer. The AQI
    scale is constant (0..300 displayed), so min/max are fixed; start/end are
    the first/last forecast-hour epochs of the NAQFC series.

    Prefers a precomputed bundle's stored range header (cheap, no fit) when one
    exists for these params; else fits the area to derive the window."""
    from datetime import datetime
    from weather_radar.lib.aqi import fit_area
    # Bundle header first (watched url) — the stored range incl. start/end.
    try:
        from weather_radar.server.bundle import read_bundle_range
        rng = read_bundle_range(pollutant, lat, lon, width, height, "aqi")
        if rng:
            return rng
    except Exception:
        pass
    coord = MapCoordinate(lat, lon)
    area = CoordinateArea(coord, width, height)
    fit = fit_area(area, pollutant)
    if fit is None:
        return JSONResponse({"min": AQI_MIN, "max": AQI_MAX, "uom": "AQI",
                             "start": None, "end": None})
    start = datetime.fromtimestamp(fit["start"]).isoformat()
    end = datetime.fromtimestamp(fit["end"]).isoformat()
    return {"min": AQI_MIN, "max": AQI_MAX, "uom": "AQI",
            "start": start, "end": end}


@router.get("/aqi/stream")
def aqi_stream(lat: float, lon: float, width: float = 1, height: float = 1,
               pollutant: str = "pm25", start: float = None, end: float = None,
               dt: int = 0, times: str = None):
    """Stream the AQI forecast as NDJSON frames —
    {"time", "uom":"AQI", "coords":[[lon,lat],...], "vals":[aqi,...]} — one per
    requested epoch, byte-compatible with the weather series stream so the
    client's fill reader consumes it unchanged.

    The area's hourly NAQFC series is fit to a per-gridpoint CubicSpline ONCE
    (fit_area), then evaluated at the requested epochs — so, like the NOAA
    layers, ANY `dt` scrubs smoothly (sub-hourly values are spline-interpolated
    between the hourly NAQFC steps). Epochs come from start/end/dt (sequence)
    and/or an explicit `times` batch, exactly like the weather series.

    BUNDLE FAST PATH: if a precomputed AQI bundle exists for these params (a
    watched url, built in the background), its frames are served straight from
    disk — zero spline fit/eval — and only epochs the bundle lacks fall through
    to the live spline path. Same mechanism as /series/stream."""
    from weather_radar.lib.aqi import fit_area_dense, predict_area, grid_points
    coord = MapCoordinate(lat, lon)
    area = CoordinateArea(coord, width, height)
    epochs = _frame_times(start, end, dt, times)

    def generate():
        if not epochs:
            return
        # Bundle first (keyed by kind="aqi", model=pollutant). Its layers say
        # which requested epochs are stored; those are sent first, in time order
        # (plain disk reads, instant, before the slow cold fit), then the rest
        # are computed in the order the client asked for (nearest the playhead
        # first). The bundle supplies the (binary) header when it holds
        # anything; otherwise the live path emits one.
        remaining = list(epochs)
        header_sent = False
        feed = None
        try:
            from weather_radar.server.bundle import open_bundle_feed
            feed = open_bundle_feed(pollutant, lat, lon, width, height, "aqi", epochs)
        except Exception:
            feed = None
        if feed is not None:
            header_sent = True
            yield feed.header_record()
            yield from feed.upto(None)
            remaining = [e for e in epochs if e not in feed.held]
            if not remaining:
                return  # entire request served from the bundle

        # Live spline path for whatever the bundle didn't hold.
        fit = fit_area_dense(area, pollutant)
        if fit is None:
            return
        # BINARY SIDECAR: header (coords + enc/scale) once, then binary frames.
        # AQI is an integer index; scale=10/offset=0 (from choose_scale("AQI"))
        # is lossless for whole-number AQI and matches the bundle's encoding.
        scale, offset = framing.choose_scale("AQI")
        if not header_sent:
            yield framing.record(
                framing.encode_header("AQI", grid_points(fit), scale, offset))
        # Keep the exact bytes streamed so they can be saved as a bundle layer
        # (read next time instead of re-fit). Token = the NAQFC cycle the fit
        # just loaded.
        new_items = []
        for fr in predict_area(fit, remaining, fixed_grid=True):
            payload = framing.encode_frame(fr["epoch"], fr["vals"], scale, offset)
            new_items.append((int(fr["epoch"]), payload))
            yield framing.record(payload)
        try:
            from datetime import datetime as _dt
            from weather_radar.server.bundle import (
                save_request_layer, _aqi_token_in_use)
            save_request_layer(
                pollutant, lat, lon, width, height, "aqi", "AQI",
                grid_points(fit), new_items, _aqi_token_in_use(pollutant),
                rng={"min": AQI_MIN, "max": AQI_MAX, "uom": "AQI",
                     "start": _dt.fromtimestamp(fit["start"]).isoformat(),
                     "end": _dt.fromtimestamp(fit["end"]).isoformat(),
                     "timeZone": None})
        except Exception:
            logger.warning("AQI bundle layer save skipped", exc_info=True)

    return StreamingResponse(_gzip_stream(generate()),
                             media_type=_STREAM_MEDIA_TYPE,
                             headers=_GZIP_STREAM_HEADERS)



@router.get("/{model}/type/range", response_class=JSONResponse)
def ensemble_range(
    model: str,
    lat: float,
    lon: float,
    width: float=1,
    height: float=1,
    kind: str="accumulation",
):
    """Global min/max of the predicted value over the whole forecast
    lifecycle for the area, plus the forecast time window (start/end), so the
    client can fix its color scale and loop the animation over the valid
    range. Fast once the area is cached (call /prefetch first)."""
    coord = MapCoordinate(lat, lon)
    area = CoordinateArea(coord, width, height)
    ensemble_class = ENSEMBLE_KINDS().get(kind, AccumulationEnsemble)
    ensemble = ensemble_class(area, model)
    result = ensemble.value_range() or {
        "min": None, "max": None, "uom": None, "start": None, "end": None
    }
    # Include the center's IANA timezone so the client can display times in
    # the map location's local zone rather than the browser's.
    result["timeZone"] = area.time_zone
    # Painted-field extent: the bounding box of the sampled lattice points plus
    # the SAME 5% pad the client's IDW raster applies (see buildField). The
    # client uses this to size the area-outline box correctly from the FIRST
    # render, instead of first drawing the nominal width×height rectangle and
    # then snapping to the data extent when the first frame paints (a visible
    # resize when the animation starts). Cheap: the points are already resolved
    # by value_range() above.
    try:
        result["extent"] = _field_extent(ensemble, area)
    except Exception:
        result["extent"] = None
    return result


def _field_extent(ensemble, area):
    """Bounding box (w/e/s/n lon/lat) of the area's field, padded 5% per side to
    MATCH the client IDW raster's padding (buildField). Returns None if no
    points resolve.

    CLAMPED TO THE REQUESTED RECTANGLE (area.bounds): JSON /gridpoints samples
    snap to NOAA WFO gridpoint CENTERS that can sit OUTSIDE the drawn box, which
    would inflate/reshape the painted field + area-outline box away from what
    the user drew. We clamp the extent to the requested lattice rectangle so the
    box matches the drawn shape. (For NDFD/NBM the points ARE the lattice, so the
    clamp is a no-op.) The underlying point DATA is unchanged — only the painted
    extent is clamped."""
    lons = []
    lats = []
    if hasattr(ensemble, "iter_points"):
        for m in ensemble.iter_points():
            try:
                _st, _et, center, _f, _u, _x = m.model
            except Exception:
                continue
            lons.append(center.lon)
            lats.append(center.lat)
    else:
        for gp in area.iter_gridpoints():
            try:
                _st, _et, center, _f, _u, _x = ensemble[gp].model
            except Exception:
                continue
            lons.append(center.lon)
            lats.append(center.lat)
    if not lons:
        return None
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)
    # Clamp the point-cloud bbox to the REQUESTED rectangle so snapped-outward
    # gridpoint centers don't reshape the box. Intersect (don't expand): if the
    # resolved cloud is smaller than the request (sparse coverage) keep the
    # cloud; if larger (snap overshoot) clip to the request.
    #
    # For the JSON `gridpoints` source, clamp to the EXACT drawn rectangle
    # (`requested_bounds`) rather than the lattice-snapped `bounds`. `bounds`
    # rounds OUTWARD to lattice lines (up to one ~2.5 km step per side), which
    # visibly inflated the displayed area box beyond what the user drew. NDFD/
    # NBM points ARE the lattice, so for them the clamp stays a no-op against
    # `bounds` (using requested_bounds there could clip a legitimate edge cell).
    try:
        rb = area.requested_bounds if weather_source() == "gridpoints" else area.bounds
        min_lon = max(min_lon, rb["w"]); max_lon = min(max_lon, rb["e"])
        min_lat = max(min_lat, rb["s"]); max_lat = min(max_lat, rb["n"])
    except Exception:
        pass
    if max_lon <= min_lon or max_lat <= min_lat:
        return None
    # Mirror buildField: pad 5% of span per side, with a small floor when the
    # span is ~0 (single point / degenerate).
    #
    # For `gridpoints` we clamped to the EXACT drawn rectangle above, and the
    # area-outline box is sized from this extent — so we DON'T pad here, or the
    # displayed box would sit 5% proud of what the user drew. The client's
    # buildField still pads its raster internally and then INTERSECTS with this
    # (unpadded) hint, so the IDW raster is unaffected while the box matches the
    # drawn rectangle. NDFD/NBM keep the pad (box == lattice extent + margin).
    if weather_source() == "gridpoints":
        pad_lon = 0.0 if (max_lon - min_lon) else 0.01
        pad_lat = 0.0 if (max_lat - min_lat) else 0.01
    else:
        pad_lon = (max_lon - min_lon) * 0.05 or 0.01
        pad_lat = (max_lat - min_lat) * 0.05 or 0.01
    return {
        "w": min_lon - pad_lon, "e": max_lon + pad_lon,
        "s": min_lat - pad_lat, "n": max_lat + pad_lat,
    }


@router.get("/{model}/type/ensemble/stream")
def ensemble_stream(
    model: str,
    lat: float,
    lon: float,
    time: time_t=None,
    dt: int=0,
    width: float=1,
    height: float=1,
    kind: str="accumulation",
):
    """Stream the ensemble as newline-delimited JSON (NDJSON): one GeoJSON
    feature per line, computed from the (NOAA-cached) gridpoint data as each
    point is produced, so the client can render progressively."""
    dtime = datetime_from_time(time)
    coord = MapCoordinate(lat, lon)
    area = CoordinateArea(coord, width, height)
    ensemble_class = ENSEMBLE_KINDS().get(kind, AccumulationEnsemble)

    def generate():
        for feature in ensemble_class(area, model).iter_predict(dtime, verbose=True):
            yield json.dumps(feature) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


def _frame_times(start, end, dt, times):
    """Resolve the list of frame epochs to stream, in the order to emit them.

    `times` is a comma-separated batch of epochs and/or `from:to:step` runs
    (emitted first, in the order given); `start`/`end`/`dt` add a contiguous
    sequence after it. See weather_radar.lib.frametimes."""
    from weather_radar.lib.frametimes import frame_times
    return frame_times(start, end, dt, times)


@router.get("/{model}/type/series/stream")
def ensemble_series_stream(
    model: str,
    lat: float,
    lon: float,
    start: float = None,
    end: float = None,
    dt: int = 0,
    times: str = None,
    width: float = 1,
    height: float = 1,
    kind: str = "accumulation",
):
    """Stream many frames as NDJSON — one line per frame,
    {"time": <epoch>, "uom", "coords", "vals"}.

    Two modes, combinable in a single request (see _frame_times):
      * BATCH    : `times=5,15,25,...` — an explicit set of epochs, streamed
                   first. Lets the client fill arbitrary gaps it's missing.
      * SEQUENCE : `start`, `end`, `dt` — a contiguous series, streamed after
                   the batch. Lets the client stream the continuous tail.

    Example (client has 0,10,20,30,40,50 cached; now stepping dt=5m): request
    `times=5,15,25,35,45&start=55&end=<window end>&dt=300` — the gap frames
    arrive as a batch, then 55,60,65,... stream in sequence.

    BUNDLE FAST PATH: a WATCHED url may have a precomputed bundle (the whole
    window at its dt, built in the background), keyed by these exact request
    params. On a pure SEQUENCE request we serve that bundle's frame lines
    straight from disk — zero gridpoint resolution / fit / per-frame compute —
    then stop. A MISS (no matching bundle / batch request) falls through to
    live compute. (The range/window for a bundled url is delivered by the
    prefetch endpoint, which also matches the bundle, so no header is needed
    here — the frame lines are identical to the live-computed ones.)

    Each live-computed frame is derived from the (NOAA-cached) gridpoint data;
    results aren't cached server-side (the browser caches them)."""
    from datetime import datetime

    epochs = _frame_times(start, end, dt, times)
    coord = MapCoordinate(lat, lon)
    area = CoordinateArea(coord, width, height)
    ensemble_class = ENSEMBLE_KINDS().get(kind, AccumulationEnsemble)
    ensemble = ensemble_class(area, model)

    def generate():
        if not epochs:
            return

        # Bundle fast path: serve whatever requested epochs the precomputed
        # bundle contains (both the sequence AND any `times` batch), straight
        # from disk with zero compute. For a watched url the bundle holds the
        # whole window, so this typically satisfies the ENTIRE request; only
        # epochs the bundle lacks fall through to live compute below. (A
        # previous version only short-circuited pure-sequence requests, so the
        # common case — a fill with a gap `times` batch — bypassed the bundle
        # and recomputed everything live. Serving by epoch fixes that.)
        #
        # GRID FORMAT: the stream emits exactly ONE grid header (coords sent
        # once). When the bundle supplies anything it sends the header first;
        # otherwise the live path emits its own. `header_sent` tracks this so
        # coords aren't double-sent — and the FE keys all subsequent values-only
        # frames to that single header.
        #
        # The bundle's layers say (from their headers alone) which requested
        # epochs they hold. Those are sent FIRST, in time order — they're plain
        # disk reads, so they arrive instantly, before the slow cold fit. The
        # rest are then computed live in the ORDER THE CLIENT ASKED FOR (nearest
        # the playhead first), so the frame being looked at is never queued
        # behind far-away ones.
        remaining = list(epochs)
        header_sent = False
        feed = None
        try:
            from weather_radar.server.bundle import open_bundle_feed
            feed = open_bundle_feed(model, lat, lon, width, height, kind, epochs)
        except Exception:
            feed = None   # bundle error: compute the full request live
        if feed is not None:
            header_sent = True
            yield feed.header_record()
            yield from feed.upto(None)
            remaining = [e for e in epochs if e not in feed.held]
            if not remaining:
                return  # entire request served from the bundle

        # --- Phase timing (to locate first-frame latency: NOAA warm vs spline
        # fit vs evaluation). Logged once per request at WARNING so it surfaces
        # without extra log config. Monotonic clock; negligible overhead. ---
        t_start = time.perf_counter()

        n_hot = 0
        n_cold = 0
        models = []

        if _is_ndfd():
            # NDFD source: no WFO gridpoint resolution, no /gridpoints warm.
            # The ensemble fits the whole area's lattice points from the GRIB
            # grid once (lazily on first access); iterate its wrapped points.
            # The GRIB download itself is writer-owned inside ndfd.fit_area.
            t_resolved = t_warmed = time.perf_counter()
            try:
                models = list(ensemble.iter_points())
            except Exception:
                logger.warning("NDFD fit failed", exc_info=True)
                return
            n_cold = len(models)
            gridpoints = models  # for the timing log's gp count
            t_fit = time.perf_counter()
        else:
            # Resolve gridpoints + fit every spline ONCE for the whole sweep.
            # Previously each frame called ensemble.predict(), which re-resolved
            # the gridpoint set AND re-ran ensure() — re-opening/parsing all ~N
            # gridpoint cache files from disk every frame (N files × M frames I/O).
            # That per-frame redundancy, not the spline math, dominated the cost.
            # Here we build the per-gridpoint models once, then each frame is just
            # spline evaluations against already-fitted models.
            try:
                gridpoints = list(area.iter_gridpoints())
            except Exception:
                return
            t_resolved = time.perf_counter()
            # Warm all gridpoint data in one batched writer job before fitting.
            try:
                from weather_radar.lib.cachewriter import ensure as _ensure_cached
                _ensure_cached([f"/gridpoints/{gp}" for gp in gridpoints])
            except Exception:
                pass
            t_warmed = time.perf_counter()
            # Fetch (and fit) each gridpoint's model once. ensemble[gp] returns
            # the process-wide cached model, fitting it on first access. Skip
            # any that fail to build so one bad cell doesn't abort the sweep.
            #
            # Track how many models were already fitted (hot in this worker's
            # _MODEL_CACHE) vs. fit cold right now — the key signal for whether
            # keeping splines hot would help. A model is "hot" if its `model`
            # cached_property is already populated and not expired.
            for gp in gridpoints:
                try:
                    m = ensemble[gp]
                    was_hot = ("model" in m.__dict__) and (not m.is_expired)
                    m.model  # force the spline fit now (cached_property)
                    if was_hot:
                        n_hot += 1
                    else:
                        n_cold += 1
                    models.append(m)
                except Exception:
                    continue
            t_fit = time.perf_counter()
        if not models:
            return

        # Precompute each model's coordinate + units ONCE (identical across all
        # frames). Wire coords are [lon, lat] (the client's normalizeFrame reads
        # coords[0]=lon, coords[1]=lat; MapLibre's image-source corners need a
        # valid latitude). `center` is an honest MapCoordinate (.lat = true
        # latitude) for BOTH sources now, so [center.lon, center.lat] is correct.
        # (Earlier this emitted [center.lat, center.lon], which relied on a
        # swapped-axis center in the JSON path and was simply wrong for NDFD —
        # it fed MapLibre an out-of-range latitude on heatmap layers, throwing
        # "Invalid LngLat" and aborting the frame render.)
        coords = []   # [[lon, lat], ...] parallel to `models` (the FIXED grid)
        uom = ""
        kept = []
        for m in models:
            try:
                _st, _et, center, _f, units, _exp = m.model
            except Exception:
                continue
            coords.append([center.lon, center.lat])
            uom = uom or units
            kept.append(m)
        models = kept
        if not models:
            return

        # BINARY SIDECAR WIRE FORMAT (see weather_radar.lib.framing). The
        # coordinate lattice is FIXED for an area, so it is sent ONCE in a JSON
        # header record; every frame is a compact binary record
        # [epoch][N × int16] (values scaled by the header's scale/offset, a
        # sentinel for missing points). This replaces ~3-5 ASCII bytes/value +
        # a per-frame JSON.parse with 2 bytes/value + a typed-array view.
        #
        # Skip the header here if the bundle path already sent one (single
        # header per stream). Live coords == bundle coords (same ensemble point
        # order for the area), so the one header covers both. scale/offset are
        # derived from the uom and MUST match what the bundle used (choose_scale
        # is a pure function of uom, so they agree).
        scale, offset = framing.choose_scale(uom)
        if not header_sent:
            yield framing.record(framing.encode_header(uom, coords, scale, offset))

        t_eval_start = time.perf_counter()
        t_first_frame = None
        frames_emitted = 0
        # Fast per-point evaluator: NDFD point models expose predict_epoch()
        # (absolute-epoch spline eval, no per-call datetime/closure allocation);
        # JSON gridpoint models use the datetime predict(). Resolve the mode
        # ONCE (all models in a sweep are the same kind) so the hot loop below
        # doesn't branch per (point × frame).
        use_epoch = bool(models) and hasattr(models[0], "predict_epoch")
        # The exact bytes streamed are kept so they can be saved as a bundle
        # layer afterwards. The version token describes the data the models were
        # just fit from (None for the JSON source -> nothing is saved).
        new_items = []
        try:
            from weather_radar.server.bundle import _source_token
            layer_token = _source_token(model, kind)
        except Exception:
            layer_token = None
        for epoch in remaining:
            t = None if use_epoch else datetime.fromtimestamp(epoch)
            vals = []
            any_val = False
            for m in models:
                try:
                    y = m.predict_epoch(epoch) if use_epoch else m.predict(t, verbose=False)
                except BoundsError:
                    vals.append(None)   # out of horizon this frame -> sentinel
                    continue
                except Exception:
                    vals.append(None)
                    continue
                vals.append(y)
                any_val = True
            if any_val:
                if t_first_frame is None:
                    t_first_frame = time.perf_counter()
                frames_emitted += 1
                payload = framing.encode_frame(epoch, vals, scale, offset)
                new_items.append((int(epoch), payload))
                yield framing.record(payload)

        if new_items and layer_token is not None:
            try:
                from weather_radar.server.bundle import save_request_layer
                save_request_layer(model, lat, lon, width, height, kind,
                                   uom, coords, new_items, layer_token)
            except Exception:
                logger.warning("bundle layer save skipped", exc_info=True)

        # --- One structured timing line per request. Durations in ms. ---
        #   resolve     : area.iter_gridpoints()
        #   warm        : _ensure_cached (NOAA gridpoint data onto disk)
        #   fit         : build/fit per-gridpoint splines (cold) or reuse (hot)
        #   to_first    : fit end -> first frame emitted (eval of frame 1)
        #   total       : whole generate() span
        #   models      : hot (reused from _MODEL_CACHE) / cold (fit now)
        # If `fit` is large and `cold` dominates, keeping splines hot would cut
        # first-frame latency; if `warm` dominates, it's NOAA I/O instead.
        t_end = time.perf_counter()
        if _TIMING:
            ms = lambda a, b: round((b - a) * 1000, 1)
            logger.warning(
                "series_stream timing model=%s kind=%s gp=%d frames=%d "
                "resolve=%sms warm=%sms fit=%sms to_first=%sms total=%sms "
                "models_hot=%d models_cold=%d",
                model, kind, len(gridpoints), frames_emitted,
                ms(t_start, t_resolved), ms(t_resolved, t_warmed), ms(t_warmed, t_fit),
                ms(t_fit, t_first_frame) if t_first_frame else None,
                ms(t_start, t_end),
                n_hot, n_cold,
            )

    return StreamingResponse(_gzip_stream(generate()),
                             media_type=_STREAM_MEDIA_TYPE,
                             headers=_GZIP_STREAM_HEADERS)
