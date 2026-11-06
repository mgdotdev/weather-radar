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
from weather_radar.lib.models.utils import BoundsError


def _gzip_stream(line_iter):
    """Wrap a text-line generator in a streaming gzip compressor.

    Yields gzip-compressed bytes incrementally: each source chunk is compressed
    with Z_SYNC_FLUSH so the client receives (and can decode) data as it
    arrives — preserving the per-frame streaming / first-frame latency that the
    whole animation pipeline depends on. The grid wire format repeats small
    integers and (for the header) a coord lattice, both of which gzip heavily;
    combined with coords-sent-once this cuts a large sweep from ~hundreds of MB
    to tens of MB on the wire.

    The browser's fetch() transparently inflates `Content-Encoding: gzip`, so
    the client's NDJSON line reader is UNCHANGED."""
    # wbits=31 -> gzip wrapper (not raw deflate / zlib).
    compressor = zlib.compressobj(6, zlib.DEFLATED, 31)
    for line in line_iter:
        if not line:
            continue
        chunk = compressor.compress(line.encode("utf-8"))
        chunk += compressor.flush(zlib.Z_SYNC_FLUSH)
        if chunk:
            yield chunk
    tail = compressor.flush(zlib.Z_FINISH)
    if tail:
        yield tail


# Standard headers for a gzipped NDJSON stream.
_GZIP_NDJSON_HEADERS = {"Content-Encoding": "gzip"}


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
# (WR_WEATHER_SOURCE): gridpoints JSON (default) or NDFD GRIB. ENSEMBLE_KINDS
# is a function (not a static dict) so a source change is picked up without a
# reimport. Callers resolve with ENSEMBLE_KINDS().get(kind, <default>).
def ENSEMBLE_KINDS():
    return ensemble_kinds()


# True when the active weather source is NDFD (GRIB grid sampling) rather than
# the NOAA /gridpoints JSON. NDFD ensembles sample the area's lattice points
# directly (no WFO gridpoint resolution), so the hot series path branches on
# this to iterate the ensemble's points instead of area.iter_gridpoints().
def _is_ndfd():
    return weather_source() == "ndfd"


@router.get("/{model}/params/{param}", response_class=JSONResponse)
def models(model: str, param: str, lat: float, lon: float, time: time_t=None, dt:int=0, counts: int=-1):
    dtime = datetime_from_time(time, dt=dt, counts=counts)
    coord = MapCoordinate(lat, lon)
    model_class = model_from_type(model)
    model = model_class.from_map_coordinate(coord, param)
    return model.predict(dtime, verbose=True)


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

    BUNDLE SHORT-CIRCUIT: if a precomputed bundle exists for (model, area, dt,
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
                rng = read_bundle_range(model, lat, lon, width, height, dt, kind)
                if rng is not None:
                    yield json.dumps({"phase": "bundle", "range": rng}) + "\n"
                    return
            except Exception:
                pass  # fall through to the normal resolve/fetch

        # NDFD source: there are no NOAA /points or /gridpoints to resolve —
        # the national element GRIB is fetched (writer-owned) and sampled in
        # the fit. Warm it here so the subsequent /type/range + series stream
        # are fast, and report a single-phase completion so the client's
        # prefetch progress bar still advances.
        if _is_ndfd() and model is not None:
            try:
                from weather_radar.lib import ndfd
                ndfd._load_grid(model)  # download (writer) + parse + cache
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
        rng = read_bundle_range(pollutant, lat, lon, width, height, 3600, "aqi")
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
    from weather_radar.lib.aqi import fit_area, predict_area, grid_points
    coord = MapCoordinate(lat, lon)
    area = CoordinateArea(coord, width, height)
    epochs = _frame_times(start, end, dt, times)

    def generate():
        if not epochs:
            return
        # Bundle first (keyed by kind="aqi", model=pollutant). The bundle emits
        # the grid header; the live path emits one only if the bundle didn't.
        remaining = list(epochs)
        header_sent = False
        try:
            from weather_radar.server.bundle import serve_bundle_epochs
            covered = set()
            for line in serve_bundle_epochs(pollutant, lat, lon, width, height,
                                            dt, "aqi", epochs, covered,
                                            emit_header=True):
                header_sent = True
                yield line
            if covered:
                remaining = [e for e in epochs if int(e) not in covered]
                if not remaining:
                    return  # entire request served from the bundle
        except Exception:
            remaining = list(epochs)

        # Live spline path for whatever the bundle didn't cover.
        fit = fit_area(area, pollutant)
        if fit is None:
            return
        # GRID FORMAT: coords once, then values-only frames aligned to them.
        if not header_sent:
            yield json.dumps({"type": "grid", "uom": "AQI",
                              "coords": grid_points(fit)},
                             separators=(",", ":")) + "\n"
        for fr in predict_area(fit, remaining, fixed_grid=True):
            yield json.dumps({"time": fr["epoch"], "vals": fr["vals"]},
                             separators=(",", ":")) + "\n"

    return StreamingResponse(_gzip_stream(generate()),
                             media_type="application/x-ndjson",
                             headers=_GZIP_NDJSON_HEADERS)



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
    return result


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

    Supports two modes, which may be COMBINED in one request:
      * batch    - an explicit comma-separated list of epochs (`times=`),
                   emitted first. Used to fill arbitrary gaps the client is
                   missing (e.g. the odd 5-min slots when it already has the
                   evens), in no particular spacing.
      * sequence - a contiguous series start..end stepped by dt, emitted after
                   the batch. Used to stream the continuous tail.

    Duplicates are removed while preserving order (batch before sequence), so a
    time present in both isn't computed/emitted twice.
    """
    out = []
    seen = set()

    def add(epoch):
        e = int(epoch)
        if e not in seen:
            seen.add(e)
            out.append(e)

    # Batch first.
    if times:
        for tok in str(times).split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                add(float(tok))
            except ValueError:
                continue

    # Then the contiguous sequence.
    if start is not None and end is not None:
        step = int(dt) if dt and int(dt) > 0 else 3600
        t = float(start)
        count = 0
        while t <= float(end) + 1 and count < 5000:
            add(t)
            t += step
            count += 1

    return out


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
        # once). The bundle path emits its header first when it serves anything;
        # if it didn't (no bundle / nothing covered), the live path emits its
        # own. `header_sent` tracks this so coords aren't double-sent — and the
        # FE keys all subsequent values-only frames to that single header.
        remaining = list(epochs)
        header_sent = False
        try:
            from weather_radar.server.bundle import serve_bundle_epochs
            covered = set()
            for line in serve_bundle_epochs(model, lat, lon, width, height,
                                            dt, kind, epochs, covered,
                                            emit_header=True):
                header_sent = True  # first line from the bundle is the header
                yield line
            if covered:
                remaining = [e for e in epochs if int(e) not in covered]
                if not remaining:
                    return  # entire request served from the bundle
        except Exception:
            # Bundle error: fall back to computing the full request live.
            remaining = list(epochs)

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

        # GRID WIRE FORMAT. The coordinate lattice is FIXED for an area, so it
        # is sent ONCE as a header line and omitted from every frame:
        #   {"type":"grid","uom":...,"coords":[[lon,lat],...]}   (first line)
        #   {"time":epoch,"vals":[v0,v1,...]}                     (each frame)
        # `vals` is aligned to the header's `coords` by INDEX and is always the
        # same length; a point with no value THIS frame (out of horizon) is
        # `null`. This removes the ~Nx coordinate repetition that dominated the
        # payload on large areas (coords were ~86% of a multi-hundred-MB sweep).
        # The client rehydrates each values-only frame against the header coords
        # before caching, so the cache/render path stays self-contained.
        #
        # Skip the header here if the bundle path already sent one (single
        # header per stream). Live coords == bundle coords (same ensemble point
        # order for the area), so the one header covers both.
        if not header_sent:
            yield json.dumps({"type": "grid", "uom": uom, "coords": coords},
                             separators=(",", ":")) + "\n"

        t_eval_start = time.perf_counter()
        t_first_frame = None
        frames_emitted = 0
        for epoch in remaining:
            t = datetime.fromtimestamp(epoch)
            vals = []
            any_val = False
            for m in models:
                try:
                    y = m.predict(t, verbose=False)
                except BoundsError:
                    vals.append(None)   # out of horizon this frame -> null slot
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
                yield json.dumps({"time": epoch, "vals": vals},
                                 separators=(",", ":")) + "\n"

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
                             media_type="application/x-ndjson",
                             headers=_GZIP_NDJSON_HEADERS)
