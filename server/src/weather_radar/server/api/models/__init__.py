from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
import json
import logging
import os
import time

from weather_radar.server.api.models.utils import datetime_from_time, time_t

from weather_radar.lib.models import model_from_type
from weather_radar.lib.models.accumulation import AccumulationEnsemble
from weather_radar.lib.models.observation import ObservationEnsemble
from weather_radar.lib.area import MapCoordinate, CoordinateArea
from weather_radar.lib.models.utils import BoundsError

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
ENSEMBLE_KINDS = {
    "accumulation": AccumulationEnsemble,
    "observation": ObservationEnsemble,
}


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
    ensemble_class = ENSEMBLE_KINDS.get(kind, AccumulationEnsemble)
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
    from weather_radar.lib.aqi import fit_area, predict_area
    coord = MapCoordinate(lat, lon)
    area = CoordinateArea(coord, width, height)
    epochs = _frame_times(start, end, dt, times)

    def generate():
        if not epochs:
            return
        # Bundle first (keyed by kind="aqi", model=pollutant).
        remaining = list(epochs)
        try:
            from weather_radar.server.bundle import serve_bundle_epochs
            covered = set()
            for line in serve_bundle_epochs(pollutant, lat, lon, width, height,
                                            dt, "aqi", epochs, covered):
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
        for fr in predict_area(fit, remaining):
            yield json.dumps({"time": fr["epoch"], "uom": "AQI",
                              "coords": fr["coords"], "vals": fr["vals"]}) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")



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
    ensemble_class = ENSEMBLE_KINDS.get(kind, AccumulationEnsemble)
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
    ensemble_class = ENSEMBLE_KINDS.get(kind, AccumulationEnsemble)

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
    ensemble_class = ENSEMBLE_KINDS.get(kind, AccumulationEnsemble)
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
        remaining = list(epochs)
        try:
            from weather_radar.server.bundle import serve_bundle_epochs
            covered = set()
            for line in serve_bundle_epochs(model, lat, lon, width, height,
                                            dt, kind, epochs, covered):
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
        # Fetch (and fit) each gridpoint's model once. ensemble[gp] returns the
        # process-wide cached model, fitting it on first access. Skip any that
        # fail to build so one bad cell doesn't abort the sweep.
        #
        # Track how many models were already fitted (hot in this worker's
        # _MODEL_CACHE) vs. fit cold right now — the key signal for whether
        # keeping splines hot would help. A model is "hot" if its `model`
        # cached_property is already populated and not expired.
        n_hot = 0
        n_cold = 0
        models = []
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
        # frames). Coordinate order is preserved EXACTLY as the GeoJSON path
        # emitted it — [center.lat, center.lon] — so the client reads coords the
        # same way it always has; this is purely a wire-format slimming, not a
        # semantic change.
        coords = []   # [[c0, c1], ...] parallel to `models`
        uom = ""
        kept = []
        for m in models:
            try:
                _st, _et, center, _f, units, _exp = m.model
            except Exception:
                continue
            coords.append([center.lat, center.lon])
            uom = uom or units
            kept.append(m)
        models = kept
        if not models:
            return

        # Each frame is emitted as a SLIM record instead of a full GeoJSON
        # FeatureCollection: per-frame `time` and `uom` once (not repeated on
        # every point), and parallel `coords`/`vals` arrays instead of ~N nested
        # Feature objects. This cuts payload size and client JSON.parse cost
        # several-fold. Records are SELF-CONTAINED (coords included each frame)
        # so the client can cache and render any frame independently; coords is
        # just numbers, so even repeated it's far smaller than GeoJSON. `vals`
        # is aligned to `coords`; null where a gridpoint is out of horizon.
        t_eval_start = time.perf_counter()
        t_first_frame = None
        frames_emitted = 0
        for epoch in remaining:
            t = datetime.fromtimestamp(epoch)
            fcoords = []
            vals = []
            any_val = False
            for m, c in zip(models, coords):
                try:
                    y = m.predict(t, verbose=False)
                except BoundsError:
                    continue  # out of horizon this frame -> omit the point
                except Exception:
                    continue
                fcoords.append(c)
                vals.append(y)
                any_val = True
            if any_val:
                if t_first_frame is None:
                    t_first_frame = time.perf_counter()
                frames_emitted += 1
                yield json.dumps({"time": epoch, "uom": uom, "coords": fcoords, "vals": vals}) + "\n"

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

    return StreamingResponse(generate(), media_type="application/x-ndjson")
