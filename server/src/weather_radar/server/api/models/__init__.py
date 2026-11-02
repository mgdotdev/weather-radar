from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
import json

from weather_radar.server.api.models.utils import datetime_from_time, time_t

from weather_radar.lib.models import model_from_type
from weather_radar.lib.models.accumulation import AccumulationEnsemble
from weather_radar.lib.models.observation import ObservationEnsemble
from weather_radar.lib.area import MapCoordinate, CoordinateArea
from weather_radar.lib import cachewriter

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
def prefetch_stream(lat: float, lon: float, width: float=1, height: float=1):
    """Stream prefetch progress as NDJSON (one JSON object per line) while the
    area's data is fetched from NOAA. Each line is
    {"phase": "points"|"gridpoints", "done": N, "total": M} so the client can
    show a live completed/total fraction. Phases run in order; the final line
    of each phase has done == total."""
    coord = MapCoordinate(lat, lon)
    area = CoordinateArea(coord, width, height)

    def generate():
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
    epoch = int(dtime.timestamp())
    params = {
        "model": model, "kind": kind, "lat": lat, "lon": lon,
        "width": width, "height": height, "time": epoch,
    }
    # Preferred path: the writer process computes + caches the frame (single-
    # writer invariant) and returns it. Fall back to inline compute if the
    # writer is unreachable (tests / standalone runs).
    try:
        return cachewriter.frame(params)
    except cachewriter.WriterUnavailable:
        coord = MapCoordinate(lat, lon)
        area = CoordinateArea(coord, width, height)
        ensemble_class = ENSEMBLE_KINDS.get(kind, AccumulationEnsemble)
        return ensemble_class(area, model).predict(dtime, dt=dt, verbose=True)


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
    feature per line. Lets the client render points progressively.

    The frame itself is computed + cached by the writer process (single-writer
    invariant); this endpoint fetches it (a cache hit after the first time) and
    streams its features out. Falls back to inline progressive compute if the
    writer is unreachable."""
    dtime = datetime_from_time(time)
    epoch = int(dtime.timestamp())
    params = {
        "model": model, "kind": kind, "lat": lat, "lon": lon,
        "width": width, "height": height, "time": epoch,
    }

    def generate():
        try:
            fc = cachewriter.frame(params)
        except cachewriter.WriterUnavailable:
            # Writer down: compute inline, streaming each gridpoint as it's
            # produced (no caching in this fallback path).
            coord = MapCoordinate(lat, lon)
            area = CoordinateArea(coord, width, height)
            ensemble_class = ENSEMBLE_KINDS.get(kind, AccumulationEnsemble)
            for feature in ensemble_class(area, model).iter_predict(dtime, verbose=True):
                yield json.dumps(feature) + "\n"
            return
        for feature in (fc or {}).get("features", []):
            yield json.dumps(feature) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")
