from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse
import json

from weather_radar.server.api.models.utils import datetime_from_time, time_t

from weather_radar.lib.models import model_from_type
from weather_radar.lib.models.accumulation import AccumulationEnsemble
from weather_radar.lib.models.observation import ObservationEnsemble
from weather_radar.lib.area import MapCoordinate, CoordinateArea

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
    ensemble = ensemble_class(area, model)
    return ensemble.predict(dtime, dt=dt, verbose=True)


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
    lifecycle for the area, so the client can fix its color scale before the
    first frame. Fast once the area is cached (call /prefetch first)."""
    coord = MapCoordinate(lat, lon)
    area = CoordinateArea(coord, width, height)
    ensemble_class = ENSEMBLE_KINDS.get(kind, AccumulationEnsemble)
    ensemble = ensemble_class(area, model)
    return ensemble.value_range() or {"min": None, "max": None, "uom": None}


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
    feature per line, emitted as each gridpoint is computed. Lets the client
    render points progressively instead of waiting for the whole area."""
    dtime = datetime_from_time(time)
    coord = MapCoordinate(lat, lon)
    area = CoordinateArea(coord, width, height)
    ensemble_class = ENSEMBLE_KINDS.get(kind, AccumulationEnsemble)
    ensemble = ensemble_class(area, model)

    def generate():
        for feature in ensemble.iter_predict(dtime, verbose=True):
            yield json.dumps(feature) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")
