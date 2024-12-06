from fastapi import APIRouter
from fastapi.responses import JSONResponse
from weather_radar.server.api.models.utils import datetime_from_time, time_t

from weather_radar.lib.models import model_from_type
from weather_radar.lib.models.accumulation import AccumulationEnsemble
from weather_radar.lib.area import MapCoordinate, CoordinateArea

router = APIRouter(prefix="/models")


@router.get("/{model}/params/{param}", response_class=JSONResponse)
def models(model: str, param: str, lat: float, lon: float, time: time_t=None, dt:int=0, counts: int=-1):
    dtime = datetime_from_time(time, dt=dt, counts=counts)
    coord = MapCoordinate(lat, lon)
    model_class = model_from_type(model)
    model = model_class.from_map_coordinate(coord, param)
    return model.predict(dtime, dt=dt, verbose=True)


@router.get("/prefetch", status_code=204)
def prefetch(lat: float, lon: float, width: float|None=None, height: float|None=None):
    coord = MapCoordinate(lat, lon)
    if width or height:
        area = CoordinateArea(coord, width, height)
        area.cache()


@router.get("/{model}/type/ensemble", response_class=JSONResponse)
def ensemble(model: str, lat: float, lon: float, time: time_t=None, dt: int=0, width: float=1, height: float=1):
    dtime = datetime_from_time(time)
    coord = MapCoordinate(lat, lon)
    area = CoordinateArea(coord, width, height)
    ensemble = AccumulationEnsemble(area, model)
    return ensemble.predict(dtime, dt=dt, verbose=True)
