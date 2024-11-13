from fastapi import APIRouter
from fastapi.responses import JSONResponse

from datetime import datetime
from weather_radar.lib.area import MapCoordinate, CoordinateArea, cache_coordinate_area
from weather_radar.lib.models.precipitation import PrecipitationEnsemble, PrecipitationModel

time_t = float|str|None

router = APIRouter(prefix="/precipitation")


def _datetime_from_time(time: time_t) -> datetime:
    if time is None:
        return datetime.now()
    try:
        dtime = float(time)
    except ValueError:
        return datetime.fromisoformat(str(time))
    else:
        return datetime.fromtimestamp(dtime)


@router.get("/prefetch", status_code=204)
def prefetch(lat: float, lon: float, width: float|None=None, height: float|None=None):
    coord = MapCoordinate(lat, lon)
    if width or height:
        area = CoordinateArea(coord, width, height)
        cache_coordinate_area(area)



@router.get("/type/point", response_class=JSONResponse)
def point(lat: float, lon: float, time: time_t=None, dt:int=0):
    dtime = _datetime_from_time(time)
    coord = MapCoordinate(lat, lon)
    model = PrecipitationModel.from_map_coordinate(coord)
    results = model.predict(dtime, dt=dt, verbose=True)
    return {
        "model": "precipitation",
        "time": dtime.isoformat(),
        "results": results
    }


@router.get("/type/ensemble", response_class=JSONResponse)
def ensemble(lat: float, lon: float, time: time_t=None, dt: int=0, width: float=1, height: float=1):
    dtime = _datetime_from_time(time)
    coord = MapCoordinate(lat, lon)
    area = CoordinateArea(coord, width, height)
    ensemble = PrecipitationEnsemble(area)
    return ensemble.predict(dtime, dt=dt, verbose=True)

