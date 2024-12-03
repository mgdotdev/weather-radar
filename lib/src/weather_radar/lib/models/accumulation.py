import re

from datetime import datetime
from functools import cached_property
from weather_radar.lib.area import CoordinateArea, MapCoordinate, NOAAGridpoint, gridpoint_from_map_coordinate
from ..connection import NOAAConnection

from scipy import interpolate

CAMEL_TO_KEBAB = re.compile(r'(?<!^)(?=[A-Z])')
KEBAB_TO_CAMEL = re.compile(r'(?<!\A)-(?=[a-zA-Z])',re.X)


def to_camel_case(value):
    tokens = KEBAB_TO_CAMEL.split(value)
    response = tokens.pop(0).lower()
    for remain in tokens:
        response += remain.capitalize()
    return response


def to_kebab_case(value):
    return CAMEL_TO_KEBAB.sub('_', value).lower()


class AccumulationModel:
    def __init__(self, coordinate: NOAAGridpoint, attribute: str):
        self.coordinate = coordinate
        self.attribute = attribute

    @classmethod
    def from_map_coordinate(cls, coordinate: MapCoordinate, attribute: str):
        return cls(
            gridpoint_from_map_coordinate(coordinate), attribute
        )

    @cached_property
    def model(self):
        """Spline of a cumulative model, derivative is accumulator at a
        given time"""

        conn = NOAAConnection()
        data = conn.get(f"/gridpoints/{self.coordinate}")

        polygon, = data["geometry"]["coordinates"]
        polygon = set(tuple(x) for x in polygon)
        polygon = list(zip(*polygon))
        center_coordinate = [sum(item) / len(item) for item in polygon]
        center_coordinate = MapCoordinate(*center_coordinate)
        data = data["properties"]
        data = data[to_camel_case(self.attribute)]["values"]
        data = [
            (datetime.fromisoformat(d["validTime"].split("/")[0]), d["value"])
            for d in data
        ]
        start_time = data[0][0]
        model_data = [
            ((a-start_time).total_seconds(), sum(d[1] for d in data[:i+1]))
            for i, (a, _) in enumerate(data)
        ]
        model_data = list(zip(*model_data))
        x, y = model_data
        model = interpolate.BSpline(x, y, k=3).derivative()
        return start_time, center_coordinate, model

    def predict(self, time, dt=0, verbose=False):
        if type(time) is list:
            return self.predict_many(time, dt=dt, verbose=verbose)

        start_time, center_coordinate, f = self.model
        time = time.astimezone()
        t = (time - start_time).total_seconds()
        y = f(t) if not dt else f.integrate(t, t+dt)
        y = y.item()
        if y < 0:
            y = 0
        if verbose:
            return {
                "type": "Feature",
                "properties": {
                    self.attribute: y,
                    "time": time.isoformat(),
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [center_coordinate.lat, center_coordinate.lon]
                }
            }
        return y

    def predict_many(self, times, dt=0, verbose=False):
        return {
            "type": "FeatureCollection",
            "features": [
                self.predict(time, dt=dt, verbose=verbose)
                for time in times
            ]
        }


class AccumulationEnsemble:
    def __init__(self, area: CoordinateArea, attribute: str):
        self.area = area
        self.attribute = attribute
        self._cache = {}

    def __getitem__(self, coordinate: NOAAGridpoint):
        model = self._cache.get(coordinate, None)
        if not model:
            model = AccumulationModel(coordinate, self.attribute)
            self._cache[coordinate] = model
        return model

    def predict(self, time: datetime, dt=0, verbose=False):
        return {
            "type": "FeatureCollection",
            "features": [
                self[coordinate].predict(time, dt=dt, verbose=verbose)
                for coordinate in self.area.gridpoints
            ]
        }
