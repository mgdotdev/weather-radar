from datetime import datetime
from functools import cached_property
from weather_radar.lib.area import CoordinateArea, MapCoordinate, NOAAGridpoint, gridpoint_from_map_coordinate
from ..connection import NOAAConnection

from scipy.interpolate import CubicSpline


class PrecipitationModel:
    def __init__(self, coordinate: NOAAGridpoint):
        self.coordinate = coordinate

    @classmethod
    def from_map_coordinate(cls, coordinate: MapCoordinate):
        return cls(
            gridpoint_from_map_coordinate(coordinate)
        )

    @cached_property
    def model(self):
        """CSpline of a cumulative model, derivative is precipitation at a
        given time"""

        conn = NOAAConnection()
        data = conn.get(f"/gridpoints/{self.coordinate}")

        polygon, = data["geometry"]["coordinates"]
        polygon = set(tuple(x) for x in polygon)
        polygon = list(zip(*polygon))
        center_coordinate = [sum(item) / len(item) for item in polygon]
        center_coordinate = MapCoordinate(*center_coordinate)

        data = data["properties"]["quantitativePrecipitation"]["values"]
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
        model = CubicSpline(model_data[0], model_data[1]).derivative()
        return start_time, center_coordinate, model

    def predict(self, time, dt=0, verbose=False):
        start_time, center_coordinate, f = self.model
        time = time.astimezone()
        t = (time - start_time).total_seconds()
        y = f(t) if not dt else f.integrate(t, t+dt)
        if verbose:
            return {
                "type": "Feature",
                "properties": {
                    "precipitation": y.item(),
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [center_coordinate.lat, center_coordinate.lon]
                }
            }
        return y


class PrecipitationEnsemble:
    def __init__(self, area: CoordinateArea):
        self.area = area
        self._cache = {}

    def __getitem__(self, coordinate: NOAAGridpoint):
        model = self._cache.get(coordinate, None)
        if not model:
            model = PrecipitationModel(coordinate)
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
