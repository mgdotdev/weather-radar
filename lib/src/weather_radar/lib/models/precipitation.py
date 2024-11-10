from datetime import datetime
from functools import cached_property
from weather_radar.lib.area import Coordinate, MapCoordinate, CoordinateArea, center_gridpoint_from_map_coordinate, center_id_from_map_coordinate
from ..connection import NOAAConnection

from scipy.interpolate import CubicSpline


class PrecipitationModel:
    def __init__(self, coordinate: Coordinate, center_id: str):
        self.coordinate = coordinate
        self.center_id = center_id

    @classmethod
    def from_map_coordinate(cls, coordinate: MapCoordinate):
        return cls(
            Coordinate(*center_gridpoint_from_map_coordinate(coordinate)),
            center_id_from_map_coordinate(coordinate)
        )


    @cached_property
    def model(self):
        """CSpline of a cumulative model, derivative is precipitation at a
        given time"""

        conn = NOAAConnection()
        data = conn.get(f"/gridpoints/{self.center_id}/{self.coordinate}")

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

    def predict(self, time, verbose=False):
        start_time, center_coordinate, f = self.model
        time = time.astimezone()
        x = (time - start_time).total_seconds()
        y = f(x).item()
        if verbose:
            return {
                "value": y,
                "lat": center_coordinate.lat,
                "lon": center_coordinate.lon
            }
        return y


class PrecipitationEnsemble:
    def __init__(self, area: CoordinateArea):
        self.area = area
        self._cache = {}

    def __getitem__(self, coordinate: Coordinate):
        model = self._cache.get(coordinate, None)
        if not model:
            model = PrecipitationModel(coordinate, self.area.center_id)
            self._cache[coordinate] = model
        return model

    def predict(self, time: datetime, verbose=False):
        return {
            "model": "precipitation",
            "time": time.isoformat(),
            "results": [
                self[coordinate].predict(time, verbose=verbose)
                for coordinate in self.area.gridpoints
            ]
        }
