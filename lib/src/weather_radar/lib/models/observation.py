from functools import cached_property

from datetime import datetime
from scipy import interpolate

from weather_radar.lib.area import MapCoordinate, NOAAGridpoint, gridpoint_from_map_coordinate
from ..connection import NOAAConnection
from .utils import BoundsError, to_camel_case


WRAPPER_FUNCTIONS = {
        "temperature": lambda f, _: ((lambda *a, **kwd: (9/5*f(*a, **kwd)) + 32), "degF")
}


class ObservationModel:
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
        attr = to_camel_case(self.attribute)

        polygon, = data["geometry"]["coordinates"]
        polygon = set(tuple(x) for x in polygon)
        polygon = list(zip(*polygon))
        center_coordinate = [sum(item) / len(item) for item in polygon]
        center_coordinate = MapCoordinate(*center_coordinate)
        data = data["properties"][attr]
        _, units = data["uom"].split(":")
        data = data["values"]
        data = [
            (datetime.fromisoformat(d["validTime"].split("/")[0]), d["value"])
            for d in data
        ]
        start_time = data[0][0]
        end_time = data[-1][0]
        model_data = [
            ((a-start_time).total_seconds(), b)
            for a, b in data
        ]
        model_data = list(zip(*model_data))
        x, y = model_data
        model = interpolate.CubicSpline(x, y)
        model, units = WRAPPER_FUNCTIONS.get(attr, lambda f, u: (f, u))(model, units)
        return start_time, end_time, center_coordinate, model, units

    def predict(self, time, verbose=False):
        if type(time) is not datetime:
            return self.predict_many(time, verbose=verbose)

        start_time, end_time, center_coordinate, f, units = self.model
        time = time.astimezone()
        if time > end_time:
            raise BoundsError
        t = (time - start_time).total_seconds()
        y = f(t).item()
        if verbose:
            return {
                "type": "Feature",
                "properties": {
                    self.attribute: y,
                    "time": time.isoformat(),
                    "uom": units
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [center_coordinate.lat, center_coordinate.lon]
                }
            }
        return y

    def predict_many(self, times, verbose=False):
        features = []
        for time in times:
            try:
                feature = self.predict(time, verbose=verbose)
            except BoundsError:
                break
            else:
                features.append(feature)
        return {
            "type": "FeatureCollection",
            "features": features
        }
