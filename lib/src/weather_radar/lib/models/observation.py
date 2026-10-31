from functools import cached_property

from datetime import datetime
from threading import Lock

from scipy import interpolate

from weather_radar.lib.area import CoordinateArea, MapCoordinate, NOAAGridpoint, gridpoint_from_map_coordinate
from ..connection import NOAAConnection
from .utils import BoundsError, to_camel_case


WRAPPER_FUNCTIONS = {
        "temperature": lambda f, _: ((lambda *a, **kwd: (9/5*f(*a, **kwd)) + 32), "degF")
}


# Process-wide cache of fitted observation models, keyed by
# (gridpoint, attribute). See accumulation.py for the rationale: fitting a
# spline is expensive and the underlying NOAA data only changes hourly.
_MODEL_CACHE: dict[tuple, "ObservationModel"] = {}
_MODEL_CACHE_LOCK = Lock()


def get_observation_model(coordinate: NOAAGridpoint, attribute: str) -> "ObservationModel":
    """Return a cached ObservationModel, rebuilding it only when the
    underlying NOAA data has expired."""
    key = (coordinate, attribute)
    with _MODEL_CACHE_LOCK:
        model = _MODEL_CACHE.get(key)
        if model is not None and not model.is_expired:
            return model
        model = ObservationModel(coordinate, attribute)
        _MODEL_CACHE[key] = model
        return model


class ObservationModel:
    def __init__(self, coordinate: NOAAGridpoint, attribute: str):
        self.coordinate = coordinate
        self.attribute = attribute

    @classmethod
    def from_map_coordinate(cls, coordinate: MapCoordinate, attribute: str):
        return cls(
            gridpoint_from_map_coordinate(coordinate), attribute
        )

    @property
    def is_expired(self) -> bool:
        """True if the fitted model is missing or its source data has
        expired and should be refit."""
        if "model" not in self.__dict__:
            return True
        expiry = self.__dict__["model"][-1]
        return expiry < datetime.now()

    @cached_property
    def model(self):
        """Spline interpolating observed values over time."""

        conn = NOAAConnection()
        expiry, data = conn.get_with_expiry(f"/gridpoints/{self.coordinate}")
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
        return start_time, end_time, center_coordinate, model, units, expiry

    def predict(self, time, verbose=False):
        if type(time) is not datetime:
            return self.predict_many(time, verbose=verbose)

        start_time, end_time, center_coordinate, f, units, _expiry = self.model
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


class ObservationEnsemble:
    """Multi-gridpoint ensemble of observation models over an area, producing
    a GeoJSON FeatureCollection suitable for a heatmap (e.g. temperature)."""

    def __init__(self, area: CoordinateArea, attribute: str):
        self.area = area
        self.attribute = attribute

    def __getitem__(self, coordinate: NOAAGridpoint):
        return get_observation_model(coordinate, self.attribute)

    def iter_predict(self, time: datetime, verbose=False):
        """Yield one GeoJSON feature per gridpoint as it is computed, so
        callers can stream/render results progressively. Gridpoints are
        resolved nearest-to-center first and emitted as they become
        available. Out-of-horizon points are skipped."""
        for coordinate in self.area.iter_gridpoints():
            try:
                yield self[coordinate].predict(time, verbose=verbose)
            except BoundsError:
                # Requested time is outside this gridpoint's forecast horizon;
                # skip the point rather than failing the whole ensemble.
                continue

    def predict(self, time: datetime, dt=0, verbose=False):
        # `dt` is accepted for API compatibility but is only meaningful when
        # expanding a single time into a series of times. Once a concrete
        # `time` is resolved, each per-gridpoint prediction does not use `dt`.
        return {
            "type": "FeatureCollection",
            "features": list(self.iter_predict(time, verbose=verbose)),
        }

    def value_range(self, samples: int = 24):
        """Global min/max of the predicted value over the ENTIRE forecast
        lifecycle, across all gridpoints. Computed by sampling each
        gridpoint's fitted model at `samples` times spanning its forecast
        horizon (so unit wrappers like degC->degF are applied consistently).

        Returns {"min", "max", "uom"} or None if there's no data. Requires the
        area's data to be cached (call /prefetch first) to be fast.
        """
        vmin = None
        vmax = None
        units = None
        for coordinate in self.area.iter_gridpoints():
            m = self[coordinate]
            try:
                start_time, end_time, _center, f, uom, _expiry = m.model
            except Exception:
                continue
            units = units or uom
            total = (end_time - start_time).total_seconds()
            if total <= 0:
                continue
            for i in range(samples + 1):
                t = total * i / samples
                y = f(t).item()
                if vmin is None or y < vmin:
                    vmin = y
                if vmax is None or y > vmax:
                    vmax = y
        if vmin is None:
            return None
        return {"min": vmin, "max": vmax, "uom": units}
