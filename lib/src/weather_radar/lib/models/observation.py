from functools import cached_property

import logging

from datetime import datetime
from threading import Lock

from scipy import interpolate

from weather_radar.lib.area import CoordinateArea, MapCoordinate, NOAAGridpoint, gridpoint_from_map_coordinate
from ..connection import NOAAConnection
from .utils import BoundsError, to_camel_case, clamp_value, source_mtime


logger = logging.getLogger(__name__)


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
        gp = gridpoint_from_map_coordinate(coordinate)
        if gp is None:
            # NOAA has no coverage for this point (outside the US). Signal "no
            # data here" rather than constructing a model on a None gridpoint.
            raise BoundsError(f"no NOAA coverage for {coordinate}")
        return cls(gp, attribute)

    @property
    def is_expired(self) -> bool:
        """True if the fitted model is missing or its underlying NOAA data has
        been UPDATED on disk since the fit.

        Invalidation is keyed on the source cache file's mtime, NOT wall-clock
        time: the cache-writer rewrites the /gridpoints doc (bumping mtime)
        only when it actually refetches the resource, so a still-current file
        keeps the SAME mtime and the spline stays valid — even past the NOAA
        `Expires` time — until the data is genuinely republished. This avoids
        needlessly refitting (×every worker) when nothing changed. A single
        os.stat; no parse of the large doc on the hot path.
        """
        if "model" not in self.__dict__:
            return True
        current = source_mtime(f"/gridpoints/{self.coordinate}")
        if current is None:
            # Source file gone/unreadable: fall back to the baked-in Expires so
            # we don't serve an indefinitely stale spline.
            expiry = self.__dict__["model"][-1]
            return expiry < datetime.now()
        return current != self._source_mtime

    @cached_property
    def model(self):
        """Spline interpolating observed values over time."""

        # Record the source file's mtime BEFORE reading it, so is_expired's
        # later comparison can detect any republish that happens after this
        # fit. (Capturing before the read is the conservative order: if the
        # writer republishes between our stat and read, we'll see the new
        # mtime next check and refit — never the reverse.)
        self._source_mtime = source_mtime(f"/gridpoints/{self.coordinate}")

        conn = NOAAConnection()
        expiry, data = conn.get_with_expiry(f"/gridpoints/{self.coordinate}")
        attr = to_camel_case(self.attribute)

        polygon, = data["geometry"]["coordinates"]
        polygon = set(tuple(x) for x in polygon)
        polygon = list(zip(*polygon))
        center_coordinate = [sum(item) / len(item) for item in polygon]
        center_coordinate = MapCoordinate(*center_coordinate)
        data = data["properties"][attr]
        # Some layers are dimensionless (e.g. probabilityOfThunder) and omit
        # `uom` entirely; others use "wmoUnit:degC" style. Default to no unit.
        uom = data.get("uom", "")
        units = uom.split(":")[-1] if uom else ""
        data = data["values"]
        data = [
            (datetime.fromisoformat(d["validTime"].split("/")[0]), d["value"])
            for d in data
        ]
        # Drop entries with no observed value: a null doesn't mean zero for an
        # observation (e.g. temperature), it means "no data at this time", so
        # we omit the point and let the spline interpolate across the gap.
        data = [(a, b) for a, b in data if b is not None]
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
        if time > end_time or time < start_time:
            raise BoundsError
        t = (time - start_time).total_seconds()
        y = f(t).item()
        y = clamp_value(self.attribute, y)
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
        gridpoints = list(self.area.iter_gridpoints())
        # Batch-warm every gridpoint's data in a single concurrent writer job
        # BEFORE the serial predict loop below. Otherwise each predict()'s
        # cache miss would make its own blocking writer round-trip, one at a
        # time — which is the dominant source of slow resolution.
        try:
            from ..cachewriter import ensure as _ensure_cached
            _ensure_cached([f"/gridpoints/{gp}" for gp in gridpoints])
        except Exception:
            pass  # writer unavailable -> predict() falls back to inline fetch
        for coordinate in gridpoints:
            try:
                yield self[coordinate].predict(time, verbose=verbose)
            except BoundsError:
                # Requested time is outside this gridpoint's forecast horizon;
                # skip the point rather than failing the whole ensemble.
                continue
            except Exception:
                # A single gridpoint may fail to build (e.g. the layer is
                # absent here, has too few points to fit a spline, or has
                # malformed data). Skip it so one bad cell doesn't 500 the
                # whole area.
                logger.warning(
                    "skipping gridpoint %s for %s", coordinate, self.attribute,
                    exc_info=True,
                )
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

        Also returns the forecast window shared by all gridpoints (the latest
        start / earliest end — i.e. the INTERSECTION of per-gridpoint
        horizons), as ISO-8601 strings. Looping over this window keeps the
        whole area populated: gridpoints with shorter horizons won't drop out
        and shrink the contour field near the ends.

        Returns {"min", "max", "uom", "start", "end"} or None if there's no
        data. Requires the area's data to be cached (call /prefetch first) to
        be fast.
        """
        vmin = None
        vmax = None
        units = None
        start = None
        end = None
        for coordinate in self.area.iter_gridpoints():
            m = self[coordinate]
            try:
                start_time, end_time, _center, f, uom, _expiry = m.model
            except Exception:
                continue
            units = units or uom
            # Intersection of horizons: latest start, earliest end.
            if start is None or start_time > start:
                start = start_time
            if end is None or end_time < end:
                end = end_time
            total = (end_time - start_time).total_seconds()
            if total <= 0:
                continue
            for i in range(samples + 1):
                t = total * i / samples
                y = clamp_value(self.attribute, f(t).item())
                if vmin is None or y < vmin:
                    vmin = y
                if vmax is None or y > vmax:
                    vmax = y
        if vmin is None:
            return None
        # A valid shared window requires start < end; if per-gridpoint horizons
        # don't overlap, report no window (client then skips looping).
        window_ok = start is not None and end is not None and start < end
        return {
            "min": vmin,
            "max": vmax,
            "uom": units,
            "start": start.isoformat() if window_ok else None,
            "end": end.isoformat() if window_ok else None,
        }
