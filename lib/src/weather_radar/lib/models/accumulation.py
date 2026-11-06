from datetime import datetime
from functools import cached_property
from threading import Lock

import logging

from weather_radar.lib.area import CoordinateArea, MapCoordinate, NOAAGridpoint, gridpoint_from_map_coordinate
from ..connection import NOAAConnection

from scipy import interpolate

from .utils import BoundsError, to_camel_case, source_mtime


logger = logging.getLogger(__name__)


# NOAA reports accumulation attributes (snowfall, precipitation) in
# millimetres (wmoUnit:mm). We display inches instead. Map the native unit to
# a (factor, display-unit) pair; unknown units pass through unchanged.
MM_TO_IN = 1.0 / 25.4
UNIT_CONVERSIONS = {
    "mm": (MM_TO_IN, "in"),
}


# Process-wide cache of fitted models, keyed by (gridpoint, attribute).
# Fitting a spline requires reading/parsing the gridpoint JSON and running
# scipy interpolation, which is expensive to redo on every request. The
# underlying NOAA data only changes on the order of hours (see the `Expires`
# header), so cached models are reused until that data expires.
_MODEL_CACHE: dict[tuple, "AccumulationModel"] = {}
_MODEL_CACHE_LOCK = Lock()


def get_accumulation_model(coordinate: NOAAGridpoint, attribute: str) -> "AccumulationModel":
    """Return a cached AccumulationModel, rebuilding it only when the
    underlying NOAA data has expired."""
    key = (coordinate, attribute)
    with _MODEL_CACHE_LOCK:
        model = _MODEL_CACHE.get(key)
        if model is not None and not model.is_expired:
            return model
        model = AccumulationModel(coordinate, attribute)
        _MODEL_CACHE[key] = model
        return model


class AccumulationModel:
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
        been UPDATED on disk since the fit. See ObservationModel.is_expired —
        invalidation is keyed on the source cache file's mtime (a republish
        signal), not wall-clock Expires, so unchanged data isn't needlessly
        refit (×every worker)."""
        if "model" not in self.__dict__:
            return True
        current = source_mtime(f"/gridpoints/{self.coordinate}")
        if current is None:
            expiry = self.__dict__["model"][-1]
            return expiry < datetime.now()
        return current != self._source_mtime

    @cached_property
    def model(self):
        """Spline of a cumulative model, derivative is accumulator at a
        given time"""

        # Capture the source file's mtime before reading (see
        # ObservationModel.model) so is_expired can detect a later republish.
        self._source_mtime = source_mtime(f"/gridpoints/{self.coordinate}")

        conn = NOAAConnection()
        expiry, data = conn.get_with_expiry(f"/gridpoints/{self.coordinate}")

        polygon, = data["geometry"]["coordinates"]
        polygon = set(tuple(x) for x in polygon)
        polygon = list(zip(*polygon))
        # [mean_lon, mean_lat]; build the center with correct axes (see the
        # matching note in ObservationModel.model — this fixes the old
        # swapped-axis construction).
        means = [sum(item) / len(item) for item in polygon]
        center_coordinate = MapCoordinate(lat=means[1], lon=means[0])
        data = data["properties"][to_camel_case(self.attribute)]
        # Some layers omit `uom` (dimensionless); default to no unit.
        uom = data.get("uom", "")
        units = uom.split(":")[-1] if uom else ""
        # Convert from NOAA's native unit (mm) to our display unit (inches).
        factor, units = UNIT_CONVERSIONS.get(units, (1.0, units))
        data = data["values"]
        data = [
            (datetime.fromisoformat(d["validTime"].split("/")[0]), d["value"])
            for d in data
        ]
        start_time = data[0][0]
        end_time = data[-1][0]
        # A null value means "no accumulation reported for this interval", so
        # treat it as 0 when summing. (Summing a None would raise TypeError and
        # silently drop the whole gridpoint.) Values are scaled to the display
        # unit as they're summed.
        model_data = [
            ((a-start_time).total_seconds(),
             sum((d[1] or 0) for d in data[:i+1]) * factor)
            for i, (a, _) in enumerate(data)
        ]
        model_data = list(zip(*model_data))
        x, y = model_data
        # `make_interp_spline` returns a BSpline that *interpolates* the given
        # (time, cumulative-precip) points. Using `BSpline(x, y, k=3)` directly
        # is incorrect: it treats `x` as the knot vector and `y` as spline
        # coefficients, so the resulting curve does not pass through the data
        # and yields meaningless accumulation values.
        model = interpolate.make_interp_spline(x, y, k=3)
        return start_time, end_time, center_coordinate, model, units, expiry

    def predict(self, time, verbose=False):
        if type(time) is not datetime:
            return self.predict_many(time, verbose=verbose)

        start_time, end_time, center_coordinate, f, units, _expiry = self.model
        time = time.astimezone()
        if time > end_time or time < start_time:
            raise BoundsError
        t = (time - start_time).total_seconds()
        time = time.replace(tzinfo=None)
        y = f(t).item()
        if y < 0:
            y = 0
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
                    # [lon, lat] (GeoJSON; normalizeFrame reads coordinates[0]=lon).
                    "coordinates": [center_coordinate.lon, center_coordinate.lat]
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



class AccumulationEnsemble:
    def __init__(self, area: CoordinateArea, attribute: str):
        self.area = area
        self.attribute = attribute

    def __getitem__(self, coordinate: NOAAGridpoint):
        # Use the process-wide model cache so fitted splines are reused across
        # requests (e.g. the map's polling loop) instead of being rebuilt from
        # disk on every call.
        return get_accumulation_model(coordinate, self.attribute)

    def iter_predict(self, time: datetime, verbose=False):
        """Yield one GeoJSON feature per gridpoint as it is computed, so
        callers can stream/render results progressively instead of waiting
        for the whole area. Gridpoints are resolved nearest-to-center first
        and emitted as they become available. Out-of-horizon points are
        skipped."""
        gridpoints = list(self.area.iter_gridpoints())
        # Batch-warm all gridpoint data concurrently via the writer before the
        # serial predict loop, so per-gridpoint cache misses don't each make
        # their own blocking writer round-trip (the main cause of slowness).
        try:
            from ..cachewriter import ensure as _ensure_cached
            _ensure_cached([f"/gridpoints/{gp}" for gp in gridpoints])
        except Exception:
            pass
        for coordinate in gridpoints:
            try:
                yield self[coordinate].predict(time, verbose=verbose)
            except BoundsError:
                # Requested time is outside this gridpoint's forecast horizon;
                # skip the point rather than failing the whole ensemble.
                continue
            except Exception:
                # A single gridpoint may fail to build (layer absent here, too
                # few points for a spline, malformed data). Skip it so one bad
                # cell doesn't 500 the whole area.
                logger.warning(
                    "skipping gridpoint %s for %s", coordinate, self.attribute,
                    exc_info=True,
                )
                continue

    def predict(self, time: datetime, dt=0, verbose=False):
        # `dt` is accepted for API compatibility but is only meaningful when
        # expanding a single time into a series of times (see
        # `datetime_from_time`). Once a concrete `time` is resolved, each
        # per-gridpoint model prediction does not use `dt`.
        return {
            "type": "FeatureCollection",
            "features": list(self.iter_predict(time, verbose=verbose)),
        }

    def value_range(self, samples: int = 24):
        """Global min/max of the predicted value over the entire forecast
        lifecycle across all gridpoints (see ObservationEnsemble.value_range).
        Negative accumulation values are clamped to 0 to match predict().

        Also returns the forecast window shared by all gridpoints (the latest
        start / earliest end — i.e. the INTERSECTION of per-gridpoint
        horizons), as ISO-8601 strings, so looping keeps the whole area
        populated instead of shrinking as shorter-horizon gridpoints drop out.

        Returns {"min", "max", "uom", "start", "end"} or None.
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
                y = f(t).item()
                if y < 0:
                    y = 0
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
