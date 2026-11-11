"""NDFD ensemble adapters — a structural drop-in for the JSON-backed
ObservationEnsemble / AccumulationEnsemble.

The server endpoints (`/models/{model}/type/range`, `/type/series/stream`,
`/type/ensemble`) and the bundle builder all operate through an ensemble object
with this contract:

    ensemble[gp]            -> a per-point model `m` where:
                                 m.model -> (start_dt, end_dt, center_coord,
                                             f(secs_since_start)->value,
                                             units, expiry_dt)
                                 m.predict(datetime, verbose=False) -> value
                                 m.is_expired -> bool
    ensemble.iter_gridpoints via its .area
    ensemble.iter_predict(time, verbose) -> yields GeoJSON features
    ensemble.predict(time, dt, verbose)  -> FeatureCollection
    ensemble.value_range(samples)        -> {min,max,uom,start,end}

Rather than NOAA gridpoints, NDFD samples a GRIB grid. We fit the WHOLE area at
once (lib.ndfd.fit_area builds a per-lattice-point CubicSpline), then wrap each
lattice point in an ObservationModel-SHAPED object so every consumer works
unchanged. The ensemble's "gridpoints" are therefore the lattice points of the
area (its map_coordinates), not NOAA WFO gridpoints — which is why NDFD needs
NO /points resolution, no polygon index, no decomposition.

Selected by WR_WEATHER_SOURCE=ndfd (see models.weather_ensembles).
"""

import logging
from datetime import datetime, timedelta

from .utils import BoundsError, clamp_value
from ..area import MapCoordinate

logger = logging.getLogger(__name__)


class _NDFDPointModel:
    """ObservationModel-shaped wrapper around one fitted lattice-point spline.

    `f` maps SECONDS-SINCE-START -> display value (to match the JSON model's
    `f(t)` contract, where t is seconds from start_time). The underlying NDFD
    spline is keyed on absolute epoch seconds, so we shift by `start_epoch`."""

    def __init__(self, center: MapCoordinate, spline, start_epoch, end_epoch,
                 units, nonneg, attribute, source_mtime):
        self._center = center
        self._spline = spline
        self._start_epoch = start_epoch
        self._end_epoch = end_epoch
        self._units = units
        self._nonneg = nonneg
        self.attribute = attribute
        self._source_mtime = source_mtime
        self._model_cache = None   # memoized `model` tuple (see below)

    @property
    def is_expired(self):
        # Freshness is handled at the ensemble/fit level (grid mtime); an
        # individual wrapped point is never independently refetched, so it is
        # considered current for its lifetime. The ensemble is rebuilt when the
        # underlying grid changes.
        return False

    @property
    def model(self):
        # Build tz-AWARE bounds (local zone), matching the JSON model whose
        # start/end come from ISO-8601 strings with offsets. predict() compares
        # against time.astimezone() (aware), so naive bounds would raise.
        #
        # MEMOIZED: this tuple is invariant for the life of the point (the
        # spline/bounds/units don't change), but it is read once per point PER
        # FRAME in the hot series_stream loop. Rebuilding it there (two
        # astimezone() calls, a datetime.now(), a timedelta, and a fresh
        # closure) every (point × epoch) time dominated large-area sweeps.
        # Compute it once and hand back the cached tuple.
        if self._model_cache is not None:
            return self._model_cache

        start_dt = datetime.fromtimestamp(self._start_epoch).astimezone()
        end_dt = datetime.fromtimestamp(self._end_epoch).astimezone()
        # Far-future expiry: freshness is grid-mtime driven, enforced upstream.
        expiry = datetime.now() + timedelta(days=365)

        start_epoch = self._start_epoch
        spline = self._spline
        nonneg = self._nonneg

        def f(secs_since_start):
            t_epoch = start_epoch + float(secs_since_start)
            y = float(spline(t_epoch))
            if nonneg and y < 0:
                y = 0.0
            return _Scalar(y)

        self._model_cache = (start_dt, end_dt, self._center, f, self._units, expiry)
        return self._model_cache

    def predict_epoch(self, epoch_seconds):
        """Fast per-frame evaluation keyed on an ABSOLUTE epoch (int/float
        seconds). Skips the datetime/timezone churn of `predict(datetime)`:
        the NDFD spline is itself keyed on absolute epoch, so bounds are a
        plain numeric compare and the value is a single spline eval + clamp.
        Raises BoundsError when the epoch is outside this point's horizon.

        Equivalent to predict(datetime.fromtimestamp(epoch)) but ~constant-time
        with no allocation — used by the hot series_stream sweep where this is
        called once per point per frame."""
        if epoch_seconds < self._start_epoch or epoch_seconds > self._end_epoch:
            raise BoundsError
        y = float(self._spline(epoch_seconds))
        if self._nonneg and y < 0:
            y = 0.0
        return clamp_value(self.attribute, y)

    def predict(self, time, verbose=False):
        if type(time) is not datetime:
            return self.predict_many(time, verbose=verbose)
        start_dt, end_dt, center, f, units, _exp = self.model
        time = time.astimezone()
        if time > end_dt or time < start_dt:
            raise BoundsError
        t = (time - start_dt).total_seconds()
        y = f(t).item()
        y = clamp_value(self.attribute, y)
        if verbose:
            return {
                "type": "Feature",
                "properties": {self.attribute: y, "time": time.isoformat(),
                               "uom": units},
                "geometry": {"type": "Point",
                             # [lon, lat] (GeoJSON; normalizeFrame reads
                             # coordinates[0]=lon). center is an honest
                             # MapCoordinate (.lat is a true latitude).
                             "coordinates": [center.lon, center.lat]},
            }
        return y

    def predict_many(self, times, verbose=False):
        features = []
        for time in times:
            try:
                features.append(self.predict(time, verbose=verbose))
            except BoundsError:
                break
        return {"type": "FeatureCollection", "features": features}


class _Scalar(float):
    """A float that also answers `.item()` — the JSON model's `f(t)` returns a
    numpy scalar (so callers do `f(t).item()`); mirror that here so our `f`
    is interchangeable."""
    def item(self):
        return float(self)


class NDFDObservationEnsemble:
    """Area ensemble backed by NDFD GRIB instead of NOAA /gridpoints JSON.

    Structurally compatible with ObservationEnsemble: same __getitem__,
    iter_predict, predict, value_range. The whole area is fit ONCE (lazily, on
    first access) into per-lattice-point splines; `__getitem__` then hands back
    the wrapped point. Gridpoints are the area's lattice points (its
    map_coordinates)."""

    # Observation mode: spline the raw per-interval values. The accumulation
    # subclass overrides this to integrate into a cumulative total.
    _accumulate = False

    def __init__(self, area, attribute):
        self.area = area
        self.attribute = attribute
        self._fit = None
        self._models = None      # MapCoordinate(lat,lon) key -> _NDFDPointModel
        self._fit_done = False

    def _ensure_fit(self):
        if self._fit_done:
            return
        self._fit_done = True
        from .. import ndfd
        # fit_area_dense spatially densifies the output lattice by
        # WR_NDFD_UPSAMPLE (default 1 = identical to fit_area). The returned
        # dict shape is unchanged, so wrapping/consumers below are untouched.
        fit = ndfd.fit_area_dense(self.area, self.attribute,
                                  accumulate=self._accumulate)
        self._fit = fit
        self._models = {}
        if fit is None:
            return
        src_mtime = ndfd.grid_mtime(self.attribute)
        for center, spline in zip(fit["points"], fit["splines"]):
            m = _NDFDPointModel(center, spline, fit["start"], fit["end"],
                                fit["units"], fit["nonneg"], self.attribute,
                                src_mtime)
            self._models[(round(center.lat, 6), round(center.lon, 6))] = m

    def iter_points(self):
        """Yield the wrapped per-lattice-point models (NDFD's analog of
        iterating resolved gridpoints)."""
        self._ensure_fit()
        if not self._models:
            return
        yield from self._models.values()

    def __getitem__(self, coordinate):
        """Accept either a MapCoordinate (NDFD lattice point) or anything with
        .lat/.lon. Returns the wrapped model, or raises KeyError."""
        self._ensure_fit()
        if not self._models:
            raise KeyError(coordinate)
        key = (round(coordinate.lat, 6), round(coordinate.lon, 6))
        return self._models[key]

    def iter_predict(self, time, verbose=False):
        for m in self.iter_points():
            try:
                yield m.predict(time, verbose=verbose)
            except BoundsError:
                continue
            except Exception:
                logger.warning("NDFD skipping point for %s", self.attribute,
                               exc_info=True)
                continue

    def predict(self, time, dt=0, verbose=False):
        return {"type": "FeatureCollection",
                "features": list(self.iter_predict(time, verbose=verbose))}

    def value_range(self, samples: int = 24):
        """Global min/max over the forecast window across all lattice points,
        plus the shared window (start/end). NDFD points all share the SAME
        window (one grid), so the intersection is just that window."""
        self._ensure_fit()
        if not self._fit or not self._models:
            return None
        vmin = vmax = None
        units = self._fit["units"]
        start = datetime.fromtimestamp(self._fit["start"])
        end = datetime.fromtimestamp(self._fit["end"])
        total = (end - start).total_seconds()
        if total <= 0:
            return None
        for m in self._models.values():
            _s, _e, _c, f, _u, _x = m.model
            for i in range(samples + 1):
                t = total * i / samples
                y = clamp_value(self.attribute, f(t).item())
                if vmin is None or y < vmin:
                    vmin = y
                if vmax is None or y > vmax:
                    vmax = y
        if vmin is None:
            return None
        return {"min": vmin, "max": vmax, "uom": units,
                "start": start.isoformat(), "end": end.isoformat()}


class NDFDAccumulationEnsemble(NDFDObservationEnsemble):
    """NDFD accumulation ensemble.

    NDFD accumulation elements (qpf/snow/ice) are published as PER-INTERVAL
    totals per message (the fall during each forecast period). To match the
    JSON AccumulationEnsemble's cumulative "storm total" semantic, this
    ensemble integrates each point's per-interval series into a RUNNING TOTAL
    from the forecast window start and fits it with a monotone interpolator
    (see ndfd.fit_area(accumulate=True)). The result is non-decreasing over
    time — a value at t+dt is always >= the value at t — so "Snowfall (total)"
    reads as an accumulation, not a per-interval rate.
    """
    _accumulate = True
