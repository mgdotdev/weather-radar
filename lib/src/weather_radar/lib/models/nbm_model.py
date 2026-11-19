"""NBM ensemble adapters — structural drop-ins for the JSON-backed
ObservationEnsemble / AccumulationEnsemble, identical in contract to the NDFD
adapters (models.ndfd_model).

NBM samples a national GRIB grid (lib.nbm) rather than NOAA /gridpoints, so —
like NDFD — the ensemble's "gridpoints" are the area's lattice points
(map_coordinates), needing NO /points resolution. The per-point wrapper is the
source-agnostic _NDFDPointModel (it just wraps a spline + bounds + units), so
it is reused here as the common point model.

Selected by WR_WEATHER_SOURCE=nbm (see models.ensemble_kinds).
"""

import logging

from .utils import BoundsError, clamp_value
from .ndfd_model import _NDFDPointModel

logger = logging.getLogger(__name__)


class NBMObservationEnsemble:
    """Area ensemble backed by NBM GRIB. Structurally compatible with
    ObservationEnsemble: same __getitem__, iter_points, iter_predict, predict,
    value_range. The whole area is fit ONCE (lazily) into per-lattice-point
    splines; gridpoints are the area's lattice points."""

    _accumulate = False

    def __init__(self, area, attribute):
        self.area = area
        self.attribute = attribute
        self._fit = None
        self._models = None
        self._fit_done = False

    def _ensure_fit(self):
        if self._fit_done:
            return
        self._fit_done = True
        from .. import nbm
        fit = nbm.fit_area_dense(self.area, self.attribute,
                                 accumulate=self._accumulate)
        self._fit = fit
        self._models = {}
        if fit is None:
            return
        src_mtime = nbm.grid_mtime(self.attribute)
        for center, spline in zip(fit["points"], fit["splines"]):
            m = _NDFDPointModel(center, spline, fit["start"], fit["end"],
                                fit["units"], fit["nonneg"], self.attribute,
                                src_mtime)
            self._models[(round(center.lat, 6), round(center.lon, 6))] = m

    def iter_points(self):
        self._ensure_fit()
        if not self._models:
            return
        yield from self._models.values()

    def __getitem__(self, coordinate):
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
                logger.warning("NBM skipping point for %s", self.attribute,
                               exc_info=True)
                continue

    def predict(self, time, dt=0, verbose=False):
        return {"type": "FeatureCollection",
                "features": list(self.iter_predict(time, verbose=verbose))}

    def value_range(self, samples: int = 24):
        from datetime import datetime, timezone
        self._ensure_fit()
        if not self._fit or not self._models:
            return None
        vmin = vmax = None
        units = self._fit["units"]
        # UTC-AWARE datetimes from the fit's epoch bounds. Emitting UTC (not a
        # naive server-local stamp) keeps the window start/end unambiguous so
        # the client localizes them to the user's zone (see the graph/map
        # "send UTC, localize in browser" convention). `total` is a duration,
        # so the zone doesn't affect it.
        start = datetime.fromtimestamp(self._fit["start"], tz=timezone.utc)
        end = datetime.fromtimestamp(self._fit["end"], tz=timezone.utc)
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


class NBMAccumulationEnsemble(NBMObservationEnsemble):
    """NBM accumulation ensemble (ASNOW/APCP). NBM publishes these as
    PER-INTERVAL amounts per forecast hour; this integrates each point's series
    into a cumulative running total from the window start, fit with a monotone
    interpolator (see nbm.fit_area accumulate=True) — a non-decreasing storm
    total, matching the JSON AccumulationEnsemble semantic."""
    _accumulate = True
