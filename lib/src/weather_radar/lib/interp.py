"""Shared stage-2 SPATIAL densification for gridded forecast fits.

Both the NDFD (lib/ndfd.py) and AQI (lib/aqi.py) sources fit one 1-D temporal
interpolator PER native lattice point (stage 1): a `f(epoch) -> value` spline.
That handles time (irregular knots, fill/no-data) per point but leaves spatial
resolution at the native grid — the client then IDW-rasters it.

This module adds the OPTIONAL stage 2: a time-then-space separable upsample.
Given the already-fitted native per-point splines (on a REGULAR lat/lon
lattice), it:

  1. evaluates every native spline at every native knot epoch -> a value grid
     per knot (holes where a point had no spline are neighbor-filled so the
     grid is complete — a regular-grid interpolator can't have holes);
  2. cubic-interpolates each knot's grid onto a FINER lattice (upsample k per
     native interval, endpoint-inclusive), giving each dense point a value
     series over the native knots;
  3. re-fits a temporal interpolator per dense point via the caller's own
     `fit_fn` (so NDFD/AQI keep their exact fitting semantics).

The native lattice axes are recovered from the point coordinates themselves
(the lattice is regular: coord = index*step), so this needs no `area` object
and works for either source's point representation.

At upsample <= 1 the caller should skip this entirely (identity).
"""

import os
import logging

logger = logging.getLogger(__name__)

# Fill/no-data sentinel used when a native point has no spline (so its grid
# cell is a hole to be neighbor-filled before spatial interpolation). Large
# enough to be unambiguous; never returned to callers (filled before use).
_HOLE = float("nan")


def upsample_env(var_name, default=1):
    """Read an integer upsample factor from an env var (>= 1)."""
    try:
        k = int(os.environ.get(var_name, str(default)))
    except (TypeError, ValueError):
        return 1
    return max(k, 1)


# Default fill/no-data sentinel for the grid sources (NDFD/NBM both use 9999).
FILL = 9999.0


def fit_column(raw, epochs, start_epoch, convert, accumulate, fill=FILL):
    """SHARED stage-1 temporal fit for one lattice point — the single
    implementation used by the NDFD, NBM, and AQI sources so their fitting
    semantics can't drift.

    Fits `raw` (values over absolute-epoch `epochs`) to an interpolator mapping
    epoch -> display value, or None if it can't be fit:
      * observation (accumulate=False): spline the raw per-interval/instant
        values (CubicSpline). `fill` sentinels are masked; needs >= 2 valid.
      * accumulation (accumulate=True): integrate a cumulative total from the
        window start (fill intervals contribute 0), fit with a MONOTONE
        PchipInterpolator so the running total never dips between knots. A 0 is
        anchored at `start_epoch`. Needs >= 1 valid sample.
    `convert` (or None) maps native GRIB value -> display units.
    """
    import numpy as np
    from scipy.interpolate import CubicSpline, PchipInterpolator

    raw = np.asarray(raw, dtype=float)
    valid = raw < (fill - 1.0)       # mask fill/no-data

    if accumulate:
        # Cumulative total from the window start. Fill intervals contribute 0
        # (no reported fall) so the running sum stays defined and monotonic.
        if valid.sum() < 1:
            return None
        amounts = np.where(valid, raw, 0.0).astype(float)
        if convert is not None:
            amounts = np.array([convert(v) for v in amounts], dtype=float)
        amounts = np.clip(amounts, 0.0, None)      # per-interval amounts >= 0
        cumulative = np.cumsum(amounts)
        # Anchor a 0 at the window start so the total begins at zero.
        xe = np.concatenate(([float(start_epoch)], epochs))
        yv = np.concatenate(([0.0], cumulative))
        # Strictly increasing knots (drop a duplicate first-frame-at-start).
        keep = np.concatenate(([True], np.diff(xe) > 0))
        xe, yv = xe[keep], yv[keep]
        if xe.size < 2:
            return None
        try:
            # PCHIP: monotone on monotone data -> cumulative never dips.
            return PchipInterpolator(xe, yv)
        except Exception:
            return None

    if valid.sum() < 2:
        return None
    xe = epochs[valid]
    yv = raw[valid]
    if convert is not None:
        yv = np.array([convert(v) for v in yv], dtype=float)
    else:
        yv = yv.astype(float)
    try:
        return CubicSpline(xe, yv)
    except Exception:
        return None


def densify_axis(axis, k):
    """Return a finer 1-D axis with k sub-samples per native interval,
    endpoint-inclusive and strictly increasing. k<=1 returns axis unchanged."""
    import numpy as np
    if k <= 1 or axis.size < 2:
        return axis
    segs = [np.linspace(axis[i], axis[i + 1], k + 1)[:-1]
            for i in range(axis.size - 1)]
    segs.append(axis[-1:])
    return np.concatenate(segs)


def _fill_holes(grid):
    """Replace NaN cells in a 2-D grid with nearest valid values so the grid is
    complete for spatial interpolation. All-NaN grids are returned unchanged
    (the temporal re-fit masks them downstream)."""
    import numpy as np
    g = np.array(grid, dtype=float)
    bad = ~np.isfinite(g)
    if not bad.any():
        return g
    if bad.all():
        return g
    try:
        from scipy.ndimage import distance_transform_edt
        _, (ii, jj) = distance_transform_edt(bad, return_indices=True)
        g[bad] = g[ii[bad], jj[bad]]
    except Exception:
        g[bad] = float(g[~bad].mean())
    return g


_cubic_warned = False


def _resample_grid(lat_ax, lon_ax, grid, query):
    """Cubic-resample a complete (lat x lon) grid onto `query` [[lat,lon],...].
    Falls back to linear if this scipy lacks N-D cubic RegularGridInterpolator."""
    global _cubic_warned
    from scipy.interpolate import RegularGridInterpolator
    try:
        rgi = RegularGridInterpolator((lat_ax, lon_ax), grid, method="cubic",
                                      bounds_error=False, fill_value=None)
        return rgi(query)
    except (ValueError, NotImplementedError):
        if not _cubic_warned:
            logger.warning("densify: cubic RegularGridInterpolator unavailable; "
                           "falling back to linear")
            _cubic_warned = True
        rgi = RegularGridInterpolator((lat_ax, lon_ax), grid, method="linear",
                                      bounds_error=False, fill_value=None)
        return rgi(query)


def densify_fit(lats, lons, epochs, native_splines, fit_fn, k):
    """Spatially densify a stage-1 fit by factor `k`.

    Args:
      lats, lons      : native point coordinates, parallel arrays length
                        n_points, in row-major (lat-major) lattice order
                        (coord[p] = (lats[p], lons[p])).
      epochs          : (n_hours,) native knot epochs (absolute seconds).
      native_splines  : list length n_points of callables epoch->value, or
                        None for a point with no data.
      fit_fn          : callable(values_over_epochs: np.ndarray, epochs:
                        np.ndarray) -> spline|None. Re-fits one dense point's
                        series; the caller supplies its own semantics (e.g.
                        CubicSpline with a non-negative clamp). Values are the
                        spatially-interpolated knot values for that dense point.
      k               : upsample factor (>= 2; callers skip at <= 1).

    Returns (dense_lats, dense_lons, dense_splines) where dense_lats/lons are
    parallel arrays in row-major order and dense_splines is the parallel list
    (None where fit_fn declined). Returns None if the native grid is not a
    usable >=2x2 regular lattice.
    """
    import numpy as np

    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    epochs = np.asarray(epochs, dtype=float)
    n_pts = lats.size
    if n_pts == 0 or epochs.size == 0:
        return None

    lat_ax = np.unique(lats)
    lon_ax = np.unique(lons)
    n_lat, n_lon = lat_ax.size, lon_ax.size
    if n_lat < 2 or n_lon < 2 or n_lat * n_lon != n_pts:
        # Not a clean regular lattice (or points were dropped) -> can't reshape.
        return None

    # Map each native point to its (i, j) in the lattice so we can place its
    # per-knot values into the grid. Row-major: i = lat index, j = lon index.
    lat_idx = {round(float(v), 9): i for i, v in enumerate(lat_ax)}
    lon_idx = {round(float(v), 9): j for j, v in enumerate(lon_ax)}

    n_hours = epochs.size
    # native_grid[h] is (n_lat, n_lon); NaN where a point has no spline.
    native_grid = np.full((n_hours, n_lat, n_lon), _HOLE, dtype=float)
    for p in range(n_pts):
        sp = native_splines[p]
        if sp is None:
            continue
        i = lat_idx[round(float(lats[p]), 9)]
        j = lon_idx[round(float(lons[p]), 9)]
        try:
            vals = np.asarray(sp(epochs), dtype=float)
        except Exception:
            continue
        native_grid[:, i, j] = vals

    # Dense output lattice.
    dense_lat = densify_axis(lat_ax, k)
    dense_lon = densify_axis(lon_ax, k)
    dgy, dgx = np.meshgrid(dense_lat, dense_lon, indexing="ij")
    query = np.column_stack([dgy.ravel(), dgx.ravel()])
    n_dense = query.shape[0]

    # Stage 2 per knot: complete the grid (neighbor-fill holes) and resample.
    dense_series = np.empty((n_hours, n_dense), dtype=float)
    for h in range(n_hours):
        g = _fill_holes(native_grid[h])
        dense_series[h] = _resample_grid(lat_ax, lon_ax, g, query)

    # Stage 1 (re-fit) on each dense point via the caller's fit_fn.
    dense_lats = query[:, 0]
    dense_lons = query[:, 1]
    dense_splines = [fit_fn(dense_series[:, c], epochs) for c in range(n_dense)]
    return dense_lats, dense_lons, dense_splines
