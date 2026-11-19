"""Fast lat/lon -> grid-cell lookup for GRIB Lambert Conformal grids.

NBM (and NDFD) publish on a projected Lambert Conformal grid, so a cell's lat/lon
isn't a simple function of its array index. The generic answer is a spatial
index over every cell's lat/lon (a scipy cKDTree over ~3.7M points: slow to
build, ~100+ MB, and measured in degrees rather than km). But the grid is a
perfectly regular lattice in PROJECTED coordinates, so we can instead project
the query lat/lon with pyproj and compute the row/column by arithmetic:

    x, y   = project(lon, lat)
    col    = round((x - x0) / dx)
    row    = round((y - y0) / dy)
    flat   = row * nx + col

That's O(1) per point, needs no per-cell arrays, and measures distance in
meters.

Safety: the projection parameters come from the GRIB message, but a wrong
earth-radius / scan-mode assumption would silently shift every lookup. So the
index VALIDATES itself against the message's own latitude/longitude arrays at a
sample of cells (every sampled cell must map back to itself, within a small
fraction of a cell). If anything is unsupported or disagrees, `index_from_message`
falls back to the old cKDTree, so behavior is never worse than before.

The returned object exposes `query(pts, k=1) -> (dist, idx)` (the cKDTree
interface the callers already use), where `pts` is an (N, 2) array of
[lat, lon] and `idx` is a flat index into the message's values array.
"""

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)

# Validated indexes by grid signature (see index_from_message): one NBM grid is
# shared by every element, so its lat/lon arrays need to be read just once.
_VALIDATED = {}
_VALIDATED_LOCK = threading.Lock()

# A sampled cell must map back to ITSELF, and the projected distance from the
# message's own lat/lon for that cell to our cell center must be within this
# fraction of a cell. 0.1 cell (~250 m at 2.5 km) catches a wrong earth radius
# (a 0.1% scale error drifts ~5 km across the grid) while allowing float noise.
_VALIDATE_TOL_CELLS = 0.1

# Spherical earth radii (m) for the GRIB2 shapeOfTheEarth codes we can handle.
# Code 1 carries its own radius in the message; ellipsoidal shapes aren't
# supported here (we fall back to the KD-tree for those).
_SHAPE_RADIUS = {0: 6367470.0, 6: 6371229.0}


def _norm_lon(lon):
    """Normalize longitude(s) to [-180, 180)."""
    return (np.asarray(lon, dtype=float) + 180.0) % 360.0 - 180.0


class LambertGridIndex:
    """O(1) nearest-cell lookup on a regular Lambert Conformal GRIB grid."""

    def __init__(self, *, nx, ny, dx, dy, lat1, lon1, lad, lov, latin1, latin2,
                 radius, i_negative=False, j_positive=True, j_consecutive=False):
        from pyproj import Transformer

        self.nx, self.ny = int(nx), int(ny)
        self.dx, self.dy = float(dx), float(dy)
        lcc = (f"+proj=lcc +lat_1={float(latin1)} +lat_2={float(latin2)} "
               f"+lat_0={float(lad)} +lon_0={float(_norm_lon(lov))} "
               f"+x_0=0 +y_0=0 +R={float(radius)} +units=m +no_defs")
        # A spherical lon/lat CRS with the SAME radius, so no datum shift is
        # applied between the geographic input and the projection.
        geo = f"+proj=longlat +R={float(radius)} +no_defs"
        self._fwd = Transformer.from_crs(geo, lcc, always_xy=True)
        self._inv = Transformer.from_crs(lcc, geo, always_xy=True)
        x0, y0 = self._fwd.transform(float(_norm_lon(lon1)), float(lat1))
        self._x0, self._y0 = float(x0), float(y0)
        # Scan direction: +1/-1 step in projected x per column / y per row.
        self._sx = -1.0 if i_negative else 1.0
        self._sy = 1.0 if j_positive else -1.0
        self._j_consecutive = bool(j_consecutive)
        # Identity of the grid this index describes. Two messages with the same
        # signature are the same grid, so a validation result can be reused.
        self.signature = (lcc, self.nx, self.ny, self.dx, self.dy,
                          self._x0, self._y0, self._sx, self._sy,
                          self._j_consecutive)

    # ---- construction from a GRIB message --------------------------------

    @classmethod
    def from_message(cls, get):
        """Build from a GRIB message via `get(key) -> value` (e.g.
        `lambda k: eccodes.codes_get(gid, k)`). Returns None when the grid type
        or earth shape isn't supported (caller falls back to a KD-tree)."""
        if str(get("gridType")) != "lambert":
            return None
        radius = _earth_radius(get)
        if radius is None:
            return None
        return cls(
            nx=get("Nx"), ny=get("Ny"),
            dx=get("DxInMetres"), dy=get("DyInMetres"),
            lat1=get("latitudeOfFirstGridPointInDegrees"),
            lon1=get("longitudeOfFirstGridPointInDegrees"),
            lad=get("LaDInDegrees"), lov=get("LoVInDegrees"),
            latin1=get("Latin1InDegrees"), latin2=get("Latin2InDegrees"),
            radius=radius,
            i_negative=bool(get("iScansNegatively")),
            j_positive=bool(get("jScansPositively")),
            j_consecutive=bool(get("jPointsAreConsecutive")),
        )

    # ---- lookup -----------------------------------------------------------

    def query(self, pts, k=1):
        """Nearest cell for each [lat, lon] row of `pts`. Returns (dist_m, idx)
        like cKDTree.query(k=1): `idx` is a flat index into the message's
        values array. Points outside the grid clamp to the nearest edge cell
        (matching the KD-tree's "always returns something" behavior)."""
        if k != 1:
            raise NotImplementedError("LambertGridIndex supports k=1 only")
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        lat, lon = pts[:, 0], pts[:, 1]
        finite = np.isfinite(lat) & np.isfinite(lon)
        x, y = self._fwd.transform(
            _norm_lon(np.where(finite, lon, 0.0)), np.where(finite, lat, 0.0))
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        col = np.clip(np.rint((x - self._x0) / (self._sx * self.dx)),
                      0, self.nx - 1).astype(np.int64)
        row = np.clip(np.rint((y - self._y0) / (self._sy * self.dy)),
                      0, self.ny - 1).astype(np.int64)
        cx = self._x0 + self._sx * col * self.dx
        cy = self._y0 + self._sy * row * self.dy
        dist = np.hypot(x - cx, y - cy)
        dist = np.where(finite, dist, np.inf)
        idx = (col * self.ny + row) if self._j_consecutive else (row * self.nx + col)
        return dist, idx

    # ---- self-check -------------------------------------------------------

    def latlon_of(self, rows, cols):
        """Lat/lon of grid positions (row, col), which may be FRACTIONAL (used
        to place densified points between cells). Returns (lats, lons) arrays
        with lons in [-180, 180)."""
        rows = np.asarray(rows, dtype=float)
        cols = np.asarray(cols, dtype=float)
        x = self._x0 + self._sx * cols * self.dx
        y = self._y0 + self._sy * rows * self.dy
        lon, lat = self._inv.transform(x, y)
        return np.asarray(lat, dtype=float), _norm_lon(lon)

    def box_cells(self, s, n, w, e, edge_samples=128):
        """Every grid cell in the row/column block that COVERS the lat/lon box
        [s..n] x [w..e] — the cells themselves, not nearest-neighbor samples.

        The grid is rotated relative to lat/lon (Lambert), so the box's edges
        are curves in grid space. We project points along all four edges and
        take the floor/ceil of their fractional row/col range. The extremes of
        a projected coordinate over the box lie on its boundary, so edge
        sampling bounds the whole box. The resulting block is a regular
        rows x cols grid (good for densifying) that extends up to about a cell
        past the box on each side, so edge interpolation has neighbors.

        Returns a GridBlock, or None when the box doesn't overlap the grid
        (rather than clamping to edge cells, which would return data from
        somewhere else entirely)."""
        s, n = sorted((float(s), float(n)))
        w, e = float(w), float(e)
        if not all(np.isfinite(v) for v in (s, n, w, e)):
            return None
        if e < w:                       # box crossing the antimeridian
            e += 360.0
        t = np.linspace(0.0, 1.0, int(edge_samples))
        lats = np.concatenate([np.full_like(t, s), np.full_like(t, n),
                               s + (n - s) * t, s + (n - s) * t])
        lons = np.concatenate([w + (e - w) * t, w + (e - w) * t,
                               np.full_like(t, w), np.full_like(t, e)])
        x, y = self._fwd.transform(_norm_lon(lons), lats)
        cf = (np.asarray(x, dtype=float) - self._x0) / (self._sx * self.dx)
        rf = (np.asarray(y, dtype=float) - self._y0) / (self._sy * self.dy)
        if not (np.all(np.isfinite(cf)) and np.all(np.isfinite(rf))):
            return None
        c0, c1 = int(np.floor(cf.min())), int(np.ceil(cf.max()))
        r0, r1 = int(np.floor(rf.min())), int(np.ceil(rf.max()))
        if c1 < 0 or c0 > self.nx - 1 or r1 < 0 or r0 > self.ny - 1:
            return None
        c0, c1 = max(c0, 0), min(c1, self.nx - 1)
        r0, r1 = max(r0, 0), min(r1, self.ny - 1)
        return GridBlock(self, r0, r1, c0, c1)

    # ---- self-check -------------------------------------------------------

    def validate(self, lats, lons, n_random=64, tol_cells=_VALIDATE_TOL_CELLS):
        """True if this index reproduces the message's own lat/lon arrays: for
        a sample of cells (corners, center, random), looking up that cell's
        lat/lon must return the same cell, within `tol_cells` of its center."""
        lats = np.asarray(lats, dtype=float)
        lons = np.asarray(lons, dtype=float)
        n = self.nx * self.ny
        if lats.size != n or lons.size != n:
            return False
        rng = np.random.default_rng(0)
        sample = np.unique(np.concatenate((
            [0, self.nx - 1, n - self.nx, n - 1, n // 2],
            rng.integers(0, n, size=n_random))))
        dist, idx = self.query(np.column_stack([lats[sample], lons[sample]]))
        return bool(np.all(idx == sample)
                    and np.all(dist <= tol_cells * min(self.dx, self.dy)))


class GridBlock:
    """A contiguous rows x cols block of a LambertGridIndex's grid.

    Attributes, all in ROW-MAJOR block order (row, then col):
      shape        (n_rows, n_cols)
      rows, cols   1-D integer grid indices of the block's rows / columns
      flat         (n,) flat indices into the GRIB message's values array
      lats, lons   (n,) cell-center coordinates
    """

    def __init__(self, index, r0, r1, c0, c1):
        self._index = index
        self.rows = np.arange(r0, r1 + 1, dtype=np.int64)
        self.cols = np.arange(c0, c1 + 1, dtype=np.int64)
        self.shape = (self.rows.size, self.cols.size)
        R, C = np.meshgrid(self.rows, self.cols, indexing="ij")
        if index._j_consecutive:
            flat = C * index.ny + R
        else:
            flat = R * index.nx + C
        self.flat = flat.ravel()
        lats, lons = index.latlon_of(R.ravel(), C.ravel())
        self.lats, self.lons = lats, lons

    def latlon(self, block_rows, block_cols):
        """Lat/lon of BLOCK-relative (row, col) positions, possibly fractional
        (block position 0 is the block's first row/col)."""
        return self._index.latlon_of(
            np.asarray(block_rows, dtype=float) + self.rows[0],
            np.asarray(block_cols, dtype=float) + self.cols[0])


def _scan_lines(get):
    """(n_lines, line_len, consecutive) for a message that uses ALTERNATING-ROW
    scanning, else None. A "line" is a row (or a column when
    jPointsAreConsecutive=1)."""
    try:
        if int(get("alternativeRowScanning")) != 1:
            return None
        nx, ny = int(get("Nx")), int(get("Ny"))
        consecutive = bool(int(get("jPointsAreConsecutive")))
    except Exception:
        return None
    return (nx, ny, True) if consecutive else (ny, nx, False)


def _unflip_alternate(arr, n_lines, length):
    """Reverse every other line (1, 3, 5, ...) of a flat scan-ordered array."""
    a = np.asarray(arr)
    if a.size != n_lines * length:
        return arr
    w = a.reshape(n_lines, length).copy()
    w[1::2] = w[1::2, ::-1]
    return w.ravel()


def uniform_scan_values(values, get):
    """Reorder a message's decoded values into the UNIFORM grid layout (every
    row running the same direction) that the grid index and eccodes' lat/lon
    arrays use.

    With alternativeRowScanning=1 (NBM and NDFD set it), GRIB stores every
    other row in REVERSE order, and eccodes' codes_get_values returns them in
    that stored order. Reading them as uniform rows pairs each odd row's values
    with the wrong cells — data from the other side of the grid — which shows
    up as row-aligned stripes on the map. `get(key)` reads message keys, e.g.
    `lambda k: eccodes.codes_get(gid, k)`. Values are returned unchanged when
    the flag is off."""
    lines = _scan_lines(get)
    if lines is None:
        return values
    n_lines, length, _ = lines
    return _unflip_alternate(values, n_lines, length)


def uniform_scan_latlon(lats, lons, get):
    """Make eccodes' latitude/longitude arrays uniform too, so they always pair
    with uniform_scan_values. eccodes 2.50 already returns these arrays uniform
    (it doesn't apply alternating-row scanning to them); this only acts if a
    version returns them in alternating order, detected by the second line
    running opposite to the first."""
    lines = _scan_lines(get)
    if lines is None:
        return lats, lons
    n_lines, length, consecutive = lines
    lats = np.asarray(lats)
    lons = np.asarray(lons)
    if n_lines < 2 or length < 2 or lats.size != n_lines * length:
        return lats, lons
    # Along a row longitude varies; along a column (consecutive) latitude does.
    coord = (lats if consecutive else lons).reshape(n_lines, length)
    d0 = coord[0, -1] - coord[0, 0]
    d1 = coord[1, -1] - coord[1, 0]
    if np.sign(d0) == np.sign(d1):
        return lats, lons            # already uniform
    return (_unflip_alternate(lats, n_lines, length),
            _unflip_alternate(lons, n_lines, length))


def _earth_radius(get):
    """Spherical earth radius (m) for the message, or None if the earth shape
    isn't a sphere we handle (ellipsoids -> KD-tree fallback)."""
    try:
        shape = int(get("shapeOfTheEarth"))
    except Exception:
        return None
    if shape == 1:
        try:
            scale = int(get("scaleFactorOfRadiusOfSphericalEarth"))
            val = float(get("scaledValueOfRadiusOfSphericalEarth"))
            return val * (10.0 ** -scale)
        except Exception:
            return None
    return _SHAPE_RADIUS.get(shape)


def index_from_message(gid):
    """Return a lat/lon -> flat-index lookup for the open eccodes message `gid`
    exposing `.query(pts, k=1) -> (dist, idx)`.

    Prefers the pyproj-based LambertGridIndex (validated against the message's
    own lat/lon arrays); falls back to a scipy cKDTree over every cell's lat/lon
    when pyproj is missing, the grid isn't a supported Lambert grid, or the
    projection fails validation.

    Validation needs the message's full lat/lon arrays (eccodes computes ~3.7M
    cells each, ~1 s), but every NBM element shares one grid. So a validated
    index is cached by grid signature, and later messages on the same grid skip
    reading those arrays entirely."""
    import eccodes as ec

    cand = None
    try:
        cand = LambertGridIndex.from_message(lambda k: ec.codes_get(gid, k))
        sig = getattr(cand, "signature", None)
        if sig is not None:
            with _VALIDATED_LOCK:
                cached = _VALIDATED.get(sig)
            if cached is not None:
                return cached
    except ImportError:
        logger.warning("pyproj not installed; using a (slower) KD-tree grid "
                       "index. Install pyproj for O(1) lookups.")
        cand = None
    except Exception:
        logger.warning("Lambert grid index unavailable; falling back to a "
                       "KD-tree", exc_info=True)
        cand = None

    lats = np.asarray(ec.codes_get_array(gid, "latitudes"))
    lons = np.asarray(ec.codes_get_array(gid, "longitudes"))
    lons = np.where(lons > 180, lons - 360, lons)
    lats, lons = uniform_scan_latlon(lats, lons, lambda k: ec.codes_get(gid, k))

    if cand is not None:
        try:
            if cand.validate(lats, lons):
                sig = getattr(cand, "signature", None)
                if sig is not None:
                    with _VALIDATED_LOCK:
                        _VALIDATED[sig] = cand
                return cand
            logger.warning("Lambert grid index failed validation against the "
                           "message's lat/lon; falling back to a KD-tree")
        except Exception:
            logger.warning("Lambert grid index validation errored; falling "
                           "back to a KD-tree", exc_info=True)

    from scipy.spatial import cKDTree
    return cKDTree(np.column_stack([lats, lons]))
