import os
import concurrent.futures
import multiprocessing
import threading
import math
import json
import logging
import time
from functools import cached_property

from .connection import NOAAConnection, PathObj, get_and_write, NoCoverageError

logger = logging.getLogger(__name__)

# When set, log per-phase timing of an area's cold resolve+cache (points warm,
# gridpoint resolution, gridpoints fetch) at WARNING, so we can measure where
# large-area load time actually goes and tune NOAA_MAX_CONCURRENCY. Off by
# default.
_RESOLVE_TIMING = bool(os.environ.get("WR_RESOLVE_TIMING"))

lock = multiprocessing.Lock()

# Maximum simultaneous requests to the NOAA API. NOAA enforces undisclosed
# rate limits and returns HTTP 429 when overwhelmed; we respect 429/Retry-After
# (see connection.retry_get) so this ceiling can be fairly generous — if we
# push too hard NOAA throttles and we automatically back off. Env-overridable.
NOAA_MAX_CONCURRENCY = int(os.environ.get("NOAA_MAX_CONCURRENCY", "48"))

# Spacing (km) between sample coordinates when resolving an area's gridpoints.
# NOAA forecast gridpoints are ~2.5km cells, so sampling finer than this just
# produces redundant /points lookups that all resolve to the same gridpoint.
# Matching the cell size covers every overlapped gridpoint with the fewest
# calls.
SAMPLE_STEP_KM = 2.5

# Fixed global sampling lattice step, in degrees, derived from SAMPLE_STEP_KM
# using the latitude-independent km->deg factor. Using a constant degree step
# everywhere keeps the sample grid globally aligned so overlapping areas (at
# any center/size/zoom) reuse each other's cached /points lookups.
LATTICE_STEP_DEG = SAMPLE_STEP_KM / 110.574

# Distance (km) the area center is snapped to when forming the gridpoint-set
# cache key. Re-centering within this distance of a previously-resolved area
# reuses its cached gridpoints instead of re-resolving a near-identical grid.
# Kept well under the data resolution so it has no visible effect on output.
CACHE_SNAP_KM = 1.0

# Cache of resolved gridpoint lists keyed by a rounded area signature.
# Resolving an area's gridpoints means mapping every sample coordinate to a
# NOAA gridpoint; for a fixed map view this result is stable, so caching it
# avoids repeating that work on every poll.
_GRIDPOINTS_CACHE: dict[tuple, list] = {}
_GRIDPOINTS_LOCK = threading.Lock()


# --- Cached-polygon spatial index -------------------------------------------
# Every /gridpoints doc we've already fetched carries the cell's polygon. Those
# polygons tile space, so a map that moves slightly reveals sample coordinates
# that often fall inside a gridpoint we ALREADY have cached. This index lets us
# resolve such a sample locally (point-in-polygon) instead of issuing a /points
# call. It only REUSES already-fetched docs — it never fetches extra data — so
# it strictly reduces work.
#
# Structure: coarse lat/lon bucket -> list of (NOAAGridpoint, polygon_ring).
# A sample is tested against its bucket (and the 8 neighbors, since a polygon
# may straddle a bucket edge). Buckets are populated lazily from the on-disk
# /gridpoints cache and augmented as new docs are read.
_POLY_INDEX_BUCKET_DEG = 0.25  # ~25-28 km cells; a handful of gridpoints each
_POLY_INDEX: dict[tuple, list] = {}
_POLY_INDEX_LOCK = threading.Lock()
# Gridpoints already registered in the index this process, so repeat resolves
# don't re-read their (large) /gridpoints docs from disk just to re-index.
_POLY_INDEXED_GPS: set = set()


def _poly_bucket(lon, lat):
    return (math.floor(lon / _POLY_INDEX_BUCKET_DEG),
            math.floor(lat / _POLY_INDEX_BUCKET_DEG))


def _ring_centroid(ring):
    pts = ring[:-1] if len(ring) > 1 and ring[0] == ring[-1] else ring
    n = len(pts)
    return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n)


def _point_in_ring(lon, lat, ring):
    """Ray-casting point-in-polygon for a ring of [lon,lat] corners."""
    pts = ring[:-1] if len(ring) > 1 and ring[0] == ring[-1] else ring
    inside = False
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if (y1 > lat) != (y2 > lat):
            xint = x1 + (lat - y1) * (x2 - x1) / (y2 - y1)
            if lon < xint:
                inside = not inside
    return inside


def register_gridpoint_polygon(gp, ring):
    """Add a resolved (gridpoint, polygon) to the index, keyed by the polygon's
    centroid bucket. No-op if already indexed."""
    try:
        clon, clat = _ring_centroid(ring)
    except Exception:
        return
    b = _poly_bucket(clon, clat)
    with _POLY_INDEX_LOCK:
        if gp in _POLY_INDEXED_GPS:
            return
        _POLY_INDEX.setdefault(b, []).append((gp, ring))
        _POLY_INDEXED_GPS.add(gp)


def _ring_from_cached_gridpoint(gp):
    """Read a cached /gridpoints doc's polygon ring from disk, or None if the
    doc isn't cached / is unreadable. Reads the whole doc but only extracts the
    geometry — cheap relative to a network fetch, and only done once per gp."""
    try:
        path = PathObj(f"/gridpoints/{gp}")
        if not os.path.isfile(path.tempfile):
            return None
        with open(path.tempfile) as f:
            doc = json.loads(f.read())
        return doc["json"]["geometry"]["coordinates"][0]
    except Exception:
        return None


def index_cached_gridpoints(gridpoints):
    """Register each gridpoint's polygon (from its cached doc) into the spatial
    index, so later nearby views can resolve samples locally. Best-effort:
    gridpoints whose doc isn't cached are skipped. Gridpoints already indexed
    this process are skipped WITHOUT a disk read (the expensive part), so a
    repeat resolve doesn't re-parse thousands of large /gridpoints docs."""
    for gp in gridpoints:
        if gp in _POLY_INDEXED_GPS:
            continue  # already indexed; don't re-read the doc
        ring = _ring_from_cached_gridpoint(gp)
        if ring is not None:
            register_gridpoint_polygon(gp, ring)


# --- Recursive WFO decomposition --------------------------------------------
# To resolve an area's gridpoints while minimizing /points calls, we exploit
# that each WFO's forecast region is a CONTIGUOUS polygon. For a convex sample
# rectangle: if every coordinate on its BOUNDARY resolves to the same WFO, no
# other office can reach the interior (it would have to cross the boundary), so
# the whole interior is that one office — resolved WITHOUT probing it.
#
# Algorithm (binary bisection on the longer axis):
#   1. Probe the rectangle's boundary coordinates via /points (concurrent).
#   2. If the boundary is a single WFO -> the rectangle is that office; assign
#      every interior coordinate to it (exact cell via the polygon test, which
#      uses the /gridpoints docs we need anyway; /points fallback for the rare
#      point no polygon places).
#   3. Else a WFO border crosses the rectangle: split it in half across its
#      longer axis and recurse on each half (concurrently).
#   4. Recursion floor: once a rectangle is small (<= RECT_FLOOR samples per
#      side) just /points every coordinate in it directly — cheap at that size
#      and robust to jagged borders.
#
# Coordinates are addressed by INTEGER LATTICE INDEX (i,j) on the same global
# lattice as CoordinateArea.map_coordinates (coord = (i*step, j*step)), so
# rectangles, boundaries, bisection and shared-edge dedup are exact integer
# arithmetic and reuse cached /points across overlapping views.

# Below this many samples per side, stop recursing and /points the whole block.
RECT_FLOOR = 4

# If the decomposition probes more than this fraction of the area's total
# cells, it has lost its advantage over a plain per-cell scan (pathological on
# fractal borders like a convoluted coastline, where nearly every sub-rectangle
# straddles the border and keeps splitting). Abort and let the caller fall back
# to the scan, which probes exactly total_cells — decomposition must never do
# worse. Set near 1.0 so we only bail when genuinely losing; a small area that
# is itself a single floor block legitimately probes ~all its cells and is
# exempted separately (see the cap computation in resolve_area_gridpoints).
DECOMP_BUDGET_FRACTION = 0.9

# Recursion levels that run their two halves on a thread pool (overlapping
# their NOAA /points probes, the dominant cost). Capped independently of the
# depth limit to bound concurrent threads: up to 2^DECOMP_PARALLEL_DEPTH nodes
# run at once; deeper levels recurse serially (cheap now — the per-node fetch
# goes through the writer's own global concurrency, and the memo is shared, so
# a serial deep node adds little beyond its own network wait).
DECOMP_PARALLEL_DEPTH = int(os.environ.get("WR_DECOMP_PARALLEL_DEPTH", "5"))

# Hard cap on split depth. A rectangle still multi-WFO after this many splits
# is resolved by ONE concurrent bulk /points probe of all its cells, instead of
# fragmenting further. Each recursion level peels off the single-WFO regions it
# finds (big /points savings) and only the still-mixed remainder recurses, so a
# deeper cap captures MORE single-WFO fills (fewer calls) at the cost of more
# boundary probes on the fragmented parts. Tune via WR_DECOMP_MAX_DEPTH; the
# wall-clock deadline is the backstop against true pathology (fractal borders).
DECOMP_MAX_DEPTH = int(os.environ.get("WR_DECOMP_MAX_DEPTH", "8"))


class _DecompBudgetExceeded(Exception):
    """Raised when the decomposition's probe count exceeds its budget, so the
    caller falls back to the plain scan."""


class _Budget:
    """Shared probe-count budget for one decomposition. `spend(n)` raises
    _DecompBudgetExceeded once total probes exceed `cap`, so the caller falls
    back to the plain concurrent bulk scan (which probes exactly total_cells —
    the decomposition must never make MORE calls than that)."""
    __slots__ = ("used", "cap", "lock")

    def __init__(self, cap):
        self.used = 0
        self.cap = cap
        self.lock = threading.Lock()

    def spend(self, n):
        with self.lock:
            self.used += n
            if self.used > self.cap:
                raise _DecompBudgetExceeded(f"probes {self.used} > {self.cap}")


def _coord_for_index(i, j, step=None):
    """MapCoordinate at global lattice index (i, j)."""
    if step is None:
        step = LATTICE_STEP_DEG
    return MapCoordinate(lat=i * step, lon=j * step)


def _resolve_points(coords):
    """Resolve many coordinates to (wfo, gridX, gridY) concurrently.

    Returns {(i,j)-free: coord -> NOAAGridpoint or None}. Uses the on-disk
    /points cache first, batches the misses into one cache-writer job, then
    reads them back concurrently. None means out-of-coverage (NOAA 404).
    Never resolves the same coordinate's /points twice (dedup by value).
    """
    coords = list({(c.lat, c.lon): c for c in coords}.values())
    out = {}
    misses = []
    for c in coords:
        gp = gridpoint_from_cached_points(c)
        if gp is not None:
            out[(c.lat, c.lon)] = gp
        else:
            misses.append(c)

    if misses:
        # One batched writer job fetches all misses CONCURRENTLY (the writer
        # process coalesces and parallelizes; concurrency is global, not
        # per-node). After it returns the files are on disk, so we just read
        # them back — no per-node thread pool (which would multiply across
        # recursion nodes and overwhelm NOAA).
        try:
            from .cachewriter import ensure as _ensure_cached
            _ensure_cached([f"/points/{c}" for c in misses])
        except Exception:
            pass
        for c in misses:
            # gridpoint_from_cached_points reads the now-cached file; fall back
            # to a direct resolve only if the writer produced nothing.
            gp = gridpoint_from_cached_points(c)
            if gp is None:
                gp = gridpoint_from_map_coordinate(c)  # inline fetch/classify
            out[(c.lat, c.lon)] = gp
    return out


def _rect_boundary_indices(i0, i1, j0, j1):
    """Yield the integer (i,j) indices on the perimeter of the inclusive index
    rectangle [i0..i1] x [j0..j1], each once."""
    if i0 == i1 or j0 == j1:  # degenerate: whole thing is its own boundary
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                yield (i, j)
        return
    for j in range(j0, j1 + 1):      # top & bottom rows
        yield (i0, j)
        yield (i1, j)
    for i in range(i0 + 1, i1):      # left & right cols (corners already done)
        yield (i, j0)
        yield (i, j1)


def _rect_all_indices(i0, i1, j0, j1):
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            yield (i, j)


def _resolve_indices(indices, memo, step, budget=None):
    """Resolve integer lattice indices to NOAAGridpoints, using `memo` as a
    per-call cache so each coordinate's /points is resolved AT MOST ONCE across
    the whole recursion (shared rectangle edges are free).

    Mutates and reads `memo` ({(i,j): NOAAGridpoint|None}). Charges newly-probed
    coordinates against `budget` (which aborts a pathological decomposition).
    """
    todo = [(i, j) for (i, j) in indices if (i, j) not in memo]
    if not todo:
        return
    if budget is not None:
        budget.spend(len(todo))
    coords = {(i, j): _coord_for_index(i, j, step) for (i, j) in todo}
    resolved = _resolve_points(list(coords.values()))
    for (i, j), c in coords.items():
        memo[(i, j)] = resolved.get((c.lat, c.lon))


def resolve_area_gridpoints(i0, i1, j0, j1, step=None, memo=None, budget=None,
                            _depth=0):
    """Resolve the unique set of NOAA gridpoints covering the inclusive index
    rectangle [i0..i1] x [j0..j1], via recursive single-WFO decomposition.
    Returns a set of NOAAGridpoints (out-of-coverage samples contribute
    nothing).

    `memo` caches resolved COORDINATES ({(i,j): gp|None}) across the recursion
    so shared rectangle boundaries are never re-probed. `budget` caps total
    probes; if exceeded (fractal border), _DecompBudgetExceeded propagates so
    the caller falls back to a plain scan. Both are created at the top level.
    """
    if step is None:
        step = LATTICE_STEP_DEG
    if memo is None:
        memo = {}
    if budget is None:
        total_cells = (i1 - i0 + 1) * (j1 - j0 + 1)
        if (i1 - i0 + 1) <= RECT_FLOOR or (j1 - j0 + 1) <= RECT_FLOOR:
            cap = total_cells + 1  # pure floor block: probing all is correct
        else:
            cap = int(total_cells * DECOMP_BUDGET_FRACTION)
        budget = _Budget(cap)

    n_i = i1 - i0 + 1
    n_j = j1 - j0 + 1

    indent = "  " * _depth
    if _RESOLVE_TIMING:
        logger.warning("%sdecomp rect i[%d..%d] j[%d..%d] (%dx%d=%d cells)",
                       indent, i0, i1, j0, j1, n_i, n_j, n_i * n_j)

    # Recursion floor: small enough that probing everything is cheap. Also
    # applies at the hard depth cap, where we stop fragmenting and bulk-probe
    # the whole sub-rectangle in ONE concurrent batch (bounds the latency
    # pathology on deeply-fragmenting complex areas).
    if n_i <= RECT_FLOOR or n_j <= RECT_FLOOR or _depth >= DECOMP_MAX_DEPTH:
        idx = list(_rect_all_indices(i0, i1, j0, j1))
        _resolve_indices(idx, memo, step, budget)
        if _RESOLVE_TIMING:
            reason = "depth-cap" if _depth >= DECOMP_MAX_DEPTH else "floor"
            logger.warning("%s  %s: probed %d cells directly", indent, reason, len(idx))
        return {memo[(i, j)] for (i, j) in idx if memo.get((i, j)) is not None}

    # 1. Probe the boundary (only coordinates not already resolved).
    bidx = list(_rect_boundary_indices(i0, i1, j0, j1))
    _resolve_indices(bidx, memo, step, budget)
    boundary = {(i, j): memo.get((i, j)) for (i, j) in bidx}

    wfos = {gp.id for gp in boundary.values() if gp is not None}
    has_gap = any(gp is None for gp in boundary.values())

    if _RESOLVE_TIMING:
        logger.warning("%s  boundary %d pts -> WFOs=%s gap=%s",
                       indent, len(bidx), sorted(wfos), has_gap)

    # 2. Single WFO on the whole boundary (and no coverage gap) -> interior is
    #    that office; enumerate its gridpoint set from the boundary indices.
    if len(wfos) == 1 and not has_gap:
        (wfo,) = wfos
        gps = _fill_single_wfo(wfo, boundary)
        if _RESOLVE_TIMING:
            logger.warning("%s  SINGLE-WFO %s: %d gridpoints (no interior probe)",
                           indent, wfo, len(gps))
        return gps

    # 3. Border crosses the rectangle (or a coverage gap touches it): bisect
    #    the longer axis and recurse concurrently.
    if n_i >= n_j:
        mid = (i0 + i1) // 2
        halves = ((i0, mid, j0, j1), (mid, i1, j0, j1))  # share row `mid`
        split_line = [(mid, j) for j in range(j0, j1 + 1)]
        axis = "i"
    else:
        mid = (j0 + j1) // 2
        halves = ((i0, i1, j0, mid), (i0, i1, mid, j1))  # share col `mid`
        split_line = [(i, mid) for i in range(i0, i1 + 1)]
        axis = "j"

    if _RESOLVE_TIMING:
        logger.warning("%s  SPLIT on %s at %d (%d WFOs on boundary)",
                       indent, axis, mid, len(wfos))

    # Pre-resolve the shared split line once, so both halves start with it
    # cached and never re-probe it. (The outer boundary is already in `memo`.)
    _resolve_indices(split_line, memo, step, budget)

    out = set()
    # Parallelize only the shallow levels: deeper splits create many small
    # rectangles, and spawning a thread pool per node (hundreds of them) costs
    # more than it saves — the /points fetches inside each node are ALREADY
    # internally concurrent (see _resolve_points), which is where the real
    # parallelism matters. Recurse serially past DECOMP_PARALLEL_DEPTH.
    if _depth < DECOMP_PARALLEL_DEPTH:
        # Run the two halves on threads. They probe DISJOINT interior cells and
        # share the split line/boundary already in `memo`, so concurrent writes
        # touch different keys; a lock guards the dict against resize races.
        # (Avoids copying the growing memo per node, which was O(nodes x memo)
        # and dominated wall-clock on complex areas.)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            futures = [ex.submit(resolve_area_gridpoints, a, b, c, d, step,
                                 memo, budget, _depth + 1)
                       for (a, b, c, d) in halves]
            for f in futures:
                out |= f.result()
    else:
        for (a, b, c, d) in halves:
            out |= resolve_area_gridpoints(a, b, c, d, step, memo, budget,
                                           _depth + 1)
    return out


def _fill_single_wfo(wfo, boundary):
    """The whole rectangle is `wfo`. Return the set of NOAAGridpoints covering
    it WITHOUT probing or reading any interior doc.

    Key idea: the rectangle's boundary cells have known, real (gridX,gridY)
    (from the /points we just did). Because each WFO's grid is a regular
    integer lattice, the interior gridpoint indices are BRACKETED by the
    boundary's index range: every interior cell's gridX lies within [min..max]
    of the boundary gridX (same for gridY). So the gridpoint SET the rectangle
    covers is the index box spanned by the boundary, padded by 1 to absorb the
    projection's slight shear at the corners.

    Exact for the SET (the caller's need): it cannot MISS a gridpoint (the
    boundary brackets the interior), and any spurious extra at a sheared corner
    is dropped by the point-in-polygon test at render time. Reads NOTHING per
    interior cell — the cost that otherwise dominated."""
    bgps = [gp for gp in boundary.values() if gp is not None]
    if not bgps:
        return set()
    # The boundary cells are the rectangle's extremes, so their gridX/gridY
    # min/max already bracket every interior cell's indices on a regular
    # lattice — NO padding needed. Padding would add a full spurious ring of
    # gridpoints, each an extra /gridpoints fetch (the costly network op).
    xmin = min(gp.x for gp in bgps)
    xmax = max(gp.x for gp in bgps)
    ymin = min(gp.y for gp in bgps)
    ymax = max(gp.y for gp in bgps)
    return {NOAAGridpoint(wfo, gx, gy)
            for gx in range(xmin, xmax + 1)
            for gy in range(ymin, ymax + 1)}


def lookup_gridpoint_in_index(lon, lat):
    """Return a cached NOAAGridpoint whose polygon contains (lon,lat), or None.
    Checks the point's bucket and its 8 neighbors (a polygon may straddle a
    bucket boundary)."""
    bx, by = _poly_bucket(lon, lat)
    with _POLY_INDEX_LOCK:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for gp, ring in _POLY_INDEX.get((bx + dx, by + dy), ()):
                    if _point_in_ring(lon, lat, ring):
                        return gp
    return None


class Coordinate:
    def __init__(self, x, y):
        self.x = round(x, 7)
        self.y = round(y, 7)

    def __str__(self):
        return f'{self.y},{self.x}'

    def __repr__(self):
        return f'"{str(self)}"'


class MapCoordinate(Coordinate):
    def __init__(self, lat, lon):
        super().__init__(x=lon, y=lat)

    @property
    def lat(self):
        return self.y

    @property
    def lon(self):
        return self.x


class NOAAGridpoint:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y

    def __str__(self) -> str:
        return f'{self.id}/{self.x},{self.y}'

    def __repr__(self) -> str:
        return f'"{str(self)}"'

    def __eq__(self, other) -> bool:
        if not isinstance(other, NOAAGridpoint):
            return NotImplemented
        return (self.id, self.x, self.y) == (other.id, other.x, other.y)

    def __hash__(self) -> int:
        return hash((self.id, self.x, self.y))


class CoordinateArea:

    def __init__(self, center: MapCoordinate, width: float, height: float):
        self.center = center
        self.width = max(width, SAMPLE_STEP_KM)
        self.height = max(height, SAMPLE_STEP_KM)

    @cached_property
    def center_gridpoint(self):
        return gridpoint_from_map_coordinate(self.center)

    @cached_property
    def center_id(self):
        return self.center_gridpoint.id

    @cached_property
    def time_zone(self):
        """IANA timezone name (e.g. "America/Los_Angeles") for the area's
        center, from NOAA's /points metadata. None if unavailable."""
        try:
            resp = NOAAConnection().get(f"/points/{self.center}")
            return resp["properties"].get("timeZone")
        except Exception:
            return None

    @property
    def index_bounds(self):
        """Inclusive integer lattice-index rectangle (i0, i1, j0, j1) covering
        this area, on the global lattice where coord = (i*step, j*step). i is
        the latitude index, j the longitude index. Shared by map_coordinates
        and the recursive WFO decomposition so they address the SAME cells."""
        dx = dkm_to_dlat(self.width)
        dy = dkm_to_dlon(self.height, self.center.lat)
        x = self.center.lon - dx / 2
        y = self.center.lat - dy / 2
        step = LATTICE_STEP_DEG
        i0 = math.floor(y / step)
        i1 = math.ceil((y + dy) / step)
        j0 = math.floor(x / step)
        j1 = math.ceil((x + dx) / step)
        return i0, i1, j0, j1

    @property
    def map_coordinates(self):
        # Generate coordinates as (integer lattice index * step) rather than by
        # repeatedly adding `step`. Integer indexing avoids floating-point
        # accumulation drift, so a given lattice cell has the EXACT same
        # coordinate value for every area that includes it — regardless of
        # center, size, or zoom. That exact-equality is what lets overlapping
        # areas reuse each other's cached /points lookups.
        i0, i1, j0, j1 = self.index_bounds
        step = LATTICE_STEP_DEG
        return (
            MapCoordinate(lat=i * step, lon=j * step)
            for i in range(i0, i1 + 1)
            for j in range(j0, j1 + 1)
        )

    @cached_property
    def signature(self):
        # Exact cache key for this area's resolved gridpoint set. (Used only
        # as a cache key, never for display or computation.) Near-identical
        # areas are matched separately via _find_cached_gridpoints, which
        # reuses an existing entry whose center is within CACHE_SNAP_KM.
        return (
            round(self.center.lat, 6),
            round(self.center.lon, 6),
            round(self.width, 1),
            round(self.height, 1),
        )

    def _find_cached_gridpoints(self):
        """Return a cached gridpoint set to reuse, or None.

        Reuses an entry when the area dimensions match and its center is
        within CACHE_SNAP_KM of this center — so re-centering slightly off a
        previously-resolved area doesn't re-resolve a near-identical grid.
        The snap distance is well below the data resolution, so reuse has no
        visible effect on the field.
        """
        w = round(self.width, 1)
        h = round(self.height, 1)
        snap_deg_lat = CACHE_SNAP_KM / 110.574
        snap_deg_lon = CACHE_SNAP_KM / (111.320 * math.cos(math.radians(self.center.lat)))
        with _GRIDPOINTS_LOCK:
            # Exact hit first.
            exact = _GRIDPOINTS_CACHE.get(self.signature)
            if exact is not None:
                return exact
            # Otherwise, nearest matching area within the snap distance.
            best = None
            best_d2 = None
            for (clat, clon, cw, ch), points in _GRIDPOINTS_CACHE.items():
                if cw != w or ch != h:
                    continue
                dlat = (self.center.lat - clat) / snap_deg_lat
                dlon = (self.center.lon - clon) / snap_deg_lon
                d2 = dlat * dlat + dlon * dlon
                if d2 <= 1.0 and (best_d2 is None or d2 < best_d2):
                    best, best_d2 = points, d2
            return best

    @property
    def coordinates_by_proximity(self):
        """Sample coordinates ordered nearest-to-center first, so resolution
        (and thus rendering) fills outward from the middle of the view."""
        center = self.center

        def dist2(c):
            # Squared planar distance in degrees; fine for ordering a small
            # local area.
            return (c.lat - center.lat) ** 2 + (c.lon - center.lon) ** 2

        return sorted(self.map_coordinates, key=dist2)

    def _resolve_by_decomposition(self):
        """Resolve the area's unique gridpoints via recursive single-WFO
        decomposition over the lattice index rectangle. Returns the unique
        gridpoint list, or None if the decomposition exceeds its probe budget
        (pathological fractal border) so the caller falls back to the
        per-sample scan."""
        i0, i1, j0, j1 = self.index_bounds
        total_cells = (i1 - i0 + 1) * (j1 - j0 + 1)
        budget = _Budget(
            total_cells + 1
            if ((i1 - i0 + 1) <= RECT_FLOOR or (j1 - j0 + 1) <= RECT_FLOOR)
            else int(total_cells * DECOMP_BUDGET_FRACTION)
        )
        try:
            gps = resolve_area_gridpoints(i0, i1, j0, j1, budget=budget)
        except _DecompBudgetExceeded as e:
            logger.warning(
                "WFO decomposition abandoned (%s; %d/%d probes); bulk scan",
                e, budget.used, total_cells)
            return None
        if _RESOLVE_TIMING:
            saved = 100.0 * (1 - budget.used / total_cells) if total_cells else 0
            logger.warning(
                "WFO decomposition: %d unique gridpoints | %d /points probed "
                "of %d cells (%.0f%% fewer than a full scan)",
                len(gps), budget.used, total_cells, saved)
        return list(gps)

    def iter_gridpoints(self):
        """Yield unique NOAA gridpoints for this area, nearest-to-center
        first, as each is resolved. Lets callers stream results instead of
        waiting for the whole area to resolve.

        If a matching (exact or nearby) gridpoint set is already cached, yields
        it from the cache. A fast recursive single-WFO decomposition
        (WR_WFO_DECOMP, default on) resolves most areas by probing only
        rectangle boundaries; failing that, a per-sample /points scan runs.
        The full resolved set is cached once produced.
        """
        cached = self._find_cached_gridpoints()
        if cached is not None:
            yield from cached
            return

        # Fast path: recursive single-WFO decomposition. Probes only rectangle
        # boundaries (plus splits where a WFO border crosses), computing the
        # interior from the boundary — a large reduction in /points calls vs.
        # probing every lattice sample, measured 3-6x on real multi-office
        # regions. Disable with WR_WFO_DECOMP=0 to use the per-sample scan.
        if os.environ.get("WR_WFO_DECOMP", "1") != "0":
            try:
                ordered = self._resolve_by_decomposition()
            except Exception:
                logger.warning("WFO decomposition failed; falling back to scan",
                               exc_info=True)
                ordered = None
            if ordered is not None:
                with _GRIDPOINTS_LOCK:
                    _GRIDPOINTS_CACHE[self.signature] = ordered
                if os.environ.get("WR_POLYGON_INDEX", "1") != "0":
                    try:
                        index_cached_gridpoints(ordered)
                    except Exception:
                        pass
                yield from ordered
                return

        coordinates = self.coordinates_by_proximity

        seen = set()
        ordered = []

        def _emit(gp):
            if gp not in seen:
                seen.add(gp)
                ordered.append(gp)
                return True
            return False

        # Pass 1: resolve what we can WITHOUT the network, in priority order:
        #   (a) the on-disk /points cache — the AUTHORITATIVE, exact mapping for
        #       this coordinate (NOAA's own answer), checked first;
        #   (b) the cached-polygon index — an INFERENCE (point-in-polygon on an
        #       already-fetched gridpoint), used only when (a) has no file. This
        #       covers newly-revealed samples that sit inside a polygon from a
        #       previous view, which is what makes a small pan/zoom cheap.
        # Anything resolved by neither becomes a /points network miss (pass 2).
        # Disable the polygon inference with WR_POLYGON_INDEX=0.
        use_index = os.environ.get("WR_POLYGON_INDEX", "1") != "0"
        misses = []
        n_fs_hits = 0
        n_index_hits = 0
        for c in coordinates:
            gp = gridpoint_from_cached_points(c)   # (a) authoritative FS hit
            if gp is not None:
                n_fs_hits += 1
                if _emit(gp):
                    yield gp
                continue
            gp = lookup_gridpoint_in_index(c.lon, c.lat) if use_index else None
            if gp is not None:                      # (b) inferred from polygon
                n_index_hits += 1
                if _emit(gp):
                    yield gp
                continue
            misses.append(c)

        # Pass 2: resolve the misses via /points. Pre-warm ALL in a single
        # batched writer job, then read concurrently.
        if misses:
            try:
                from .cachewriter import ensure as _ensure_cached
                _ensure_cached([f"/points/{c}" for c in misses])
            except Exception:
                # Writer unavailable: fall back to resolving inline (each call
                # fetches on miss via connection's inline fallback).
                pass

            with concurrent.futures.ThreadPoolExecutor(max_workers=NOAA_MAX_CONCURRENCY) as executor:
                for gp in executor.map(gridpoint_from_map_coordinate, misses):
                    if gp is None:
                        continue  # out-of-coverage sample (e.g. across border)
                    if _emit(gp):
                        yield gp

        if _RESOLVE_TIMING:
            logger.warning(
                "  resolve breakdown: %d/%d from FS /points cache, %d from "
                "polygon index, %d needed /points network -> %d unique gridpoints",
                n_fs_hits, len(coordinates), n_index_hits, len(misses), len(ordered),
            )

        with _GRIDPOINTS_LOCK:
            _GRIDPOINTS_CACHE[self.signature] = ordered

        # Populate the index with this area's gridpoint polygons so subsequent
        # nearby views can reuse them. Best-effort; reads cached docs only.
        if use_index:
            try:
                index_cached_gridpoints(ordered)
            except Exception:
                pass

    @cached_property
    def gridpoints(self):
        cached = self._find_cached_gridpoints()
        if cached is not None:
            return cached

        # Fully consume the streaming resolver (which also populates the
        # cache) and return the complete, proximity-ordered list.
        return list(self.iter_gridpoints())

    def cache(self):
        # Eagerly consume the streaming cacher (ignoring progress). Kept for
        # callers that just want the side effect of populating the cache.
        for _ in self.iter_cache():
            pass

    def iter_cache(self):
        """Populate the on-disk cache for this area, yielding progress for two
        ordered phases, each with its OWN fraction:
          1. "points"     — resolve gridpoints via /points lookups;
          2. "gridpoints" — fetch /gridpoints forecast data.
        Each yielded dict is {"phase", "done", "total"}. Only not-fresh files
        are counted/fetched; all fetching is delegated to the cache-writer.
        """
        from .cachewriter import ensure_stream, not_fresh

        if _RESOLVE_TIMING:
            logger.warning(
                "iter_cache ENTER: size=%.0fx%.0fkm NOAA_MAX_CONCURRENCY=%r",
                self.width, self.height, os.environ.get("NOAA_MAX_CONCURRENCY"),
            )
        t0 = time.perf_counter()

        # Throttled progress logger (every ~2s) so a long resolve streams live
        # progress + rate to the log instead of going silent until the end.
        def _progress(phase, done, total, phase_t0, state):
            now = time.perf_counter()
            if done < total and (now - state["last"]) < 2.0:
                return
            state["last"] = now
            elapsed = now - phase_t0
            rate = done / elapsed if elapsed > 0 else 0.0
            remaining = (total - done) / rate if rate > 0 else float("inf")
            logger.warning(
                "  %s: %d/%d (%.0f%%) %.1f/s elapsed=%.1fs eta=%.0fs",
                phase, done, total, (100.0 * done / total if total else 100.0),
                rate, elapsed, remaining,
            )

        # Phase 1: resolve the unique gridpoint set. The decomposition (inside
        # self.gridpoints) fetches only the /points it actually probes
        # (boundaries + border splits), NOT the full lattice — so we must NOT
        # bulk-fetch every /points/{c} here. Progress for this phase is coarse
        # (resolution is now fast relative to the /gridpoints fetch); we report
        # a single step.
        yield {"phase": "points", "done": 0, "total": 1}
        gridpoints = self.gridpoints  # runs decomposition; fetches probed /points
        yield {"phase": "points", "done": 1, "total": 1}
        t_points = time.perf_counter()
        t_resolve = t_points  # resolution + /points fetch folded into one phase

        if _RESOLVE_TIMING:
            logger.warning("  resolved %d unique gridpoints in %.1fs",
                           len(gridpoints), t_resolve - t0)

        # Phase 2: /gridpoints (gridpoint set now known from phase 1).
        all_grid = [f"/gridpoints/{gp}" for gp in gridpoints]
        grid_paths = not_fresh(all_grid)
        n_grid_total = len(all_grid)
        n_grid_fetched = len(grid_paths)
        if _RESOLVE_TIMING:
            logger.warning(
                "  gridpoints phase: %d unique (%d need fetch) after %.1fs",
                n_grid_total, n_grid_fetched, t_resolve - t0,
            )
        yield {"phase": "gridpoints", "done": 0, "total": len(grid_paths)}
        _gstate = {"last": 0.0}
        for prog in ensure_stream(grid_paths):
            if _RESOLVE_TIMING:
                _progress("gridpoints", prog["done"], len(grid_paths), t_resolve, _gstate)
            yield {"phase": "gridpoints", "done": prog["done"], "total": len(grid_paths)}
        t_grid = time.perf_counter()

        if _RESOLVE_TIMING:
            ms = lambda a, b: round((b - a), 2)
            try:
                conc = int(os.environ.get("NOAA_MAX_CONCURRENCY", "8"))
            except ValueError:
                conc = 8
            logger.warning(
                "area resolve timing: center=(%.4f,%.4f) size=%.0fx%.0fkm conc=%d "
                "| resolve+points: %d gridpoints in %ss "
                "| gridpoints: %d (%d fetched) in %ss "
                "| TOTAL %ss",
                self.center.lat, self.center.lon, self.width, self.height, conc,
                len(gridpoints), ms(t0, t_points),
                n_grid_total, n_grid_fetched, ms(t_resolve, t_grid),
                ms(t0, t_grid),
            )


def gridpoint_from_cached_points(coordinate: MapCoordinate):
    """Resolve a sample to its NOAA gridpoint using ONLY the on-disk /points
    cache — no network, no writer. Returns the NOAAGridpoint if the /points
    file for this exact coordinate is already cached, else None. This is the
    authoritative, exact answer (NOAA's own point->grid mapping), so it is
    checked before the inferred polygon-index lookup."""
    try:
        path = PathObj(f"/points/{coordinate}")
        if not os.path.isfile(path.tempfile):
            return None
        with open(path.tempfile) as f:
            doc = json.loads(f.read())
        props = doc["json"]["properties"]
        return NOAAGridpoint(props["gridId"], props["gridX"], props["gridY"])
    except Exception:
        return None


def gridpoint_from_map_coordinate(coordinate: MapCoordinate):
    """Resolve a sample coordinate to its NOAA gridpoint, or None if NOAA has
    no coverage for that point (e.g. it falls outside the US — in Canada,
    Mexico, or open ocean). Returning None instead of raising lets an area that
    straddles the coverage boundary resolve its in-coverage gridpoints and skip
    the rest, rather than aborting the whole resolution."""
    conn = NOAAConnection()
    try:
        resp = conn.get(f"/points/{coordinate}")
    except NoCoverageError:
        return None
    resp = resp["properties"]
    return NOAAGridpoint(resp["gridId"], resp["gridX"], resp["gridY"])


def dkm_to_dlat(km):
    return km / 110.574


def dkm_to_dlon(km, lat):
    return km / (111.320 * math.cos(math.radians(lat)))


def arange(start, stop, step):
    r = []
    while start < stop:
        r.append(start)
        start += step
    return r


