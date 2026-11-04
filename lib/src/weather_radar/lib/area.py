import os
import concurrent.futures
import multiprocessing
import threading
import math
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
    def map_coordinates(self):

        dx = dkm_to_dlat(self.width)
        dy = dkm_to_dlon(self.height, self.center.lat)

        x = self.center.lon - dx / 2
        y = self.center.lat - dy / 2

        step = LATTICE_STEP_DEG

        # Generate coordinates as (integer lattice index * step) rather than by
        # repeatedly adding `step`. Integer indexing avoids floating-point
        # accumulation drift, so a given lattice cell has the EXACT same
        # coordinate value for every area that includes it — regardless of
        # center, size, or zoom. That exact-equality is what lets overlapping
        # areas reuse each other's cached /points lookups.
        i0 = math.floor(y / step)
        i1 = math.ceil((y + dy) / step)
        j0 = math.floor(x / step)
        j1 = math.ceil((x + dx) / step)

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

    def iter_gridpoints(self):
        """Yield unique NOAA gridpoints for this area, nearest-to-center
        first, as each is resolved. Lets callers stream results instead of
        waiting for the whole area to resolve.

        If a matching (exact or nearby) gridpoint set is already cached, yields
        it from the cache. Otherwise resolves lazily (one /points lookup per
        lattice sample, deduped to unique gridpoints) and populates the cache
        once the full set has been produced.
        """
        cached = self._find_cached_gridpoints()
        if cached is not None:
            yield from cached
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

        # Phase 1: /points. Its total is the not-fresh point count.
        all_points = [f"/points/{c}" for c in self.map_coordinates]
        point_paths = not_fresh(all_points)
        n_points_total = len(all_points)
        n_points_fetched = len(point_paths)
        yield {"phase": "points", "done": 0, "total": len(point_paths)}
        _pstate = {"last": 0.0}
        for prog in ensure_stream(point_paths):
            if _RESOLVE_TIMING:
                _progress("points", prog["done"], len(point_paths), t0, _pstate)
            yield {"phase": "points", "done": prog["done"], "total": len(point_paths)}
        t_points = time.perf_counter()

        # Resolve the unique gridpoint set (reads the warmed /points files and
        # dedupes). Timed separately from the fetch below.
        if _RESOLVE_TIMING:
            logger.warning("  resolving unique gridpoints from %d points...", n_points_total)
        gridpoints = self.gridpoints
        t_resolve = time.perf_counter()

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
                "| points: %d samples (%d fetched) in %ss "
                "| resolve_gridpoints: %d unique in %ss "
                "| gridpoints: %d (%d fetched) in %ss "
                "| TOTAL %ss",
                self.center.lat, self.center.lon, self.width, self.height, conc,
                n_points_total, n_points_fetched, ms(t0, t_points),
                len(gridpoints), ms(t_points, t_resolve),
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


