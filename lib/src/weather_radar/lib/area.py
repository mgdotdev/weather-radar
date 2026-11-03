import os
import concurrent.futures
import multiprocessing
import threading
import math
from functools import cached_property

from .connection import NOAAConnection, PathObj, get_and_write, NoCoverageError

lock = multiprocessing.Lock()

# Maximum simultaneous requests to the NOAA API. NOAA enforces undisclosed
# rate limits and returns HTTP 429 when overwhelmed; we respect 429/Retry-After
# (see connection.retry_get) so this ceiling can be fairly generous — if we
# push too hard NOAA throttles and we automatically back off. Env-overridable.
NOAA_MAX_CONCURRENCY = int(os.environ.get("NOAA_MAX_CONCURRENCY", "8"))

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
        it from the cache. Otherwise resolves lazily and populates the cache
        once the full set has been produced.
        """
        cached = self._find_cached_gridpoints()
        if cached is not None:
            yield from cached
            return

        coordinates = self.coordinates_by_proximity

        # Pre-warm ALL /points lookups in a single batched writer job, rather
        # than letting each gridpoint_from_map_coordinate() trigger its own
        # per-path writer round-trip. The writer fetches them concurrently;
        # afterwards every /points read below is a fast local cache hit. This
        # turns N blocking IPC round-trips into one.
        try:
            from .cachewriter import ensure as _ensure_cached
            _ensure_cached([f"/points/{c}" for c in coordinates])
        except Exception:
            # Writer unavailable: fall back to resolving inline (each call
            # fetches on miss via connection's inline fallback).
            pass

        seen = set()
        ordered = []
        # With /points warmed, resolve gridpoints. The thread pool keeps this
        # fast even though each call is now just a cache read; dedupe since
        # adjacent samples often map to the same gridpoint.
        with concurrent.futures.ThreadPoolExecutor(max_workers=NOAA_MAX_CONCURRENCY) as executor:
            for gp in executor.map(gridpoint_from_map_coordinate, coordinates):
                if gp is None:
                    continue  # out-of-coverage sample (e.g. across the border)
                if gp in seen:
                    continue
                seen.add(gp)
                ordered.append(gp)
                yield gp

        with _GRIDPOINTS_LOCK:
            _GRIDPOINTS_CACHE[self.signature] = ordered

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

        # Phase 1: /points. Its total is the not-fresh point count.
        point_paths = not_fresh([f"/points/{c}" for c in self.map_coordinates])
        yield {"phase": "points", "done": 0, "total": len(point_paths)}
        for prog in ensure_stream(point_paths):
            yield {"phase": "points", "done": prog["done"], "total": len(point_paths)}

        # Phase 2: /gridpoints (gridpoint set now known from phase 1).
        grid_paths = not_fresh([f"/gridpoints/{gp}" for gp in self.gridpoints])
        yield {"phase": "gridpoints", "done": 0, "total": len(grid_paths)}
        for prog in ensure_stream(grid_paths):
            yield {"phase": "gridpoints", "done": prog["done"], "total": len(grid_paths)}


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


