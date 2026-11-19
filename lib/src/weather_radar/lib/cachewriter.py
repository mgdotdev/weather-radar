"""Single-writer cache process.

All writes to the on-disk NOAA cache go through ONE background process. Web
workers are readers only: on a cache miss (or expiry) they submit the needed
paths to this process and wait for it to fetch + atomically write them.

Design
------
* The writer process hosts a ``multiprocessing.managers.BaseManager`` server
  listening on a local socket. Web workers (separate uvicorn worker processes)
  connect to it as clients using a shared address + authkey, so the queue is
  shared regardless of how the workers were started (fork or spawn).
* Work is a set of NOAA API paths to "ensure cached". The writer fetches any
  that are missing or whose cached ``Expires`` has passed (which is how new
  forecasts get picked up), using a bounded thread pool so we stay within
  NOAA's tolerances. Each response is written to a temp file and then
  ``os.replace``'d into place so readers never observe a partial file.
* Two entry points are exposed to clients:
    - ``ensure(paths)``   : block until every path is cached (model A).
    - ``ensure_stream(paths)`` : returns a Queue that yields incremental
                                  ``{"done", "total"}`` progress then ``None``.

Only the Python standard library is used (plus ``requests`` for HTTP, already
a dependency of the connection layer).
"""

import os
import json
import queue
import logging
import threading
import tempfile
import time
import math
import glob
import concurrent.futures
from datetime import datetime, timezone, timedelta
from multiprocessing.managers import BaseManager

import requests


logger = logging.getLogger(__name__)

# Max simultaneous NOAA requests the writer makes. We respect 429/Retry-After
# (see _retry_get) so this can be generous — NOAA throttles us if we overdo it
# and we back off automatically. Kept in sync with area.NOAA_MAX_CONCURRENCY.
MAX_CONCURRENCY = int(os.environ.get("NOAA_MAX_CONCURRENCY", "48"))

# Max simultaneous BINARY (GRIB range) fetches in ensure_binary_many. Kept
# LOWER than MAX_CONCURRENCY by default: a burst of dozens of parallel TLS
# handshakes + DNS lookups to the same host (NBM/NAQFC on S3) can overwhelm a
# constrained container's resolver (sporadic getaddrinfo failures) and gains
# little over a modest pool for ~0.5 MB records. Tunable via env.
BINARY_CONCURRENCY = int(os.environ.get("WR_BINARY_CONCURRENCY", "8"))

# Address + auth for the manager socket. A TCP loopback address keeps this
# working across fork/spawn and is trivial to connect to from worker procs.
CACHE_WRITER_HOST = os.environ.get("CACHE_WRITER_HOST", "127.0.0.1")
CACHE_WRITER_PORT = int(os.environ.get("CACHE_WRITER_PORT", "8137"))
CACHE_WRITER_AUTHKEY = os.environ.get("CACHE_WRITER_AUTHKEY", "weather-radar-cache").encode()

NOAA_BASE_URL = os.environ.get("NOAA_BASE_URL", "https://api.weather.gov")
CACHE_ROOT = os.environ.get("CACHE_DIR", "/tmp")


def _noaa_user_agent(warn=False):
    """NOAA User-Agent for the writer's session. Delegates to
    connection.noaa_user_agent when importable as a package submodule; when
    this module is loaded standalone (e.g. the test harness loads it by file
    path, so relative imports can't resolve), fall back to the SAME env-driven
    logic inline so behavior is identical.

    `warn` controls the one-time "no contact configured" warning. It defaults
    to False here because the writer resolves this once at __init__ just to set
    its session UA — nagging then would fire even under NBM/NDFD/AQI, which
    never hit api.weather.gov. The warning is instead triggered lazily at the
    actual gridpoints fetch (see _fetch_one), so it only appears when a NOAA
    JSON request is really made.
    """
    try:
        from .connection import noaa_user_agent
        return noaa_user_agent(warn=warn)
    except (ImportError, ValueError):
        override = os.environ.get("NOAA_USER_AGENT")
        if override and override.strip():
            return override.strip()
        app = (os.environ.get("WR_APP_NAME") or "weather-radar").strip()
        email = (os.environ.get("WR_CONTACT_EMAIL") or "").strip()
        return f"{app} ({email})" if email else app


# --- Path / cache-file helpers (kept independent of connection.PathObj) -----

def _tempfile_for(path):
    """Return (dir, data.json path) for a NOAA API path, mirroring the layout
    connection.PathObj uses so readers find what the writer writes."""
    from urllib.parse import urljoin, urlparse
    url = urljoin(NOAA_BASE_URL + "/", path.lstrip("/"))
    parse = urlparse(url)
    d = os.path.join(CACHE_ROOT, parse.netloc, parse.path[1:])
    return url, d, os.path.join(d, "data.json")


def _stamp_expiry(cache_file, expires_header):
    """Set `cache_file`'s mtime to its NOAA `Expires` time so freshness checks
    are a cheap os.stat (see _is_fresh). Best-effort; a missing/unparseable
    header leaves mtime at the write time (which _is_fresh handles via its
    JSON fallback)."""
    if not expires_header:
        return
    try:
        exp = datetime.strptime(expires_header, "%a, %d %b %Y %H:%M:%S %Z")
        ts = exp.replace(tzinfo=timezone.utc).timestamp()
        os.utime(cache_file, (ts, ts))
    except Exception:
        pass


# Sidecar that marks when a path's CONTENT last changed (as opposed to when it
# was last refreshed). The data.json mtime is overloaded for freshness (it's
# stamped to Expires), so it advances on EVERY refetch — even when NOAA returns
# identical data. Derived artifacts (fitted splines) must refit only on a real
# content change, so they key on THIS marker's mtime instead: the writer only
# touches it when the content hash actually differs. A tiny file -> cheap stat.
def _change_marker(cache_file):
    return cache_file + ".changed"


def _content_hash(payload):
    """Stable hash of a response's CONTENT (the NOAA `json` body only), so
    volatile headers (Expires/Date/Age) don't count as a change. Returns a hex
    digest string."""
    import hashlib
    body = payload.get("json") if isinstance(payload, dict) else payload
    # sort_keys + compact separators => canonical, order-independent bytes.
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_existing_hash(cache_file):
    """Return the stored content hash for `cache_file`, or None. Prefers the
    cheap marker sidecar (which holds the hash); falls back to hashing the
    on-disk body for files written before markers existed."""
    marker = _change_marker(cache_file)
    try:
        with open(marker, "r") as f:
            h = f.read().strip()
            if h:
                return h
    except OSError:
        pass
    # Legacy file with no marker: hash its stored body so an unchanged refetch
    # is still recognised as unchanged (and gets a marker written this time).
    try:
        with open(cache_file, "r") as f:
            return _content_hash(json.load(f))
    except Exception:
        return None


def _write_change_marker(cache_file, content_hash):
    """Record `content_hash` in the change marker and bump its mtime to NOW
    (the moment content changed). Best-effort."""
    marker = _change_marker(cache_file)
    try:
        # Atomic: write temp + replace so a reader never sees a partial hash.
        d = os.path.dirname(marker)
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            f.write(content_hash)
        os.replace(tmp, marker)
    except OSError:
        pass


# --- Local /points resolution from cached /gridpoints polygons --------------
# Option A: before the writer issues a /points network call to api.weather.gov,
# try to resolve the point LOCALLY against the polygons of /gridpoints docs we
# already have on disk. Every NOAA gridpoint doc carries its cell polygon
# (geometry.coordinates[0]); if the requested lat/lon falls inside one, that
# gridpoint IS the point's answer (NOAA's /points -> grid mapping is just
# "which cell contains this coordinate"). When we find a containing polygon we
# synthesize the /points data.json ourselves and skip the network entirely.
#
# The index is built by scanning the on-disk /gridpoints cache and bucketing
# each polygon by its centroid (coarse lat/lon buckets, same scheme as
# area._POLY_INDEX). It's cached in-process and rebuilt only when the
# gridpoints cache directory's mtime changes (a new doc was written), so the
# scan cost is amortized across many /points resolutions. Disable with
# WR_POLYGON_INDEX=0 (mirrors the reader-side gate in area.py).

_POLY_INDEX_BUCKET_DEG = 0.25  # ~25-28 km cells; a handful of gridpoints each

# How long a synthesized /points doc stays "fresh". The /points -> grid mapping
# is effectively static (NOAA office grids don't move), so a long TTL is safe;
# a rare re-probe costs one network /points. Overridable for testing.
_SYNTH_POINTS_TTL_SEC = int(os.environ.get("WR_POINTS_SYNTH_TTL_SECONDS", str(7 * 24 * 3600)))

# In-process polygon index: bucket (bx,by) -> list of ((gridId, x, y), ring).
# Guarded by a lock since the writer's thread pool resolves paths concurrently.
_poly_index = None
_poly_index_dir_mtime = None
_poly_index_lock = threading.Lock()


def _gridpoints_root():
    """Directory that holds the cached /gridpoints docs."""
    return os.path.join(CACHE_ROOT, "api.weather.gov", "gridpoints")


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


def _dir_mtime(root):
    """mtime of the gridpoints tree, used to decide whether to rebuild the
    index. We stat the top dir and its immediate WFO subdirs: a newly-written
    gridpoint doc creates/updates a <WFO>/<x,y> dir, which bumps the WFO dir's
    mtime. Cheap relative to re-scanning every doc."""
    latest = 0.0
    try:
        latest = os.path.getmtime(root)
    except OSError:
        return None
    try:
        for wfo in os.scandir(root):
            if wfo.is_dir():
                try:
                    latest = max(latest, wfo.stat().st_mtime)
                except OSError:
                    pass
    except OSError:
        pass
    return latest


def _build_poly_index(root):
    """Scan the on-disk /gridpoints cache and return a centroid-bucketed index
    {bucket: [((gridId, x, y), ring), ...]}. Best-effort: unreadable/partial
    docs are skipped."""
    index = {}
    # Layout: <root>/<WFO>/<x>,<y>/data.json
    for data_file in glob.iglob(os.path.join(root, "*", "*", "data.json")):
        try:
            # .../gridpoints/<WFO>/<x>,<y>/data.json
            xy_dir = os.path.dirname(data_file)
            wfo = os.path.basename(os.path.dirname(xy_dir))
            x_str, y_str = os.path.basename(xy_dir).split(",")
            gid = (wfo, int(x_str), int(y_str))
        except Exception:
            continue
        try:
            with open(data_file) as f:
                doc = json.load(f)
            ring = doc["json"]["geometry"]["coordinates"][0]
            clon, clat = _ring_centroid(ring)
        except Exception:
            continue
        index.setdefault(_poly_bucket(clon, clat), []).append((gid, ring))
    return index


def _get_poly_index():
    """Return the current polygon index, rebuilding it if the gridpoints cache
    directory has changed since the last build. Returns {} if the cache dir is
    absent or the feature is disabled."""
    if os.environ.get("WR_POLYGON_INDEX", "1") == "0":
        return {}
    root = _gridpoints_root()
    mtime = _dir_mtime(root)
    if mtime is None:
        return {}
    global _poly_index, _poly_index_dir_mtime
    with _poly_index_lock:
        if _poly_index is None or _poly_index_dir_mtime != mtime:
            _poly_index = _build_poly_index(root)
            _poly_index_dir_mtime = mtime
        return _poly_index


def _parse_points_path(path):
    """Return (lat, lon) floats for a '/points/{lat},{lon}' path, or None if it
    isn't a resolvable /points path."""
    p = path.lstrip("/")
    if not p.startswith("points/"):
        return None
    rest = p[len("points/"):]
    try:
        lat_s, lon_s = rest.split(",")
        return float(lat_s), float(lon_s)
    except ValueError:
        return None


def _lookup_gridpoint(lon, lat):
    """Return (gridId, x, y) for the cached gridpoint polygon containing
    (lon,lat), or None. Checks the point's bucket and its 8 neighbors (a
    polygon may straddle a bucket boundary)."""
    index = _get_poly_index()
    if not index:
        return None
    bx, by = _poly_bucket(lon, lat)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for gid, ring in index.get((bx + dx, by + dy), ()):
                if _point_in_ring(lon, lat, ring):
                    return gid
    return None


def _resolve_points_locally(path, cache_file, d):
    """Try to satisfy a '/points/{lat},{lon}' request from cached /gridpoints
    polygons, WITHOUT a network call. On success, synthesize and atomically
    write the /points data.json (mirroring NOAA's shape: properties.gridId /
    gridX / gridY, plus an Expires header so the reader's freshness check
    works) and return True. Returns False if the point can't be resolved
    locally (caller then falls through to the network fetch)."""
    parsed = _parse_points_path(path)
    if parsed is None:
        return False
    lat, lon = parsed
    gid = _lookup_gridpoint(lon, lat)
    if gid is None:
        return False
    wfo, gx, gy = gid
    expires_dt = datetime.now(timezone.utc) + timedelta(seconds=_SYNTH_POINTS_TTL_SEC)
    expires_hdr = expires_dt.strftime("%a, %d %b %Y %H:%M:%S GMT")
    payload = {
        "headers": {
            "Expires": expires_hdr,
            "Content-Type": "application/geo+json",
            # Marks this doc as writer-synthesized (vs. a real NOAA response),
            # for debugging; readers ignore unknown headers.
            "X-WR-Synthesized": "points-from-gridpoint",
        },
        "json": {
            "properties": {
                "gridId": wfo,
                "gridX": gx,
                "gridY": gy,
            }
        },
    }
    try:
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f)
            os.replace(tmp, cache_file)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        _stamp_expiry(cache_file, expires_hdr)
        logger.debug("cachewriter: resolved %s locally -> %s/%s,%s (no network)",
                     path, wfo, gx, gy)
        return True
    except OSError:
        return False


def _is_fresh(cache_file):
    """True if the cache file exists and is unexpired.

    Fast path: the writer sets each file's MTIME to its NOAA `Expires` time
    (see _fetch_one), so freshness is a single os.stat — no read/parse. This
    matters a lot: an area can have thousands of cached /gridpoints docs (each
    large), and checking them all by parsing JSON dominated prefetch/model-
    switch time. Falls back to parsing the stored Expires header for files
    written before this optimization (mtime not yet meaningful)."""
    try:
        mtime = os.path.getmtime(cache_file)
    except OSError:
        return False
    # mtime == expiry epoch. A file whose mtime is in the future is fresh.
    if mtime >= time.time():
        return True
    # Legacy/ambiguous: mtime in the past could mean "expired" OR "written
    # before we set mtime=expiry" (where mtime is the write time). Parse the
    # stored Expires to be sure, and (cheaply) correct the mtime for next time.
    try:
        with open(cache_file, "r") as f:
            body = json.loads(f.read())
        expires = body["headers"]["Expires"]
        exp = datetime.strptime(expires, "%a, %d %b %Y %H:%M:%S %Z")
        fresh = exp >= datetime.now()
        if fresh:
            # Repair mtime so future checks hit the fast path.
            try:
                ts = exp.replace(tzinfo=timezone.utc).timestamp()
                os.utime(cache_file, (ts, ts))
            except OSError:
                pass
        return fresh
    except Exception:
        return False


def is_fresh(path):
    """True if the NOAA `path` does NOT need (re)fetching: either its cache
    file exists and is unexpired, OR it's tombstoned as permanently
    no-coverage (a 404 cell that NOAA will never serve) or tombstoned as a
    recent server error (5xx). A pure local disk read (no IPC), so callers can
    cheaply filter before enqueuing."""
    _url, _d, cache_file = _tempfile_for(path)
    if _is_fresh(cache_file):
        return True
    return _tombstone_valid(cache_file)


# Tombstone TTLs. A 404 (no coverage) is effectively permanent — NOAA office
# coverage is static — so trust it for a long time (a rare re-probe costs one
# 404). A 5xx is a transient-ish NOAA server error on a specific resource;
# trust it only briefly so we retry soon once NOAA recovers, but not on EVERY
# pass (which spammed retries + tracebacks).
NOCOVERAGE_TTL_SEC = 7 * 24 * 3600
SERVERERROR_TTL_SEC = 10 * 60


def _tombstone_file(cache_file):
    return cache_file + ".tombstone"


def _write_tombstone(path, ttl):
    """Record that `path` should NOT be re-fetched for `ttl` seconds (a 404 or
    persistent 5xx). Stores expiry epoch. Best-effort."""
    _url, d, cache_file = _tempfile_for(path)
    try:
        os.makedirs(d, exist_ok=True)
        with open(_tombstone_file(cache_file), "w") as f:
            f.write(str(time.time() + ttl))
    except OSError:
        pass


def _tombstone_valid(cache_file):
    """True if a tombstone exists and its stored expiry is still in the future."""
    try:
        with open(_tombstone_file(cache_file)) as f:
            expiry = float(f.read().strip())
    except (OSError, ValueError):
        return False
    return time.time() < expiry


def not_fresh(paths):
    """Return the subset of `paths` that are missing or expired (i.e. the ones
    that actually need to be (re)written)."""
    return [p for p in paths if not is_fresh(p)]


# --- HTTP (bounded, polite) --------------------------------------------------

def _retry_time(attempt):
    return min(2 ** attempt, 16)


class _NoCoverage(Exception):
    """NOAA 404 for a path it can't serve (point/cell past coverage). Expected
    during gridpoint-walk edge probing — NOT an error to log loudly."""


class _ServerError(Exception):
    """Persistent NOAA 5xx for a specific resource (seen: /gridpoints 500 on
    certain cells). Tombstoned briefly so it isn't retried every pass."""


def _retry_get(session, url, retries=6, attempt=0):
    """GET with 429/Retry-After handling + exponential backoff. Mirrors
    connection.retry_get but lives in the writer so all network writes share
    one implementation."""
    import time as _time
    # Transient network failures (DNS hiccup, connection refused/reset, read
    # timeout) raise before a response exists. Retry with backoff rather than
    # letting a brief blip permanently drop a path from the cache.
    try:
        resp = session.get(url, timeout=30)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        if not retries:
            raise RuntimeError(f"NWS unreachable: {url}") from e
        logger.warning(
            "cachewriter transient error (%s); backing off %ss: %s",
            type(e).__name__, _retry_time(attempt), url,
        )
        _time.sleep(_retry_time(attempt))
        return _retry_get(session, url, retries - 1, attempt + 1)

    if resp.status_code == 200:
        return resp
    # Permanent no-coverage (NOAA 404): never retry. Fail fast with a
    # recognizable exception so _fetch_one can treat it as an expected miss
    # (no traceback) — points/cells outside NOAA coverage return 404 normally.
    if resp.status_code == 404:
        raise _NoCoverage(url)
    if not retries:
        # Exhausted retries. A persistent 5xx is a NOAA-side error on this
        # specific resource (seen: /gridpoints 500 on certain cells); surface
        # it as a recognizable exception so _fetch_one can tombstone it briefly
        # and NOT retry it every pass (which spammed tracebacks). Other codes
        # keep the old raise_for_status behavior.
        if 500 <= resp.status_code < 600:
            raise _ServerError(f"{resp.status_code} from {url}")
        resp.raise_for_status()
        raise RuntimeError(f"NWS unresponsive: {url}")
    # Rate limiting. NOAA usually signals this with 429 (+ Retry-After), but
    # under a sustained burst it may instead return 403 Forbidden — same cause,
    # so treat it the same: honor Retry-After if present, else back off
    # exponentially, rather than burning short retries against an active block.
    if resp.status_code in (429, 403):
        ra = resp.headers.get("Retry-After")
        try:
            wait = float(ra)
        except (TypeError, ValueError):
            wait = _retry_time(attempt)
        logger.warning(
            "cachewriter %s throttled; backing off %ss: %s",
            resp.status_code, wait, url,
        )
        _time.sleep(wait)
    else:
        _time.sleep(_retry_time(attempt))
    return _retry_get(session, url, retries - 1, attempt + 1)


# Terminal sentinel for streaming jobs. A string so it survives serialization
# across the manager boundary (an object()'s identity would not).
_JOB_DONE = "__done__"


class _Writer:
    """Owns the session + thread pool and performs the actual fetch/write."""

    def __init__(self):
        self._session = requests.session()
        # Resolve the NOAA UA here (not at import) so env configured before the
        # writer PROCESS starts is honored. Imported lazily to keep this module
        # independent of the connection layer (which lazily imports us).
        self._session.headers["User-Agent"] = _noaa_user_agent()
        # Size the connection pool so concurrent fetches REUSE connections
        # instead of thrashing (default urllib3 pool is ~10). Covers both the
        # JSON pool and the binary fan-out.
        try:
            from requests.adapters import HTTPAdapter
            pool_n = max(MAX_CONCURRENCY, BINARY_CONCURRENCY) + 4
            adapter = HTTPAdapter(pool_connections=pool_n, pool_maxsize=pool_n,
                                  max_retries=0)
            self._session.mount("https://", adapter)
            self._session.mount("http://", adapter)
        except Exception:
            pass  # best-effort; default pool still works
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENCY)
        # Separate, SMALLER pool for binary (GRIB range) fan-out so a cold NBM
        # fit doesn't open dozens of simultaneous TLS/DNS connections to one
        # host (which can trip a constrained container's resolver).
        self._bin_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=BINARY_CONCURRENCY)
        # Per-path locks so two overlapping jobs don't fetch the same URL twice.
        self._path_locks = {}
        self._registry_lock = threading.Lock()
        # Streaming jobs: id -> progress queue (server-side only).
        self._jobs = {}
        import itertools
        self._job_counter = itertools.count(1)

    def _lock_for(self, url):
        with self._registry_lock:
            lk = self._path_locks.get(url)
            if lk is None:
                lk = threading.Lock()
                self._path_locks[url] = lk
            return lk

    def _fetch_one(self, path):
        """Ensure a single path is cached+fresh. Returns True if the file is
        fresh on disk when we return (already-fresh or newly written), False
        if the fetch failed. Safe to call concurrently for different paths."""
        url, d, cache_file = _tempfile_for(path)
        if _is_fresh(cache_file) or _tombstone_valid(cache_file):
            return True
        lk = self._lock_for(url)
        with lk:
            # Re-check inside the lock (another thread may have just written).
            if _is_fresh(cache_file) or _tombstone_valid(cache_file):
                return True
            # Option A: resolve /points locally from cached /gridpoints
            # polygons before hitting the network. If the point falls inside a
            # gridpoint we already have on disk, synthesize the /points doc and
            # skip the api.weather.gov call entirely.
            if _resolve_points_locally(path, cache_file, d):
                return True
            try:
                # About to actually hit api.weather.gov — this is the only path
                # that needs the NWS contact UA, so emit the one-time "no
                # contact configured" warning HERE (lazily) rather than at
                # writer startup. No-op after the first call / when configured.
                _noaa_user_agent(warn=True)
                resp = _retry_get(self._session, url)
                payload = {"headers": dict(resp.headers), "json": resp.json()}
                os.makedirs(d, exist_ok=True)

                # Content-change check: if NOAA returned the SAME content we
                # already have, skip the rewrite entirely and only refresh the
                # freshness stamp (mtime = new Expires). This leaves the change
                # marker untouched, so derived artifacts (splines) keyed on the
                # marker's mtime do NOT needlessly refit on a no-op refresh.
                new_hash = _content_hash(payload)
                old_hash = _read_existing_hash(cache_file)
                if old_hash == new_hash and os.path.exists(cache_file):
                    # Unchanged: extend freshness only.
                    _stamp_expiry(cache_file, payload["headers"].get("Expires"))
                    # Ensure the marker exists (legacy files may lack one) but
                    # DON'T bump its mtime — content didn't change.
                    if not os.path.exists(_change_marker(cache_file)):
                        _write_change_marker(cache_file, new_hash)
                    try:
                        os.unlink(_tombstone_file(cache_file))
                    except OSError:
                        pass
                    return True

                # Changed (or new): atomic publish — write to a temp file in
                # the same dir, then os.replace() so readers only ever see a
                # complete file.
                fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
                try:
                    with os.fdopen(fd, "w") as f:
                        json.dump(payload, f)
                    os.replace(tmp, cache_file)
                finally:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
                # Record the new content hash + bump the change marker's mtime
                # (content changed NOW) so derived artifacts know to refit.
                _write_change_marker(cache_file, new_hash)
                # Stamp the file's mtime with its NOAA Expires time so the
                # freshness check (_is_fresh) is a cheap os.stat instead of a
                # JSON parse of this (large) doc. Best-effort.
                _stamp_expiry(cache_file, payload["headers"].get("Expires"))
                # Clear any stale tombstone (the resource is serving again).
                try:
                    os.unlink(_tombstone_file(cache_file))
                except OSError:
                    pass
                return True
            except _NoCoverage:
                # Permanent: this cell/point is past NOAA's coverage (the
                # index-box fill / grid-edge walk can over-include cells an
                # office doesn't actually have). NOAA keeps 404ing it, so
                # tombstone it (long TTL) and stop re-fetching every pass —
                # otherwise these phantom edges show as "need fetch" forever.
                _write_tombstone(path, NOCOVERAGE_TTL_SEC)
                logger.debug("cachewriter: no coverage for %s (tombstoned)", path)
                return False
            except _ServerError as e:
                # Transient-ish NOAA 5xx on a specific resource. Tombstone
                # briefly (short TTL) so we retry soon once NOAA recovers but
                # don't hammer it (and spam tracebacks) every pass. Logged at
                # WARNING WITHOUT a traceback — it's a NOAA-side error, not ours.
                _write_tombstone(path, SERVERERROR_TTL_SEC)
                logger.warning("cachewriter: NOAA server error for %s (%s); "
                               "tombstoned %ds", path, e, SERVERERROR_TTL_SEC)
                return False
            except Exception:
                logger.warning("cachewriter failed to cache %s", path, exc_info=True)
                return False

    def ensure(self, paths):
        """Block until all paths are cached+fresh."""
        futures = [self._pool.submit(self._fetch_one, p) for p in paths]
        concurrent.futures.wait(futures)

    def ensure_binary(self, url, path, force=False, rng=None, headers=None,
                      stamp_expires=True):
        """Download an arbitrary BINARY resource (e.g. a NAQFC GRIB2 from
        NOMADS, or an NBM record via an HTTP byte range) to `path`, atomically,
        so ALL disk writes stay inside the single writer process (no
        cross-worker download races).

        Freshness: by default the file's mtime is stamped to the HTTP `Expires`
        header (cheap stat-based freshness, same model as the JSON cache). For
        sources that send no usable Expires and instead use an age-based TTL
        (NBM/NDFD), pass `stamp_expires=False` so the mtime stays at write time
        (== now) and the caller's `age < TTL` check reads a just-written file as
        fresh.

        `rng` (e.g. "123-456") issues an HTTP Range request (accepting 206) so
        only that byte range is fetched — used for NBM's per-record reads.
        `headers` supplies extra request headers (e.g. a source User-Agent).

        De-duped via the per-URL(+range) lock: concurrent callers for the same
        resource collapse to one download. Returns a JSON string {"ok": bool,
        "path": path}. Skips the download when `path` is already fresh unless
        `force`.
        """
        return json.dumps(self._ensure_binary_one(
            url, path, force, rng, headers, stamp_expires))

    def ensure_binary_many(self, specs):
        """Download MANY binary resources CONCURRENTLY in this one writer
        process, fanning out over the shared thread pool (MAX_CONCURRENCY).
        Used by the NBM fit to fetch all of a cycle's per-record byte ranges in
        parallel instead of one blocking RPC per record.

        `specs` is a list of dicts, each:
          {url, path, force?, rng?, headers?, stamp_expires?}
        Returns a JSON string: a list of {ok, path} in the SAME ORDER as
        `specs`. Per-resource de-dup still applies via the per-URL(+range) lock,
        so overlapping specs for the same range collapse to one fetch.
        """
        results = [None] * len(specs)
        futs = {}
        for i, s in enumerate(specs):
            futs[self._bin_pool.submit(
                self._ensure_binary_one,
                s.get("url"), s.get("path"), s.get("force", False),
                s.get("rng"), s.get("headers"), s.get("stamp_expires", True),
            )] = i
        for fut in concurrent.futures.as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception:
                p = specs[i].get("path")
                results[i] = {"ok": bool(p and os.path.exists(p)), "path": p}
        return json.dumps(results)

    def _ensure_binary_one(self, url, path, force=False, rng=None, headers=None,
                           stamp_expires=True):
        """Core single-resource binary fetch+write. Returns a dict
        {"ok": bool, "path": path}. Shared by ensure_binary (one) and
        ensure_binary_many (fanned out over the pool). Safe to call
        concurrently for distinct url+range pairs."""
        def _fresh(p):
            try:
                return os.path.getmtime(p) >= time.time()
            except OSError:
                return False

        # Expires-based freshness only applies when we stamp Expires. TTL-based
        # callers always pass force=True (their own _is_fresh gates the call),
        # so this fast-path mainly helps the AQI (Expires) path.
        if not force and stamp_expires and _fresh(path):
            return {"ok": True, "path": path}

        lk = self._lock_for(url if rng is None else f"{url}#{rng}")
        with lk:
            if not force and stamp_expires and _fresh(path):   # re-check in lock
                return {"ok": True, "path": path}
            d = os.path.dirname(path)
            req_headers = dict(headers) if headers else {}
            if rng is not None:
                req_headers["Range"] = f"bytes={rng}"
            try:
                os.makedirs(d, exist_ok=True)
            except OSError:
                pass
            # Retry transient connection/DNS/timeout failures a few times with
            # backoff before giving up — a burst of concurrent fetches can
            # momentarily trip name resolution on a constrained host. A genuine
            # bad response (non-2xx/206) is NOT retried.
            attempts = 3
            last_exc = None
            for attempt in range(attempts):
                tmp = None
                try:
                    with self._session.get(url, timeout=180, stream=True,
                                           headers=req_headers or None) as r:
                        if not (r.ok or r.status_code == 206):
                            logger.warning("cachewriter binary %s -> HTTP %s",
                                           url, r.status_code)
                            return {"ok": os.path.exists(path), "path": path}
                        expires = r.headers.get("Expires")
                        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
                        with os.fdopen(fd, "wb") as f:
                            for chunk in r.iter_content(chunk_size=1 << 20):
                                f.write(chunk)
                        os.replace(tmp, path)
                        tmp = None
                    # mtime = Expires -> cheap freshness. Skipped for TTL
                    # sources (stamp_expires=False) so mtime stays at write time.
                    if stamp_expires and expires:
                        try:
                            exp = datetime.strptime(expires, "%a, %d %b %Y %H:%M:%S %Z")
                            ts = exp.replace(tzinfo=timezone.utc).timestamp()
                            os.utime(path, (ts, ts))
                        except Exception:
                            pass
                    logger.warning("cachewriter binary wrote %s (%d bytes, expires %s)",
                                   os.path.basename(path),
                                   os.path.getsize(path), expires)
                    return {"ok": True, "path": path}
                except Exception as e:
                    last_exc = e
                    if tmp and os.path.exists(tmp):
                        try:
                            os.unlink(tmp)
                        except OSError:
                            pass
                    if attempt < attempts - 1:
                        time.sleep(0.5 * (attempt + 1))   # 0.5s, 1.0s backoff
                        continue
            # All attempts failed — transient-looking; log without a traceback
            # spam and report not-ok so the caller can retry on the NEXT load
            # (NBM does NOT negative-cache a failed binary fetch).
            logger.warning("cachewriter binary failed for %s (%s)", url, last_exc)
            return {"ok": os.path.exists(path), "path": path}

    # --- streaming jobs ---
    # Each streaming job keeps its progress queue ENTIRELY on the writer side
    # (a real queue.Queue, never a proxy). Clients drive it by id via
    # start_job()/poll_job(), so no queue proxy is ever passed through a
    # manager call (which would deadlock by re-entering the server).
    def start_job(self, paths):
        job_id = next(self._job_counter)
        q = queue.Queue()
        self._jobs[job_id] = q

        def worker():
            total = len(paths)
            done = 0
            try:
                q.put({"done": done, "total": total})
                futures = [self._pool.submit(self._fetch_one, p) for p in paths]
                for fut in concurrent.futures.as_completed(futures):
                    # Advance only on a successful write/fresh file, so the
                    # count reflects real cache state, not mere attempts.
                    if fut.result():
                        done += 1
                    q.put({"done": done, "total": total})
            finally:
                q.put(_JOB_DONE)

        threading.Thread(target=worker, daemon=True).start()
        # Return a JSON string so the value crosses the manager boundary
        # by value (not wrapped in a proxy).
        return json.dumps({"job_id": job_id})

    def poll_job(self, job_id):
        """Block for the next progress update, then drain any further items
        already queued, returning them as a BATCH. Returning all currently
        available items per call (instead of one) means a burst of parallel
        fetch completions is delivered together, so the client's counter
        reflects real concurrency instead of a metronomic one-per-round-trip
        drip. Returns a JSON string: {"items": [ {done,total}, ... ],
        "done": bool} where done=True once the job has ended."""
        q = self._jobs.get(job_id)
        if q is None:
            return json.dumps({"items": [], "done": True})
        items = []
        ended = False
        # Block for at least one item so we don't busy-spin.
        first = q.get()
        if first == _JOB_DONE:
            self._jobs.pop(job_id, None)
            return json.dumps({"items": [], "done": True})
        items.append(first)
        # Drain whatever else is immediately available.
        while True:
            try:
                nxt = q.get_nowait()
            except queue.Empty:
                break
            if nxt == _JOB_DONE:
                ended = True
                self._jobs.pop(job_id, None)
                break
            items.append(nxt)
        return json.dumps({"items": items, "done": ended})



# --- Manager wiring ----------------------------------------------------------

class _CacheManager(BaseManager):
    pass


def run_writer(ready_event=None):
    """Entry point for the writer process: start the manager server and serve
    forever. If `ready_event` is given, set it once the server is listening."""
    writer = _Writer()

    _CacheManager.register("ensure", callable=writer.ensure)
    _CacheManager.register("ensure_binary", callable=writer.ensure_binary)
    _CacheManager.register("ensure_binary_many", callable=writer.ensure_binary_many)
    _CacheManager.register("start_job", callable=writer.start_job)
    _CacheManager.register("poll_job", callable=writer.poll_job)

    mgr = _CacheManager(
        address=(CACHE_WRITER_HOST, CACHE_WRITER_PORT),
        authkey=CACHE_WRITER_AUTHKEY,
    )
    server = mgr.get_server()
    logger.warning("cache writer listening on %s:%s", CACHE_WRITER_HOST, CACHE_WRITER_PORT)
    if ready_event is not None:
        ready_event.set()
    server.serve_forever()


# --- Client side (used by web workers) --------------------------------------

# Client connection state. `_client` is the live connection; `_client_failed`
# records that connecting failed so we don't retry (and block) on every miss.
_client = None
_client_failed = False
_client_lock = threading.Lock()


class _CacheClient(BaseManager):
    pass


_CacheClient.register("ensure")
_CacheClient.register("ensure_binary")
_CacheClient.register("ensure_binary_many")
_CacheClient.register("start_job")
_CacheClient.register("poll_job")


class WriterUnavailable(Exception):
    """Raised when the cache-writer process can't be reached, so callers can
    fall back to fetching inline."""


def _get_client():
    """Connect (once per worker process) to the writer's manager server.
    Raises WriterUnavailable if it can't connect; the failure is remembered so
    we don't block retrying on every cache miss."""
    global _client, _client_failed
    if _client is not None:
        return _client
    if _client_failed:
        raise WriterUnavailable()
    with _client_lock:
        if _client is not None:
            return _client
        if _client_failed:
            raise WriterUnavailable()
        c = _CacheClient(
            address=(CACHE_WRITER_HOST, CACHE_WRITER_PORT),
            authkey=CACHE_WRITER_AUTHKEY,
        )
        try:
            c.connect()
        except Exception as e:
            _client_failed = True
            raise WriterUnavailable() from e
        _client = c
        return _client


def _value(proxy):
    """Registered manager methods return an AutoProxy to the result; this
    materializes the real (by-value) return, e.g. the JSON string."""
    try:
        return proxy._getvalue()
    except AttributeError:
        return proxy


def ensure(paths):
    """Block until the writer has cached all `paths`. Paths already fresh on
    disk are NOT enqueued (they're just read by the caller); only missing or
    expired paths go to the writer. Raises WriterUnavailable if the writer
    process isn't reachable AND there is work to do."""
    todo = not_fresh(paths)
    if not todo:
        return  # everything is already fresh; nothing to write
    client = _get_client()
    client.ensure(todo)


def ensure_binary(url, path, force=False, rng=None, headers=None,
                  stamp_expires=True):
    """Have the WRITER process download a binary resource at `url` to `path`
    (atomic). Used for NAQFC GRIB2 files and NBM/NDFD GRIB records so ALL disk
    writes stay in the one writer. `rng` issues an HTTP Range request (NBM
    per-record reads); `headers` adds request headers (e.g. a source UA);
    `stamp_expires=False` skips Expires-mtime stamping for age-TTL sources
    (NBM/NDFD). Returns True if the file is present afterward. Raises
    WriterUnavailable if the writer can't be reached (caller may then fall back
    to an inline download)."""
    client = _get_client()
    import json as _json
    resp = client.ensure_binary(url, path, force, rng, headers, stamp_expires)
    try:
        return _json.loads(_value(resp)).get("ok", False)
    except Exception:
        return False


def ensure_binary_many(specs):
    """Have the WRITER process download MANY binary resources CONCURRENTLY (one
    RPC, fanned out over the writer's thread pool). `specs` is a list of dicts
    {url, path, force?, rng?, headers?, stamp_expires?}. Returns a list of
    bools (ok per spec, in order). Used by the NBM fit to fetch all of a
    cycle's per-record byte ranges in parallel instead of one blocking RPC per
    record. Raises WriterUnavailable if the writer can't be reached."""
    client = _get_client()
    import json as _json
    resp = client.ensure_binary_many(list(specs))
    try:
        arr = _json.loads(_value(resp))
        return [bool(r and r.get("ok")) for r in arr]
    except Exception:
        return [False] * len(specs)


def ensure_stream(paths):
    """Generator yielding {"done","total"} progress as the writer caches the
    paths that actually need writing (missing/expired). Already-fresh paths
    are skipped entirely, so the total reflects real work, not cache hits.
    Drives a server-side job by id (start_job/poll_job) so no queue proxy is
    passed through a manager call (which would deadlock)."""
    todo = not_fresh(paths)
    if not todo:
        return  # nothing to write -> no progress events
    client = _get_client()

    started = json.loads(_value(client.start_job(todo)))
    job_id = started["job_id"]
    while True:
        resp = json.loads(_value(client.poll_job(job_id)))
        # Batched: a list of {done,total} items plus a terminal `done` flag.
        for item in resp.get("items", []):
            yield {"done": item["done"], "total": item["total"]}
        if resp.get("done"):
            break

