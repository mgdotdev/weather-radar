"""Disk cache of finished single-point forecast series (the graph / forecast
page data: one FeatureCollection per (kind, attribute, point, step)).

Why: a point series is tiny (a few KB gzipped) and cheap to compute ONCE the
source grid is parsed (~0.2 s), but the first request in each worker after a new
forecast cycle pays for parsing ~100 full-grid frames (~11 s for NBM). Caching
the RESULT lets any worker answer from a file read instead.

Each entry is stamped with the data-version token also used by the bundles (the
forecast-cycle id for NBM, a file mtime for NDFD), so an entry is only served
for the data it was computed from:
  * token equals the current one      -> served;
  * entry is NEWER than our view      -> served (our token lags);
  * entry is OLDER but was written by the background pre-warm ("managed") and is
    younger than STALE_SECONDS         -> served while the pre-warm catches up;
  * otherwise                          -> a miss (the caller recomputes + rewrites).
Writes run in the cache-writer process (cachewriter.write_series_cache); a
worker writes inline only when the writer is unreachable. An entry is never
replaced by an OLDER token.
"""

import gzip
import hashlib
import json
import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

CACHE_ROOT = os.environ.get("CACHE_DIR", "/tmp")
SERIES_ROOT = os.path.join(CACHE_ROOT, "weather-radar-series")

# Bump when the CONTENT a series is computed to changes (not its inputs), so
# entries written by older code are never served. v2: AQI keeps one decimal
# instead of whole numbers (v1 entries were stair-stepped).
FORMAT_VERSION = 2

TOKEN_EPS = 1e-3
# How long a pre-warmed entry from the previous cycle may still be served while
# the builder recomputes it for the new one.
STALE_SECONDS = float(os.environ.get("WR_SERIES_STALE_SECONDS", "7200"))

_SAVE_LOCK = threading.Lock()
_READ_CACHE = {}               # path -> ((size, mtime_ns), entry)
_READ_CACHE_LOCK = threading.Lock()
_READ_CACHE_MAX = 64


def normalize_step(dt):
    """The time step a request resolves to (<=0 means the default hour)."""
    try:
        dt = int(dt)
    except (TypeError, ValueError):
        dt = 0
    return dt if dt > 0 else 3600


def path_for(kind, attribute, lat, lon, dt, root=None):
    """Cache file for one series. Coordinates are rounded to 6 decimals (as the
    graph links write them) and dt is normalized, so `dt=0` and `dt=3600` are
    one entry."""
    sig = "|".join(("v%d" % FORMAT_VERSION, str(kind), str(attribute),
                    "%.6f" % round(float(lat), 6), "%.6f" % round(float(lon), 6),
                    str(normalize_step(dt))))
    name = hashlib.sha256(sig.encode()).hexdigest()[:20] + ".json.gz"
    return os.path.join(root or SERIES_ROOT, name)


def pack(data, token, managed=False):
    """Serialize a series result into the cache file's bytes."""
    entry = {"token": None if token is None else float(token),
             "managed": bool(managed), "built_ts": time.time(), "data": data}
    return gzip.compress(json.dumps(entry, separators=(",", ":")).encode("utf-8"),
                         compresslevel=6, mtime=0)


def read(path):
    """The parsed entry ({token, managed, built_ts, data}) or None. Parsed
    entries are memoized per file version, so repeat reads cost a stat."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    ver = (st.st_size, st.st_mtime_ns)
    with _READ_CACHE_LOCK:
        hit = _READ_CACHE.get(path)
        if hit is not None and hit[0] == ver:
            return hit[1]
    try:
        with gzip.open(path, "rb") as f:
            entry = json.loads(f.read().decode("utf-8"))
    except Exception:
        return None
    with _READ_CACHE_LOCK:
        if len(_READ_CACHE) >= _READ_CACHE_MAX:
            _READ_CACHE.clear()
        _READ_CACHE[path] = (ver, entry)
    return entry


def _tok(entry):
    try:
        return None if entry.get("token") is None else float(entry["token"])
    except (TypeError, ValueError, AttributeError):
        return None


def is_fresh(path, current_token):
    """True if the file holds a result for the current data (or newer)."""
    entry = read(path)
    tok = _tok(entry) if entry else None
    return (current_token is not None and tok is not None
            and tok >= float(current_token) - TOKEN_EPS)


def get(path, current_token, now=None):
    """The cached series for the current data, or None (see module docstring).
    With an unknown current token nothing can be verified, so nothing is served."""
    if current_token is None:
        return None
    entry = read(path)
    tok = _tok(entry) if entry else None
    if tok is None:
        return None
    current = float(current_token)
    if tok >= current - TOKEN_EPS:
        return entry["data"]
    if entry.get("managed"):
        age = (time.time() if now is None else now) - float(entry.get("built_ts") or 0)
        if age < STALE_SECONDS:
            return entry["data"]
    return None


def save(path, blob, token, managed=False):
    """Atomically store `blob` (from pack()). Returns "ok", "dup" (already holds
    this data), "stale" (stored data is newer) or "error". Run in the writer."""
    token = None if token is None else float(token)
    with _SAVE_LOCK:
        existing = read(path)
        etok = _tok(existing) if existing else None
        if token is not None and etok is not None:
            if etok > token + TOKEN_EPS:
                return "stale"
            if abs(etok - token) <= TOKEN_EPS and (existing.get("managed") or not managed):
                return "dup"
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = "%s.tmp.%d.%d" % (path, os.getpid(), threading.get_ident())
            with open(tmp, "wb") as f:
                f.write(blob)
            os.replace(tmp, path)
            return "ok"
        except OSError:
            logger.warning("series cache write failed: %s", path, exc_info=True)
            return "error"
