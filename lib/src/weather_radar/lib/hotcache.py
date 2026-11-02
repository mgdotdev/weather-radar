"""Hot-cache scheduler process.

Keeps a set of "watched" app URLs warm by proactively refreshing their NOAA
cache files just before they expire, so navigating to a watched URL finds
everything already cached and current (not a shortened/stale forecast).

Architecture
------------
* Runs as its OWN process — a read-only client of the cache-writer, exactly
  like a web worker. It never writes cache files itself; it enqueues
  ``ensure(paths)`` jobs on the writer, which refetches only expired/missing
  paths (its freshness check) and atomically publishes them.
* State is an expiry min-heap of ``(expiry_ts, path)`` plus a
  ``url -> set(paths)`` map, guarded by a Condition. The loop sleeps until the
  soonest expiry (minus a lead time); dynamic ``watch``/``unwatch`` calls push
  into the heap and notify the Condition so a sleeping loop re-evaluates
  immediately and new entries slot into the schedule in order.
* The watched-URL list is persisted to JSON under CACHE_DIR so it survives
  restarts.
* Admin control (watch/unwatch/list) is exposed via its own BaseManager server
  on a separate port; web workers connect as clients.

Standard library only (plus the existing cachewriter client for the writer).
"""

import os
import json
import time
import heapq
import logging
import threading
from datetime import datetime
from multiprocessing.managers import BaseManager

from . import cachewriter


logger = logging.getLogger(__name__)

SCHED_HOST = os.environ.get("HOTCACHE_HOST", "127.0.0.1")
SCHED_PORT = int(os.environ.get("HOTCACHE_PORT", "8138"))
SCHED_AUTHKEY = os.environ.get(
    "HOTCACHE_AUTHKEY", "weather-radar-hotcache"
).encode()

# Refresh this many seconds BEFORE a file's Expires, so a watched URL is never
# momentarily stale when a user hits it.
LEAD_SECONDS = int(os.environ.get("HOTCACHE_LEAD_SECONDS", "120"))

# Floor on how long to sleep between wakeups even if nothing is due, so a burst
# of past-due entries doesn't spin.
MIN_SLEEP = 1.0
# Cap on sleep so we re-check periodically even if the heap is empty.
MAX_SLEEP = 300.0

CACHE_ROOT = os.environ.get("CACHE_DIR", "/tmp")
WATCHLIST_PATH = os.path.join(CACHE_ROOT, "weather-radar-hotcache-urls.json")


# --- URL -> NOAA paths -------------------------------------------------------

def paths_for_url(url):
    """Resolve an app URL to the set of NOAA API paths its cache depends on.

    Supports the map URL (/map/models/{model}/?lat&lon&width&height) and the
    graph URL (/graph/?lat&lon). Returns a list of paths like
    "/points/..." and "/gridpoints/WFO/x,y". Resolving gridpoints may enqueue
    /points fetches via the writer (that's fine — it warms them).
    """
    from urllib.parse import urlparse, parse_qs
    # Import here to avoid importing scipy-adjacent modules at process start.
    from .area import CoordinateArea, MapCoordinate

    q = parse_qs(urlparse(url).query)

    def _num(name, default=None):
        vals = q.get(name)
        if not vals:
            return default
        try:
            return float(vals[0])
        except (TypeError, ValueError):
            return default

    lat = _num("lat")
    lon = _num("lon")
    if lat is None or lon is None:
        raise ValueError(f"URL missing lat/lon: {url}")
    width = _num("width", 1.0)
    height = _num("height", 1.0)

    area = CoordinateArea(MapCoordinate(lat, lon), width, height)
    paths = [f"/points/{c}" for c in area.map_coordinates]
    paths += [f"/gridpoints/{gp}" for gp in area.gridpoints]
    # De-dupe while preserving order.
    seen = set()
    unique = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


# --- cache-file expiry -------------------------------------------------------

def _cache_file_for(path):
    from urllib.parse import urljoin, urlparse
    base = os.environ.get("NOAA_BASE_URL", "https://api.weather.gov")
    url = urljoin(base + "/", path.lstrip("/"))
    parse = urlparse(url)
    return os.path.join(CACHE_ROOT, parse.netloc, parse.path[1:], "data.json")


def expiry_of(path):
    """Return the UNIX timestamp when `path`'s cache file expires. A missing
    or unreadable/undated file returns 0.0 (i.e. due for refresh now)."""
    cache_file = _cache_file_for(path)
    try:
        with open(cache_file, "r") as f:
            body = json.loads(f.read())
        exp = datetime.strptime(
            body["headers"]["Expires"], "%a, %d %b %Y %H:%M:%S %Z"
        )
        return exp.timestamp()
    except Exception:
        return 0.0


# --- Scheduler ---------------------------------------------------------------

class _Scheduler:
    def __init__(self):
        self._cond = threading.Condition()
        self._heap = []                 # (expiry_ts, path)
        self._url_paths = {}            # url -> set(paths)
        self._stop = False

    # ---- public (manager) API ----
    def watch(self, url):
        """Start keeping `url` hot. Resolves its paths, schedules each by
        expiry, persists the watch-list, and wakes the loop."""
        try:
            paths = paths_for_url(url)
        except Exception as e:
            logger.warning("hotcache: cannot resolve %s: %s", url, e)
            return json.dumps({"ok": False, "error": str(e)})
        with self._cond:
            self._url_paths[url] = set(paths)
            now = time.time()
            for p in paths:
                heapq.heappush(self._heap, (expiry_of(p) or now, p))
            self._persist_locked()
            self._cond.notify()
        logger.warning("hotcache: watching %s (%d paths)", url, len(paths))
        return json.dumps({"ok": True, "paths": len(paths)})

    def unwatch(self, url):
        with self._cond:
            existed = self._url_paths.pop(url, None) is not None
            # Heap entries for now-orphaned paths are dropped lazily when popped
            # (see _reschedule / loop), so no need to rebuild the heap here.
            self._persist_locked()
            self._cond.notify()
        return json.dumps({"ok": existed})

    def list_watched(self):
        with self._cond:
            return json.dumps({
                "urls": sorted(self._url_paths.keys()),
                "paths": sum(len(v) for v in self._url_paths.values()),
                "scheduled": len(self._heap),
            })

    # ---- internals ----
    def _is_watched_path(self, path):
        for paths in self._url_paths.values():
            if path in paths:
                return True
        return False

    def _persist_locked(self):
        try:
            tmp = WATCHLIST_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump(sorted(self._url_paths.keys()), f)
            os.replace(tmp, WATCHLIST_PATH)
        except Exception:
            logger.warning("hotcache: failed to persist watch-list", exc_info=True)

    def _load_persisted(self):
        try:
            with open(WATCHLIST_PATH, "r") as f:
                urls = json.load(f)
        except Exception:
            return
        for url in urls:
            self.watch(url)

    def run(self):
        logger.warning("hotcache scheduler started")
        self._load_persisted()
        while True:
            due_paths = []
            with self._cond:
                if self._stop:
                    return
                now = time.time()
                # Pop everything whose (expiry - lead) has arrived.
                while self._heap and self._heap[0][0] - LEAD_SECONDS <= now:
                    _exp, path = heapq.heappop(self._heap)
                    if self._is_watched_path(path):
                        due_paths.append(path)
                    # else: orphaned by unwatch -> drop silently.

                if not due_paths:
                    # Sleep until the next entry's lead time, bounded.
                    if self._heap:
                        wait = self._heap[0][0] - LEAD_SECONDS - now
                    else:
                        wait = MAX_SLEEP
                    wait = max(MIN_SLEEP, min(wait, MAX_SLEEP))
                    self._cond.wait(timeout=wait)
                    continue

            # Outside the lock: ask the writer to refresh the due paths. The
            # writer only refetches those actually expired/missing. De-dupe.
            due_paths = list(dict.fromkeys(due_paths))
            try:
                cachewriter.ensure(due_paths)
            except Exception:
                logger.warning("hotcache: writer unavailable; will retry", exc_info=True)
                # Re-schedule shortly so we try again.
                with self._cond:
                    retry_at = time.time() + 30
                    for p in due_paths:
                        if self._is_watched_path(p):
                            heapq.heappush(self._heap, (retry_at + LEAD_SECONDS, p))
                    self._cond.wait(timeout=5)
                continue

            # Reschedule each refreshed path by its NEW expiry.
            with self._cond:
                now = time.time()
                for p in due_paths:
                    if not self._is_watched_path(p):
                        continue
                    new_exp = expiry_of(p)
                    if not new_exp:
                        # Still missing (e.g. NOAA error): retry soon.
                        new_exp = now + 60 + LEAD_SECONDS
                    heapq.heappush(self._heap, (new_exp, p))


# --- Manager wiring ----------------------------------------------------------

class _HotCacheManager(BaseManager):
    pass


def run_scheduler(ready_event=None):
    """Entry point for the scheduler process."""
    sched = _Scheduler()

    _HotCacheManager.register("watch", callable=sched.watch)
    _HotCacheManager.register("unwatch", callable=sched.unwatch)
    _HotCacheManager.register("list_watched", callable=sched.list_watched)

    mgr = _HotCacheManager(address=(SCHED_HOST, SCHED_PORT), authkey=SCHED_AUTHKEY)
    server = mgr.get_server()
    logger.warning("hotcache manager listening on %s:%s", SCHED_HOST, SCHED_PORT)

    # Run the scheduler loop in a background thread; serve the manager in the
    # foreground.
    t = threading.Thread(target=sched.run, daemon=True)
    t.start()
    if ready_event is not None:
        ready_event.set()
    server.serve_forever()


# --- Client side (used by the admin API in web workers) ----------------------

_client = None
_client_failed = False
_client_lock = threading.Lock()


class _HotCacheClient(BaseManager):
    pass


_HotCacheClient.register("watch")
_HotCacheClient.register("unwatch")
_HotCacheClient.register("list_watched")


class SchedulerUnavailable(Exception):
    pass


def _get_client():
    global _client, _client_failed
    if _client is not None:
        return _client
    if _client_failed:
        raise SchedulerUnavailable()
    with _client_lock:
        if _client is not None:
            return _client
        if _client_failed:
            raise SchedulerUnavailable()
        c = _HotCacheClient(address=(SCHED_HOST, SCHED_PORT), authkey=SCHED_AUTHKEY)
        try:
            c.connect()
        except Exception as e:
            _client_failed = True
            raise SchedulerUnavailable() from e
        _client = c
        return _client


def _value(proxy):
    try:
        return proxy._getvalue()
    except AttributeError:
        return proxy


def watch(url):
    return json.loads(_value(_get_client().watch(url)))


def unwatch(url):
    return json.loads(_value(_get_client().unwatch(url)))


def list_watched():
    return json.loads(_value(_get_client().list_watched()))
