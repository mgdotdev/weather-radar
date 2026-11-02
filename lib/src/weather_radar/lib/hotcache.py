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


# --- app URL -> frame spec (for precomputing animation frames) ---------------

# NOAA layer name (kebab in URL) -> ensemble kind, mirroring the server's
# MODEL_CATALOG. Accumulation layers are cumulative totals; everything else is
# an instantaneous observation. Used to pick the right ensemble when warming
# frames. Unknown models default to observation.
_ACCUMULATION_MODELS = {
    "quantitative-precipitation", "snowfall-amount", "ice-accumulation",
}


def frame_spec_for_url(url):
    """Extract the parameters needed to precompute a map URL's animation
    frames: model, kind, lat/lon/width/height, and dt. Returns None for URLs
    that aren't map URLs (e.g. /graph/), which have no frames to warm."""
    from urllib.parse import urlparse, parse_qs
    import re

    parsed = urlparse(url)
    m = re.search(r"/map/models/([^/]+)/?", parsed.path)
    if not m:
        return None
    model = m.group(1)
    q = parse_qs(parsed.query)

    def _num(name, default=None):
        vals = q.get(name)
        if not vals:
            return default
        try:
            return float(vals[0])
        except (TypeError, ValueError):
            return default

    lat, lon = _num("lat"), _num("lon")
    if lat is None or lon is None:
        return None
    kind = (q.get("kind", [None])[0]
            or ("accumulation" if model in _ACCUMULATION_MODELS else "observation"))
    dt = _num("dt", 3600.0)
    return {
        "model": model,
        "kind": kind,
        "lat": lat,
        "lon": lon,
        "width": _num("width", 1.0),
        "height": _num("height", 1.0),
        "dt": int(dt) if dt else 3600,
    }


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
        self._url_spec = {}             # url -> frame spec (map URLs only)
        self._path_urls = {}            # path -> set(urls) that depend on it
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
        spec = None
        try:
            spec = frame_spec_for_url(url)
        except Exception:
            logger.warning("hotcache: cannot parse frame spec for %s", url, exc_info=True)
        with self._cond:
            self._url_paths[url] = set(paths)
            if spec is not None:
                self._url_spec[url] = spec
            now = time.time()
            for p in paths:
                self._path_urls.setdefault(p, set()).add(url)
                heapq.heappush(self._heap, (expiry_of(p) or now, p))
            self._persist_locked()
            self._cond.notify()
        logger.warning("hotcache: watching %s (%d paths)", url, len(paths))
        # Warm the computed frames for this URL now (NOAA data is scheduled
        # above; frames are derived from it). Done in a background thread so the
        # watch() call returns promptly.
        if spec is not None:
            threading.Thread(
                target=self._warm_frames, args=(url, spec), daemon=True
            ).start()
        return json.dumps({"ok": True, "paths": len(paths)})

    def unwatch(self, url):
        with self._cond:
            paths = self._url_paths.pop(url, None)
            existed = paths is not None
            self._url_spec.pop(url, None)
            if paths:
                for p in paths:
                    urls = self._path_urls.get(p)
                    if urls:
                        urls.discard(url)
                        if not urls:
                            self._path_urls.pop(p, None)
            # Heap entries for now-orphaned paths are dropped lazily when popped
            # (see loop), so no need to rebuild the heap here.
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

    def _warm_frames(self, url, spec):
        """Precompute the animation frames for a watched map URL across its
        forecast window, so navigating there finds every frame already cached.

        Also the invalidation hook: because NOAA data may have just been
        refreshed, any previously cached frames for this area were interpolated
        from now-stale data. The writer's compute_frame version-stamps frames
        by the current gridpoint data, so recomputing here overwrites them with
        current-data frames; we additionally clear the area dir first to reclaim
        the superseded files. Runs in its own thread; best-effort."""
        from datetime import datetime
        try:
            from . import framecache
            from .area import CoordinateArea, MapCoordinate
            from .models.accumulation import AccumulationEnsemble
            from .models.observation import ObservationEnsemble
        except Exception:
            logger.warning("hotcache: frame warming unavailable (imports)", exc_info=True)
            return

        area = CoordinateArea(MapCoordinate(spec["lat"], spec["lon"]),
                              spec["width"], spec["height"])
        kinds = {"accumulation": AccumulationEnsemble, "observation": ObservationEnsemble}
        ensemble_class = kinds.get(spec["kind"], AccumulationEnsemble)
        try:
            vr = ensemble_class(area, spec["model"]).value_range()
        except Exception:
            logger.warning("hotcache: value_range failed for %s", url, exc_info=True)
            return
        if not vr or not vr.get("start") or not vr.get("end"):
            return  # no valid forecast window -> nothing to warm

        start = int(datetime.fromisoformat(vr["start"]).timestamp())
        end = int(datetime.fromisoformat(vr["end"]).timestamp())
        dt = int(spec["dt"]) or 3600

        # Drop superseded frames for this area (defensive; version check already
        # prevents serving them).
        akey = framecache.area_key(
            spec["model"], spec["kind"], spec["lat"], spec["lon"],
            spec["width"], spec["height"],
        )
        try:
            framecache.clear_area(akey)
        except Exception:
            pass

        # Enqueue each frame to the writer (single-writer invariant: the writer
        # computes + caches; we're just a client, like a web worker). Cap to a
        # sane number of frames.
        count = 0
        t = start
        warmed = 0
        while t <= end and count < 5000:
            params = {
                "model": spec["model"], "kind": spec["kind"],
                "lat": spec["lat"], "lon": spec["lon"],
                "width": spec["width"], "height": spec["height"],
                "time": t,
            }
            try:
                cachewriter.frame(params)
                warmed += 1
            except cachewriter.WriterUnavailable:
                logger.warning("hotcache: writer unavailable while warming frames")
                return
            except Exception:
                logger.warning("hotcache: frame warm failed at %s", t, exc_info=True)
            count += 1
            t += dt
        logger.warning("hotcache: warmed %d frames for %s", warmed, url)

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

            # Reschedule each refreshed path by its NEW expiry, and collect the
            # URLs whose GRIDPOINT data changed so we can recompute their frames
            # (any interpolation built from the old data is now invalid).
            affected_urls = set()
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
                    # Only gridpoint changes affect computed frames; /points
                    # just resolve the grid geometry (stable for the area).
                    if p.startswith("/gridpoints/"):
                        for u in self._path_urls.get(p, ()):
                            if u in self._url_spec:
                                affected_urls.add(u)
                specs = {u: self._url_spec[u] for u in affected_urls}

            # Recompute frames for affected URLs (outside the lock). The writer
            # version-stamps frames by current data, so this overwrites stale
            # ones. Threaded so a slow warm doesn't stall the scheduler loop.
            for u, spec in specs.items():
                logger.warning("hotcache: NOAA data changed, rewarming frames for %s", u)
                threading.Thread(
                    target=self._warm_frames, args=(u, spec), daemon=True
                ).start()


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
