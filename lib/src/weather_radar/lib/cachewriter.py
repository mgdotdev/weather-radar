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
import concurrent.futures
from datetime import datetime, timezone
from multiprocessing.managers import BaseManager

import requests


logger = logging.getLogger(__name__)

# Max simultaneous NOAA requests the writer makes. We respect 429/Retry-After
# (see _retry_get) so this can be generous — NOAA throttles us if we overdo it
# and we back off automatically. Kept in sync with area.NOAA_MAX_CONCURRENCY.
MAX_CONCURRENCY = int(os.environ.get("NOAA_MAX_CONCURRENCY", "48"))

# Address + auth for the manager socket. A TCP loopback address keeps this
# working across fork/spawn and is trivial to connect to from worker procs.
CACHE_WRITER_HOST = os.environ.get("CACHE_WRITER_HOST", "127.0.0.1")
CACHE_WRITER_PORT = int(os.environ.get("CACHE_WRITER_PORT", "8137"))
CACHE_WRITER_AUTHKEY = os.environ.get("CACHE_WRITER_AUTHKEY", "weather-radar-cache").encode()

NOAA_BASE_URL = os.environ.get("NOAA_BASE_URL", "https://api.weather.gov")
USER_AGENT = os.environ.get(
    "NOAA_USER_AGENT", "Michael Green, self@michaelgreen.dev"
)
CACHE_ROOT = os.environ.get("CACHE_DIR", "/tmp")


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
        self._session.headers["User-Agent"] = USER_AGENT
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENCY)
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
            try:
                resp = _retry_get(self._session, url)
                payload = {"headers": dict(resp.headers), "json": resp.json()}
                os.makedirs(d, exist_ok=True)
                # Atomic publish: write to a temp file in the same dir, then
                # os.replace() so readers only ever see a complete file.
                fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
                try:
                    with os.fdopen(fd, "w") as f:
                        json.dump(payload, f)
                    os.replace(tmp, cache_file)
                finally:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
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

    def _value(proxy):
        # Registered methods return an AutoProxy to the result; materialize
        # the real (JSON string) value.
        try:
            return proxy._getvalue()
        except AttributeError:
            return proxy

    started = json.loads(_value(client.start_job(todo)))
    job_id = started["job_id"]
    while True:
        resp = json.loads(_value(client.poll_job(job_id)))
        # Batched: a list of {done,total} items plus a terminal `done` flag.
        for item in resp.get("items", []):
            yield {"done": item["done"], "total": item["total"]}
        if resp.get("done"):
            break

