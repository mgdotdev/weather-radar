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
import concurrent.futures
from datetime import datetime
from multiprocessing.managers import BaseManager

import requests


logger = logging.getLogger(__name__)

# Max simultaneous NOAA requests the writer makes. We respect 429/Retry-After
# (see _retry_get) so this can be generous — NOAA throttles us if we overdo it
# and we back off automatically. Kept in sync with area.NOAA_MAX_CONCURRENCY.
MAX_CONCURRENCY = int(os.environ.get("NOAA_MAX_CONCURRENCY", "8"))

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


def _is_fresh(cache_file):
    """True if the cache file exists and its stored Expires is still in the
    future. A missing/!unreadable/undated file is treated as not fresh."""
    try:
        with open(cache_file, "r") as f:
            body = json.loads(f.read())
        expires = body["headers"]["Expires"]
        exp = datetime.strptime(expires, "%a, %d %b %Y %H:%M:%S %Z")
        return exp >= datetime.now()
    except Exception:
        return False


def is_fresh(path):
    """True if the NOAA `path`'s cache file exists and is unexpired. A pure
    local disk read (no IPC), so callers can cheaply filter out paths that
    don't need the writer before enqueuing."""
    _url, _d, cache_file = _tempfile_for(path)
    return _is_fresh(cache_file)


def not_fresh(paths):
    """Return the subset of `paths` that are missing or expired (i.e. the ones
    that actually need to be (re)written)."""
    return [p for p in paths if not is_fresh(p)]


# --- HTTP (bounded, polite) --------------------------------------------------

def _retry_time(attempt):
    return min(2 ** attempt, 16)


def _retry_get(session, url, retries=6, attempt=0):
    """GET with 429/Retry-After handling + exponential backoff. Mirrors
    connection.retry_get but lives in the writer so all network writes share
    one implementation."""
    import time as _time
    resp = session.get(url)
    if resp.status_code == 200:
        return resp
    # Permanent no-coverage (NOAA 404 /problems/InvalidPoint): never retry.
    # Fail fast so the writer doesn't spend the full backoff ladder on points
    # it can never serve (e.g. coordinates outside US coverage). _fetch_one
    # catches this and simply writes nothing for the path.
    if resp.status_code == 404:
        raise RuntimeError(f"no coverage (404): {url}")
    if not retries:
        resp.raise_for_status()
        raise RuntimeError(f"NWS unresponsive: {url}")
    if resp.status_code == 429:
        ra = resp.headers.get("Retry-After")
        try:
            wait = float(ra)
        except (TypeError, ValueError):
            wait = _retry_time(attempt)
        logger.warning("cachewriter 429; backing off %ss: %s", wait, url)
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
        if _is_fresh(cache_file):
            return True
        lk = self._lock_for(url)
        with lk:
            # Re-check inside the lock (another thread may have just written).
            if _is_fresh(cache_file):
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
                return True
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

