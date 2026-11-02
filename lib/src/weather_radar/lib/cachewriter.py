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


# --- Frame-compute worker pool ----------------------------------------------
# Frame compute (spline evaluation) is CPU-bound, so the GIL serializes it
# within one process. Real parallelism needs separate PROCESSES. The manager
# keeps its single-authority role but dispatches frame compute to a pool of
# worker processes, and DEDUPLICATES concurrent requests for the same frame so
# a duplicate attaches to the in-flight job instead of recomputing.

FRAME_WORKERS = int(os.environ.get("FRAME_WORKERS", str(max(2, (os.cpu_count() or 4)))))


def _set_pdeathsig():
    """Ask the kernel to send SIGTERM to this process when its parent dies, so
    non-daemon workers (and the non-daemon writer) can't be orphaned if their
    parent goes away. Linux-only; a no-op elsewhere."""
    try:
        import ctypes, signal
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except Exception:
        pass


def _compute_frame_payload(params_json):
    """Compute one frame and write it to the per-frame cache, returning the
    FeatureCollection as a JSON string. Runs inside a frame-worker PROCESS, so
    it's a module-level pure function (importable under the 'spawn' start
    method) with no dependence on manager state."""
    params = json.loads(params_json)
    from . import framecache
    from .area import MapCoordinate, CoordinateArea
    from .models.accumulation import AccumulationEnsemble
    from .models.observation import ObservationEnsemble

    akey = framecache.area_key(
        params["model"], params["kind"], params["lat"], params["lon"],
        params["width"], params["height"],
    )
    epoch = int(params["time"])
    coord = MapCoordinate(params["lat"], params["lon"])
    area = CoordinateArea(coord, params["width"], params["height"])
    grid_paths = [f"/gridpoints/{gp}" for gp in area.gridpoints]
    fresh = framecache.area_is_fresh(grid_paths)
    # Version-stamp frames by the current gridpoint data so a frame built from
    # superseded data is detected as stale even once the (new) gridpoints are
    # themselves fresh — NOAA data changing invalidates interpolation from it.
    version = framecache.data_version(grid_paths)

    if fresh:
        cached = framecache.read_frame(akey, epoch, version=version)
        if cached is not None:
            return json.dumps(cached)

    ENSEMBLE_KINDS = {
        "accumulation": AccumulationEnsemble,
        "observation": ObservationEnsemble,
    }
    ensemble_class = ENSEMBLE_KINDS.get(params["kind"], AccumulationEnsemble)
    ensemble = ensemble_class(area, params["model"])
    fc = ensemble.predict(datetime.fromtimestamp(epoch), verbose=True)
    if fresh:
        try:
            framecache.write_frame(akey, epoch, fc, version=version)
        except Exception:
            logger.warning("failed to cache frame %s", epoch, exc_info=True)
    return json.dumps(fc)


def _frame_cache_key(params):
    """Dedup key for a frame request: a stable function of the params that
    determine the output (area + timestamp; dt-independent). Self-contained so
    it works regardless of how the module was imported."""
    import hashlib
    raw = "|".join(str(params[k]) for k in (
        "model", "kind", "lat", "lon", "width", "height", "time"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _frame_worker_loop(req_q, res_q):
    """Frame-worker process entry point: pull (job_id, params_json), compute,
    push (job_id, ok, payload). Exits on a None sentinel."""
    _set_pdeathsig()
    while True:
        item = req_q.get()
        if item is None:
            return
        job_id, params_json = item
        try:
            res_q.put((job_id, True, _compute_frame_payload(params_json)))
        except Exception as e:
            res_q.put((job_id, False, repr(e)))


class _PendingJob:
    """A frame compute in flight. Duplicate requests block on `event`; the
    result dispatcher fills ok/payload and sets it, so all waiters return the
    same result without recomputing."""
    __slots__ = ("event", "ok", "payload")

    def __init__(self):
        self.event = threading.Event()
        self.ok = False
        self.payload = None


class _FramePool:
    """Pool of frame-worker processes with request dedup.

    submit(params_json) (called by a manager RPC thread) blocks until the frame
    is computed. If an identical frame is already in flight, it attaches to the
    existing job (no duplicate compute). A background dispatcher thread drains
    worker results and wakes the matching job's waiters.
    """

    def __init__(self, n_workers, ctx=None):
        import multiprocessing as mp
        # Start method, in order of preference:
        #   forkserver - forks workers from a clean, single-threaded server
        #                process: no fork-with-threads deadlock AND no fragile
        #                re-import of our module (unlike spawn under --reload).
        #   spawn      - clean interpreter, but re-imports the module (can fail
        #                to bootstrap under some launchers).
        #   fork       - last resort (tests inject this).
        if ctx is None:
            for method in ("forkserver", "spawn"):
                try:
                    ctx = mp.get_context(method)
                    break
                except ValueError:
                    continue
            else:
                ctx = mp.get_context()
        self._req_q = ctx.Queue()
        self._res_q = ctx.Queue()
        self._procs = []
        for _ in range(n_workers):
            p = ctx.Process(target=_frame_worker_loop,
                            args=(self._req_q, self._res_q), daemon=False)
            p.start()
            self._procs.append(p)
        self._lock = threading.Lock()
        self._pending = {}        # key -> _PendingJob
        self._job_by_id = {}      # job_id -> key
        import itertools
        self._ids = itertools.count(1)
        self._healthy = True      # cleared if the pool proves unresponsive
        threading.Thread(target=self._dispatch_loop, daemon=True).start()
        logger.warning("frame pool started with %d %s worker process(es)",
                       n_workers, ctx.get_start_method())

    def _dispatch_loop(self):
        while True:
            job_id, ok, payload = self._res_q.get()
            with self._lock:
                key = self._job_by_id.pop(job_id, None)
                job = self._pending.pop(key, None) if key is not None else None
            if job is not None:
                job.ok, job.payload = ok, payload
                job.event.set()

    def submit(self, params_json, timeout=20.0):
        # If the pool has already proven unresponsive, skip it entirely and
        # compute inline — don't make every request eat the timeout.
        if not self._healthy:
            return _compute_frame_payload(params_json)

        key = _frame_cache_key(json.loads(params_json))
        with self._lock:
            job = self._pending.get(key)
            if job is not None:
                existing = True            # dedup onto the in-flight compute
            else:
                job = _PendingJob()
                self._pending[key] = job
                job_id = next(self._ids)
                self._job_by_id[job_id] = key
                existing = False
        if not existing:
            self._req_q.put((job_id, params_json))
        if job.event.wait(timeout=timeout):
            if not job.ok:
                raise RuntimeError(f"frame compute failed: {job.payload}")
            return job.payload
        # Timed out: the worker pool isn't delivering results (broken IPC under
        # a reloader, crashed-on-import workers, etc.). Mark it unhealthy so
        # subsequent requests go straight to the inline path, and compute this
        # one inline now. The frame is still cached on disk by the payload, so
        # the system stays correct — just without the extra process parallelism.
        logger.warning(
            "frame pool unresponsive after %ss; falling back to inline compute",
            timeout,
        )
        self._healthy = False
        with self._lock:
            self._pending.pop(key, None)
        return _compute_frame_payload(params_json)

    def shutdown(self):
        for _ in self._procs:
            try:
                self._req_q.put(None)
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


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
        # Per-frame-key locks so two concurrent requests for the same frame
        # compute it once (the second waits, then reads the cached result).
        self._frame_locks = {}
        self._frame_registry_lock = threading.Lock()
        import itertools
        self._job_counter = itertools.count(1)
        # Worker-process pool for CPU-bound frame compute (created lazily so a
        # NOAA-only deployment or tests don't spawn workers they never use).
        self._frame_pool = None
        self._frame_pool_lock = threading.Lock()

    def _get_frame_pool(self):
        if self._frame_pool is None:
            with self._frame_pool_lock:
                if self._frame_pool is None:
                    self._frame_pool = _FramePool(FRAME_WORKERS)
        return self._frame_pool

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

    # --- frame computation (coordinated worker pool) ---
    # The manager stays the single authority but dispatches CPU-bound frame
    # compute to a pool of worker processes (real parallelism), deduplicating
    # concurrent requests for the same frame. The authoritative cache-hit /
    # freshness / version logic lives in _compute_frame_payload (one source of
    # truth, runs in the worker).

    def compute_frame(self, params_json):
        """Return the ensemble FeatureCollection for one frame as a JSON
        string, computed by the worker pool with request dedup."""
        return self._get_frame_pool().submit(params_json)


# --- Manager wiring ----------------------------------------------------------

class _CacheManager(BaseManager):
    pass


def run_writer(ready_event=None):
    """Entry point for the writer process: start the manager server and serve
    forever. If `ready_event` is given, set it once the server is listening.

    This process lazily spawns a pool of frame-worker child processes, so it
    must be non-daemon (a daemon process can't have children). It sets
    PR_SET_PDEATHSIG so it (and its workers, which do the same) are reaped by
    the kernel if the parent dies."""
    _set_pdeathsig()
    writer = _Writer()

    _CacheManager.register("ensure", callable=writer.ensure)
    _CacheManager.register("start_job", callable=writer.start_job)
    _CacheManager.register("poll_job", callable=writer.poll_job)
    _CacheManager.register("compute_frame", callable=writer.compute_frame)

    mgr = _CacheManager(
        address=(CACHE_WRITER_HOST, CACHE_WRITER_PORT),
        authkey=CACHE_WRITER_AUTHKEY,
    )
    server = mgr.get_server()
    logger.warning("cache writer listening on %s:%s", CACHE_WRITER_HOST, CACHE_WRITER_PORT)

    # Clean up the frame-worker pool on exit so its queue semaphores aren't
    # leaked (otherwise you get the resource_tracker "leaked semaphore objects"
    # warning on shutdown). Handle both normal interpreter exit and a SIGTERM/
    # SIGINT (Ctrl-C / reload), then re-raise the default behavior.
    import atexit
    import signal

    def _cleanup():
        pool = getattr(writer, "_frame_pool", None)
        if pool is not None:
            try:
                pool.shutdown()
            except Exception:
                pass

    atexit.register(_cleanup)

    def _on_signal(signum, _frame):
        _cleanup()
        # Restore default and re-raise so the process actually terminates.
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for _sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(_sig, _on_signal)
        except Exception:
            pass

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
_CacheClient.register("compute_frame")


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


def frame(params):
    """Return the computed ensemble FeatureCollection (dict) for one frame,
    produced BY THE WRITER PROCESS (single-writer invariant: the writer is the
    only process that computes + writes frame cache files). `params` is a dict
    with keys model, kind, lat, lon, width, height, time (epoch seconds).

    The writer serves from its per-frame cache when the underlying NOAA data is
    fresh, else computes + caches it. Raises WriterUnavailable if the writer
    isn't reachable (caller may fall back to computing inline)."""
    client = _get_client()

    def _value(proxy):
        try:
            return proxy._getvalue()
        except AttributeError:
            return proxy

    return json.loads(_value(client.compute_frame(json.dumps(params))))
