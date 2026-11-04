import json
import logging
import os
import time
import threading
from datetime import datetime
from urllib.parse import urljoin, urlparse

import requests
import multiprocessing


logger = logging.getLogger(__name__)
lock = multiprocessing.Lock()


class _KeyedLocks:
    """Hands out a distinct lock per key (here, per URL) so that concurrent
    requests for the *same* resource are coalesced (only one fetch happens,
    the rest wait and then read the freshly written cache), while requests
    for *different* resources still proceed in parallel."""

    def __init__(self):
        self._registry_lock = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}

    def get(self, key: str) -> threading.Lock:
        with self._registry_lock:
            existing = self._locks.get(key)
            if existing is None:
                existing = threading.Lock()
                self._locks[key] = existing
            return existing


_url_locks = _KeyedLocks()


class NoCoverageError(Exception):
    """Raised when NOAA reports it has no data for a requested point (HTTP 404
    /problems/InvalidPoint) — e.g. a coordinate outside US coverage (Canada,
    Mexico, open ocean). This is PERMANENT, not transient: callers should skip
    the point rather than retry it."""


def _retry_time(attempt):
    """Exponential backoff with a cap, in seconds, for the given 0-based
    attempt number: 1, 2, 4, 8, ... capped at 16s."""
    return min(2 ** attempt, 16)


def retry_get(session, path, retries=6, resp=None, attempt=0):
    logger.info(f"HTTP ATTEMPT {attempt + 1}: {path}")
    if not retries:
        if resp is not None:
            resp.raise_for_status()
        raise RuntimeError("NWS unresponsive after multiple tries")

    # Transient network failures (DNS hiccup, connection refused/reset, read
    # timeout) raise BEFORE a response exists. These are retryable — a brief
    # blip shouldn't permanently drop a path from the cache — so back off and
    # retry rather than letting the exception escape.
    try:
        resp = session.get(path, timeout=30)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        logger.warning(
            "HTTP transient error (%s); backing off %ss: %s",
            type(e).__name__, _retry_time(attempt), path,
        )
        time.sleep(_retry_time(attempt))
        return retry_get(session, path, retries - 1, None, attempt + 1)

    if resp.status_code == 200:
        return resp

    # Permanent "no coverage" for this point: NOAA returns 404 with
    # /problems/InvalidPoint for coordinates it can't serve (outside the US,
    # ocean, etc.). Retrying never helps, so fail fast and let callers skip the
    # point instead of burning the full backoff ladder (~47s) on every one.
    if resp.status_code == 404:
        raise NoCoverageError(path)

    # Rate limited: NOAA documents that throttling typically clears within a
    # few seconds. It usually signals this with 429 (+ Retry-After), but under
    # a sustained burst it may instead return 403 Forbidden for the same
    # reason, so treat both the same. Honor an explicit Retry-After header when
    # present, else fall back to exponential backoff.
    if resp.status_code in (429, 403):
        retry_after = resp.headers.get("Retry-After")
        try:
            wait = float(retry_after)
        except (TypeError, ValueError):
            wait = _retry_time(attempt)
        logger.warning(f"HTTP {resp.status_code} throttled; backing off {wait}s: {path}")
        time.sleep(wait)
        return retry_get(session, path, retries - 1, resp, attempt + 1)

    # Other transient errors: exponential backoff.
    time.sleep(_retry_time(attempt))
    return retry_get(session, path, retries - 1, resp, attempt + 1)


def get_and_write(path, session):
    os.makedirs(path.tempdir, exist_ok=True)
    resp = retry_get(session, path.url)
    resp = {
        "headers": dict(resp.headers),
        "json": resp.json()
    }
    with open(path.tempfile, "w") as f:
        f.write(json.dumps(resp))
    return resp


bad_cache = object()

def get_from_external_source(path, session):
    # Fast path: file already cached on disk.
    if os.path.isfile(path.tempfile):
        try:
            with open(path.tempfile, "r") as f:
                return json.loads(f.read())
        except Exception:
            return bad_cache

    # Not cached. Preferred path: delegate the fetch+write to the single
    # cache-writer process and wait (model A) for it to atomically publish the
    # file, then read it. The writer coalesces duplicate URLs so concurrent
    # misses for the same path cause a single NOAA request.
    #
    # If the writer process isn't reachable (e.g. tests, scripts, REPL, or the
    # server hasn't spawned it yet), fall back to fetching inline so callers
    # still work. The inline path uses the same per-URL coalescing lock.
    try:
        from .cachewriter import ensure as _ensure_cached
        _ensure_cached([path.path])
    except Exception:
        logger.info(
            "cache writer unavailable for %s; fetching inline", path.path
        )
        return _inline_fetch(path, session)

    if os.path.isfile(path.tempfile):
        try:
            with open(path.tempfile, "r") as f:
                return json.loads(f.read())
        except Exception:
            return bad_cache
    # Writer ran but produced nothing. This is either a transient miss (NOAA
    # error, timeout) or a PERMANENT no-coverage 404 (the writer can't signal
    # which — it just doesn't write a file). Do one inline fetch to classify:
    # a NoCoverageError propagates (so callers skip the point); anything else
    # is treated as a transient miss. Without this, a 404 would make
    # get_with_expiry recurse forever (bad_cache -> retry -> bad_cache -> ...).
    return _inline_fetch(path, session)


def _inline_fetch(path, session):
    """Fetch + write a single path in-process (fallback when the cache-writer
    process isn't available, or to classify a writer miss). Coalesces
    concurrent fetches of the same URL. Propagates NoCoverageError (permanent
    404) so callers can skip the point instead of retrying."""
    url_lock = _url_locks.get(path.url)
    with url_lock:
        if os.path.isfile(path.tempfile):
            try:
                with open(path.tempfile, "r") as f:
                    return json.loads(f.read())
            except Exception:
                return bad_cache
        try:
            return get_and_write(path, session)
        except NoCoverageError:
            raise  # permanent: let it propagate so the point is skipped
        except Exception:
            logger.warning("inline fetch failed for %s", path.path, exc_info=True)
            return bad_cache


class PathObj:
    temproot = os.environ.get("CACHE_DIR", "/tmp")

    def __init__(self, path):
        conn = NOAAConnection()
        self.path = path
        path = urljoin(conn.url, path)
        self.url = path
        parse = urlparse(path)
        self.tempdir = os.path.join(
            self.temproot, parse.netloc, parse.path[1:]
        )
        self.tempfile = os.path.join(self.tempdir, "data.json")


class NOAAConnection:
    _instance = None

    def __new__(cls, *_, **__):
        instance = cls._instance
        if not instance:
            instance = object.__new__(cls)
            cls._instance = instance
        return instance


    def __init__(self, url="https://api.weather.gov"):
        self.session = requests.session()
        self.session.headers["User-Agent"] = "Michael Green, self@michaelgreen.dev"
        self.url = url
        self._cache = {}

    def get(self, path):
        _, body = self.get_with_expiry(path)
        return body

    def get_with_expiry(self, path):
        """Like `get`, but also returns the `Expires` datetime for the
        resource so callers can cache derived artifacts (e.g. fitted
        splines) with the same lifetime as the underlying data."""
        path = PathObj(path)
        resp = self._cache.get(path.url, None)

        if not resp:
            resp = get_from_external_source(path, self.session)
            if resp is bad_cache:
                try:
                    self._cache.pop(path.url)
                except KeyError:
                    pass
                try:
                    os.unlink(path.tempfile)
                except FileNotFoundError:
                    pass
                return self.get_with_expiry(path.path)

            resp = (
                datetime.strptime(
                    resp["headers"]["Expires"],
                    "%a, %d %b %Y %H:%M:%S %Z"
                ),
                resp["json"]
            )

            self._cache[path.url] = resp

        expiry, body = resp

        is_expired = expiry < datetime.now()

        if is_expired:
            try:
                self._cache.pop(path.url)
            except KeyError:
                pass
            try:
                os.unlink(path.tempfile)
            except FileNotFoundError:
                pass
            return self.get_with_expiry(path.path)
        return expiry, body
