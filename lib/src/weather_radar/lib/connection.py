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

    resp = session.get(path)
    if resp.status_code == 200:
        return resp

    # Rate limited: NOAA documents that throttling typically clears within a
    # few seconds. Honor an explicit Retry-After header when present, else
    # fall back to exponential backoff.
    if resp.status_code == 429:
        retry_after = resp.headers.get("Retry-After")
        try:
            wait = float(retry_after)
        except (TypeError, ValueError):
            wait = _retry_time(attempt)
        logger.warning(f"HTTP 429 rate limited; backing off {wait}s: {path}")
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

    # Not cached. Coalesce concurrent requests for the *same* URL behind a
    # per-URL lock so only one thread hits NOAA; the others wait and then read
    # the file the first thread wrote. Different URLs are not blocked.
    url_lock = _url_locks.get(path.url)
    with url_lock:
        # Re-check: another thread may have fetched it while we waited.
        if os.path.isfile(path.tempfile):
            try:
                with open(path.tempfile, "r") as f:
                    return json.loads(f.read())
            except Exception:
                return bad_cache
        return get_and_write(path, session)


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
