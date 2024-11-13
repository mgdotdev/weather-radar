import json
import logging
import os
import time
from datetime import datetime
from urllib.parse import urljoin, urlparse

import requests
import multiprocessing


logger = logging.getLogger(__name__)
lock = multiprocessing.Lock()


def _retry_time(retries):
    if retries >= 4:
        return 0
    return 4 - retries


def retry_get(session, path, retries=10, resp=None):
    logger.info(f"HTTP ATTEMPT {5-retries+1}: {path}")
    if not retries:
        if resp:
            resp.raise_for_status()
        else:
            raise RuntimeError(f"NWS unresponsive after 5 tries")
    time.sleep(_retry_time(retries))
    resp = session.get(path)
    if resp.status_code != 200:
        return retry_get(session, path, retries-1, resp)
    return resp


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
    with lock:
        if not os.path.isfile(path.tempfile):
            return get_and_write(path, session)

    try:
        with open(path.tempfile, "r") as f:
            resp = json.loads(f.read())
    except Exception:
        return bad_cache
    return resp


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
                return self.get(path.path)

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
            return self.get(path.path)
        return body
