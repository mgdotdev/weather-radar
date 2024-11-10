import json
import os
import time
from datetime import datetime
from urllib.parse import urljoin, urlparse

import requests


def retry_get(session, path, retries=5):
    if not retries:
        raise RuntimeError(f"NWS unresponsive after 5 tries")
    time.sleep(5-retries)
    resp = session.get(path)
    if resp.status_code != 200:
        return retry_get(session, path, retries-1)
    return resp


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
        path = urljoin(self.url, path)
        parse = urlparse(path)
        tempdir = os.path.join(
            "/tmp", parse.netloc, parse.path[1:]
        )
        tempfile = os.path.join(tempdir, "data.json")
        resp = self._cache.get(path, None)

        if not resp:
            if not os.path.isfile(tempfile):
                os.makedirs(tempdir, exist_ok=True)
                resp = retry_get(self.session, path)
                resp = {
                    "headers": dict(resp.headers),
                    "json": resp.json()
                }
                with open(tempfile, "w") as f:
                    f.write(json.dumps(resp))

            else:
                with open(tempfile, "r") as f:
                    resp = json.loads(f.read())
            resp = (
                datetime.strptime(
                    resp["headers"]["Expires"],
                    "%a, %d %b %Y %H:%M:%S %Z"
                ),
                resp["json"]
            )

            self._cache[path] = resp

        expiry, body = resp

        is_expired = expiry < datetime.now()

        if is_expired:
            self._cache.pop(path)
            os.unlink(tempfile)
            return self.get(path)
        return body
