"""Serve the preview tests' API calls from the REAL sample-data module
(server/fixtures.py), the way the server does for `fixture=1` — so the browser
tests exercise the same responses a user gets, for every layer, not hand-written
stubs."""
import gzip
from urllib.parse import urlparse, parse_qs

from weather_radar.server import fixtures


def serve_api(route, seen=None):
    """Fulfil a Playwright route from fixtures.respond. `seen` (a list) records
    (path, query dict) for every API call, so tests can assert what the page asked
    for — e.g. that a settings change made NO data request."""
    u = urlparse(route.request.url)
    q = {k: v[0] for k, v in parse_qs(u.query, keep_blank_values=True).items()}
    if seen is not None:
        seen.append((u.path, q))
    r = fixtures.respond(u.path, q)
    if r is None:
        return route.fulfill(status=404, body="")
    status, headers, body, media = r
    headers = dict(headers)
    if headers.pop("Content-Encoding", None) == "gzip":
        # (Chromium does not inflate a response that a test harness fulfils itself;
        # the real server's gzip is inflated by the browser as usual.)
        body = gzip.decompress(body)
    return route.fulfill(status=status, headers=headers, body=body, content_type=media)


def map_html(url):
    """What the real /map/models/{model}/ route serves for this URL (any layer)."""
    import weather_radar.server.app as app
    u = urlparse(url)
    q = parse_qs(u.query)
    f = lambda k, d: float(q[k][0]) if k in q else d
    model = u.path.strip("/").split("/")[2]
    return app.map(model, width=f("width", 293.8), height=f("height", 128.76),
                   lat=f("lat", 48.123221), lon=f("lon", -116.630892),
                   dt=int(f("dt", 3600)), speed=f("speed", 1.0))
