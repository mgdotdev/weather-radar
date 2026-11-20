"""Real-browser smoke test of the map page (headless Chromium via Playwright).

`node --check` and string tests can't catch runtime errors in the page's script
(e.g. touching a not-yet-initialized `let` during startup, which once aborted
the whole load handler and broke the page). This loads the REAL page for a real
URL — pins, pin boxes, legend position — with only the API answered by stubs,
and fails on any uncaught page error. Skipped unless Playwright, its Chromium
and network access to the map library CDN are available.
"""
import datetime as dt
import json
import time
from urllib.parse import urlparse, parse_qs

import pytest

pw = pytest.importorskip("playwright.sync_api")
pytest.importorskip("jinja2")
pytest.importorskip("fastapi")

from weather_radar.lib import framing as F                    # noqa: E402
from weather_radar.lib.frametimes import frame_times          # noqa: E402

ORIGIN = "http://localhost:8000"
LAT, LON, W, H = 48.145506, -116.391332, 201.90, 183.69
# The URL from a real session (6 pins, pin boxes, legend position).
QUERY = ("lat=48.145506&lon=-116.391332&width=201.90&height=183.69&dt=900"
         "&pins=47.77877%2C-116.70459%2CHome%3B48.65516%2C-116.65461%2CBeehive+Lake%3B"
         "47.43378%2C-115.75997%2CStevens+Lake%3B48.18882%2C-116.08184%2CScotchman%27s%3B"
         "47.67359%2C-116.78221%2CCoeur+d%27Alene%3B48.27823%2C-116.55037%2CSandpoint"
         "&pinBoxes=%3B%3B-203%2C34%3B%3B-187%2C-27%3B-201%2C-9"
         "&legendLat=47.79725&legendLon=-115.67787")
MODEL = "air-quality-pm25"


def _lattice(n=14):
    return [[LON - 1.4 + 2.8 * i / (n - 1), LAT - 1.2 + 2.4 * j / (n - 1)]
            for j in range(n) for i in range(n)]


def _stub_api(route):
    """Answer the API the page calls for an AQI map (no real data needed)."""
    url = urlparse(route.request.url)
    q = parse_qs(url.query)
    if url.path.startswith("/api/models/aqi/range"):
        start = int(time.time()) // 3600 * 3600
        iso = lambda t: dt.datetime.fromtimestamp(t).isoformat()
        return route.fulfill(json={"min": 0, "max": 300, "uom": "AQI",
                                   "start": iso(start), "end": iso(start + 72 * 3600),
                                   "timeZone": None})
    if url.path.startswith("/api/models/aqi/stream"):
        g = lambda k: (q.get(k) or [None])[0]
        epochs = frame_times(float(g("start")) if g("start") else None,
                             float(g("end")) if g("end") else None,
                             int(g("dt") or 0), g("times"))
        coords = _lattice()
        scale, offset = F.choose_scale("AQI")
        body = F.record(F.encode_header("AQI", coords, scale, offset))
        for e in epochs:
            vals = [30 + (i * 7 + e // 900) % 90 for i in range(len(coords))]
            body += F.record(F.encode_frame(e, vals, scale, offset))
        # Sent UNCOMPRESSED: Chromium does not inflate Content-Encoding on a
        # response a test harness fulfills itself (the real server's gzip is
        # inflated by the browser as usual), so a gzip body would reach the
        # page's reader as raw gzip bytes and be silently ignored.
        return route.fulfill(body=body, headers={"Content-Type": "application/octet-stream"})
    return route.fulfill(status=404, body="")


@pytest.fixture(scope="module")
def page_html():
    import weather_radar.server.app as app
    return app.map(MODEL, width=W, height=H, lat=LAT, lon=LON, dt=900)


@pytest.fixture
def page(page_html):
    errors, console = [], []
    with pw.sync_playwright() as p:
        try:
            browser = p.chromium.launch(args=["--no-sandbox", "--use-gl=angle",
                                              "--use-angle=swiftshader", "--ignore-gpu-blocklist"])
        except Exception as e:                       # browser not installed
            pytest.skip(f"chromium unavailable: {e}")
        ctx = browser.new_context(viewport={"width": 1400, "height": 900})
        pg = ctx.new_page()
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.on("console", lambda m: console.append((m.type, m.text)))

        def handle(route):
            path = urlparse(route.request.url).path
            if path.startswith(f"/map/models/{MODEL}"):
                return route.fulfill(body=page_html, content_type="text/html; charset=utf-8")
            return _stub_api(route)
        ctx.route(f"{ORIGIN}/**", handle)
        try:
            pg.goto(f"{ORIGIN}/map/models/{MODEL}/?{QUERY}", wait_until="load", timeout=45000)
            pg.wait_for_function("typeof maplibregl !== 'undefined'", timeout=15000)
        except Exception as e:
            pytest.skip(f"map library not reachable: {e}")
        try:
            pg.wait_for_selector("#legend", state="visible", timeout=20000)
        except Exception:
            if any("WebGL" in t for _, t in console) or any("WebGL" in e for e in errors):
                pytest.skip("WebGL unavailable in this browser")
            raise
        pg.wait_for_timeout(2500)                    # let pins restore + frames load
        yield pg, errors, console
        browser.close()


def _width(pg, selector):
    return pg.evaluate("(s) => document.querySelector(s).getBoundingClientRect().width", selector)


class TestRealPage:
    def test_loads_without_uncaught_errors_and_restores_the_pins(self, page):
        pg, errors, _ = page
        assert errors == []
        assert pg.evaluate("document.querySelectorAll('#pins > div').length") >= 6

    def test_frames_load_and_render_values_into_the_pins(self, page):
        """Guards against a vacuous pass: the stream must be consumed and the
        field sampled, i.e. pins show numbers, not the empty-state dash."""
        pg, errors, _ = page
        pg.wait_for_function(
            "[...document.querySelectorAll('#pins > div')].some("
            "d => /\\(PM2\\.5\\): *[0-9]/.test(d.textContent))", timeout=45000)
        assert errors == []

    def test_cursor_box_appears_on_hover_and_matches_a_pin_box(self, page):
        pg, errors, _ = page
        pg.mouse.move(400, 400)
        pg.mouse.move(420, 410)
        pg.wait_for_timeout(300)
        assert pg.evaluate("getComputedStyle(document.getElementById('cursorReadout')).display") == "block"
        cursor_w = _width(pg, "#cursorReadout")
        pin_w = pg.evaluate("""() => {
            const boxes = [...document.querySelectorAll('#pins > div')].filter(
              d => getComputedStyle(d).width === '190px');
            return boxes.length ? boxes[0].getBoundingClientRect().width : null }""")
        assert pin_w is not None and abs(cursor_w - pin_w) < 1.0
        assert errors == []

    def test_setting_hides_the_cursor_box_and_persists(self, page):
        pg, errors, _ = page
        pg.click("#tabSettings")
        pg.uncheck("#showCursorBox")
        pg.mouse.move(500, 500); pg.mouse.move(520, 505)
        pg.wait_for_timeout(300)
        assert pg.evaluate("getComputedStyle(document.getElementById('cursorReadout')).display") == "none"
        assert "cursorBox=0" in pg.url
        pg.check("#showCursorBox")
        pg.mouse.move(540, 520); pg.mouse.move(560, 525)
        pg.wait_for_timeout(300)
        assert pg.evaluate("getComputedStyle(document.getElementById('cursorReadout')).display") == "block"
        assert "cursorBox" not in pg.url
        assert errors == []

    def test_cursor_box_scales_with_pin_size_like_the_pins(self, page):
        pg, errors, _ = page
        pg.click("#tabSettings")
        for _ in range(4):                           # 1.00x -> 2.00x
            pg.click("#pinSizeUp")
        pg.mouse.move(300, 300); pg.mouse.move(310, 305)
        pg.wait_for_timeout(300)
        cursor_w = _width(pg, "#cursorReadout")
        pin_w = pg.evaluate("""() => {
            const b = [...document.querySelectorAll('#pins > div')].filter(
              d => getComputedStyle(d).width === '190px')[0];
            return b.getBoundingClientRect().width }""")
        assert abs(cursor_w - 380) < 2 and abs(cursor_w - pin_w) < 2, (cursor_w, pin_w)
        assert errors == []

    def test_scrubber_handle_tracks_the_cursor_at_a_non_default_legend_size(self, page):
        pg, errors, _ = page
        # (click through the DOM: a 2x legend can extend past the viewport edge)
        for _ in range(4):                           # legend 1.00x -> 2.00x
            pg.evaluate("document.getElementById('legendSizeUp').click()")
        pg.wait_for_timeout(300)
        box = pg.evaluate("""() => { const b = document.getElementById('timeScrubber')
              .getBoundingClientRect(); return {left: b.left, top: b.top, width: b.width, height: b.height} }""")
        y = box["top"] + box["height"] / 2
        target_x = box["left"] + box["width"] * 0.5
        # a press-drag on the track moves the playhead to the pointer
        pg.mouse.move(target_x, y); pg.mouse.down(); pg.mouse.move(target_x + 1, y); pg.mouse.up()
        pg.wait_for_timeout(300)
        # where the playhead is DRAWN on screen (scaled) should be ~ under the cursor
        cx = pg.evaluate("""() => { const b = document.getElementById('scPlayhead')
              .getBoundingClientRect(); return b.left + b.width / 2 }""")
        assert abs(cx - target_x) < 12, (cx, target_x)
        assert errors == []
