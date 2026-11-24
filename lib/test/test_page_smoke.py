"""Real-browser smoke test of the map page (headless Chromium via Playwright).

`node --check` and string tests can't catch runtime errors in the page's script
(e.g. touching a not-yet-initialized `let` during startup, which once aborted
the whole load handler and broke the page). This loads the REAL page for a real
URL — pins, pin boxes, legend position — with only the API answered by stubs,
and fails on any uncaught page error. Skipped unless Playwright, its Chromium
and network access to the map library CDN are available.
"""
import json
import os
import re
import time
from urllib.parse import urlparse, parse_qs

import pytest

pw = pytest.importorskip("playwright.sync_api")
pytest.importorskip("jinja2")
pytest.importorskip("fastapi")

from fixture_harness import serve_api, map_html                # noqa: E402

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
    """Answer the page's data API from the real sample-data module (server/fixtures.py),
    for every layer — the same responses `fixture=1` gives a user."""
    return serve_api(route)


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
            if re.fullmatch(r"/map/models/[^/]+/", path):          # any layer, as the real route renders it
                return route.fulfill(body=map_html(route.request.url), content_type="text/html; charset=utf-8")
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


def _click(pg, element_id, shift=False):
    """Click through the DOM (a scaled legend can extend past the viewport edge)."""
    pg.evaluate("([id, shift]) => document.getElementById(id).dispatchEvent("
                "new MouseEvent('click', {bubbles: true, shiftKey: shift}))", [element_id, shift])


def _label(pg, element_id):
    return pg.inner_text(f"#{element_id}").strip()


def _url_param(pg, name):
    return pg.evaluate("(n) => new URL(location.href).searchParams.get(n)", name)


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

    def test_renaming_a_pin_commits_cleanly(self, page):
        """Real UI: click a pin's name, type, press Enter. The editor used to throw
        ("node to be removed is no longer a child") because removing the focused
        input fires `blur`, which committed a second time."""
        pg, errors, _ = page
        pg.wait_for_function("document.querySelectorAll('#pins [title=\"Click to rename\"]').length >= 6", timeout=30000)
        name = pg.locator('#pins [title="Click to rename"]').first
        name.click()
        pg.keyboard.press("Control+A")
        pg.keyboard.type("Renamed pin")
        pg.keyboard.press("Enter")
        pg.wait_for_function("[...document.querySelectorAll('#pins [title=\"Click to rename\"]')]"
                             ".some(e => e.textContent === 'Renamed pin')", timeout=8000)
        assert "Renamed" in pg.evaluate("new URL(location.href).searchParams.get('pins')")
        assert pg.evaluate("document.querySelectorAll('#pins input[type=text]').length") == 0   # editor closed
        assert errors == []

    def test_escape_cancels_a_pin_rename_without_errors(self, page):
        pg, errors, _ = page
        pg.wait_for_function("document.querySelectorAll('#pins [title=\"Click to rename\"]').length >= 6", timeout=30000)
        before = pg.locator('#pins [title="Click to rename"]').first.text_content()
        pg.locator('#pins [title="Click to rename"]').first.click()
        pg.keyboard.type("zzz"); pg.keyboard.press("Escape")
        pg.wait_for_timeout(300)
        assert pg.locator('#pins [title="Click to rename"]').first.text_content() == before
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
        for _ in range(4):                           # 1.00x -> 2.00x (Shift = coarse 0.25 step)
            pg.click("#pinSizeUp", modifiers=["Shift"])
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
        for _ in range(4):                           # legend 1.00x -> 2.00x (coarse steps)
            _click(pg, "legendSizeUp", shift=True)
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


class TestSizeSteps:
    """Pin and legend size: fine 0.05 clicks, Shift = coarse 0.25, hold to repeat."""

    @pytest.mark.parametrize("up,label,param", [("pinSizeUp", "pinSizeLabel", "pinSize"),
                                                ("legendSizeUp", "legendSizeLabel", "legendSize")])
    def test_a_click_is_one_fine_step(self, page, up, label, param):
        pg, errors, _ = page
        pg.click("#tabSettings")
        _click(pg, up)
        assert _label(pg, label) == "1.05×" and _url_param(pg, param) == "1.05"
        _click(pg, up)
        assert _label(pg, label) == "1.10×" and _url_param(pg, param) == "1.1"
        assert errors == []

    @pytest.mark.parametrize("up,down,label,param", [
        ("pinSizeUp", "pinSizeDown", "pinSizeLabel", "pinSize"),
        ("legendSizeUp", "legendSizeDown", "legendSizeLabel", "legendSize")])
    def test_no_floating_point_noise_after_many_steps(self, page, up, down, label, param):
        pg, _, _ = page
        pg.click("#tabSettings")
        for _ in range(7):
            _click(pg, up)                                   # 1.35: classic 0.05*n drift
        assert _url_param(pg, param) == "1.35" and _label(pg, label) == "1.35×"
        for _ in range(3):
            _click(pg, down)
        assert _url_param(pg, param) == "1.2"
        for _ in range(4):
            _click(pg, down)
        assert _label(pg, label) == "1.00×"
        assert _url_param(pg, param) is None                     # back at the default: omitted

    def test_shift_click_is_the_coarse_step(self, page):
        pg, _, _ = page
        pg.click("#tabSettings")
        _click(pg, "pinSizeUp", shift=True)
        assert _label(pg, "pinSizeLabel") == "1.25×"
        _click(pg, "pinSizeDown", shift=True); _click(pg, "pinSizeDown", shift=True)
        assert _label(pg, "pinSizeLabel") == "0.75×"
        _click(pg, "pinSizeUp")
        assert _label(pg, "pinSizeLabel") == "0.80×"           # fine steps resume from there

    @pytest.mark.parametrize("up,down,label,lo,hi", [
        ("pinSizeUp", "pinSizeDown", "pinSizeLabel", "0.50×", "3.00×"),
        ("legendSizeUp", "legendSizeDown", "legendSizeLabel", "0.50×", "3.00×")])
    def test_stays_within_the_range(self, page, up, down, label, lo, hi):
        pg, _, _ = page
        pg.click("#tabSettings")
        for _ in range(12):
            _click(pg, down, shift=True)
        assert _label(pg, label) == lo
        for _ in range(20):
            _click(pg, up, shift=True)
        assert _label(pg, label) == hi

    def test_holding_a_button_repeats_and_releasing_stops(self, page):
        pg, errors, _ = page
        pg.click("#tabSettings")
        pg.hover("#pinSizeUp")
        pg.mouse.down()
        # Wait for the CONDITION, not a fixed time (the repeat rate depends on how
        # busy the machine is): holding must keep stepping — at least 3 fine steps.
        pg.wait_for_function(
            "parseFloat(document.getElementById('pinSizeLabel').textContent) >= 1.15", timeout=15000)
        pg.mouse.up()
        settled = _label(pg, "pinSizeLabel")
        pg.wait_for_timeout(600)
        assert _label(pg, "pinSizeLabel") == settled             # stopped on release
        assert errors == []

    def test_any_value_in_the_url_is_honoured(self, page):
        pg, errors, _ = page
        pg.goto(f"{ORIGIN}/map/models/{MODEL}/?{QUERY}&pinSize=1.35&legendSize=0.65")
        pg.wait_for_selector("#legend", state="visible", timeout=30000)
        pg.wait_for_function("document.querySelectorAll('#pins > div').length >= 6", timeout=30000)
        assert _width(pg, "#legend") == pytest.approx(360 * 0.65, abs=1.5)
        pin_w = pg.evaluate("""() => [...document.querySelectorAll('#pins > div')].filter(
            d => getComputedStyle(d).width === '190px')[0].getBoundingClientRect().width""")
        assert pin_w == pytest.approx(190 * 1.35, abs=1.5)
        pg.click("#tabSettings")
        assert _label(pg, "pinSizeLabel") == "1.35×" and _label(pg, "legendSizeLabel") == "0.65×"
        _click(pg, "pinSizeUp")                                  # steps continue from the odd value
        assert _label(pg, "pinSizeLabel") == "1.40×"
        assert errors == []


class TestFloatingScrubber:
    """The "Floating scrub bar" setting moves the play button + scrubber + time
    label out of the legend into a bar at the bottom-center of the map."""

    VIEW_W = 1400

    def _inside(self, pg, element_id, container):
        return pg.evaluate("([a, c]) => document.getElementById(a).closest('#' + c) !== null",
                           [element_id, container])

    def _rect(self, pg, selector):
        return pg.evaluate("""(s) => { const b = document.querySelector(s).getBoundingClientRect()
            return {left: b.left, right: b.right, top: b.top, bottom: b.bottom, width: b.width} }""", selector)

    def test_docked_in_the_legend_by_default(self, page):
        pg, errors, _ = page
        for el in ("timeScrubber", "playPause", "timeSliderLabel"):
            assert self._inside(pg, el, "legend") and not self._inside(pg, el, "scrubberFloatBar")
        assert pg.evaluate("getComputedStyle(document.getElementById('scrubberFloatBar')).display") == "none"
        assert errors == []

    def test_the_setting_floats_it_to_the_bottom_center_and_back(self, page):
        pg, errors, _ = page
        pg.click("#tabSettings")
        pg.check("#floatScrubber")
        for el in ("timeScrubber", "playPause", "timeSliderLabel"):     # all three travel together
            assert self._inside(pg, el, "scrubberFloatBar") and not self._inside(pg, el, "legend")
        bar = self._rect(pg, "#scrubberFloatBar")
        assert abs((bar["left"] + bar["right"]) / 2 - self.VIEW_W / 2) < 2          # centered
        assert 840 <= bar["bottom"] <= 880 and bar["width"] <= 761                  # at the bottom
        assert "floatScrubber=1" in pg.url
        pg.uncheck("#floatScrubber")
        for el in ("timeScrubber", "playPause", "timeSliderLabel"):
            assert self._inside(pg, el, "legend") and not self._inside(pg, el, "scrubberFloatBar")
        assert "floatScrubber" not in pg.url
        assert pg.evaluate("getComputedStyle(document.getElementById('scrubberFloatBar')).display") == "none"
        assert errors == []

    def test_docking_back_restores_the_original_order_in_the_controls_panel(self, page):
        pg, _, _ = page
        before = pg.evaluate("[...document.getElementById('panelControls').children].map(c => c.id || c.tagName)")
        pg.click("#tabSettings")
        pg.check("#floatScrubber"); pg.uncheck("#floatScrubber")
        after = pg.evaluate("[...document.getElementById('panelControls').children].map(c => c.id || c.tagName)")
        assert before == after

    def test_it_starts_floating_when_the_url_says_so(self, page):
        pg, errors, _ = page
        pg.goto(f"{ORIGIN}/map/models/{MODEL}/?{QUERY}&floatScrubber=1")
        pg.wait_for_selector("#scrubberFloatBar", state="visible", timeout=30000)
        assert self._inside(pg, "timeScrubber", "scrubberFloatBar")
        pg.click("#tabSettings")
        assert pg.is_checked("#floatScrubber")
        assert errors == []

    def test_it_matches_the_legends_translucent_look(self, page):
        """Same fill, corner radius and shadow as the legend panel."""
        pg, errors, _ = page
        pg.click("#tabSettings"); pg.check("#floatScrubber")
        look = lambda sel: pg.evaluate("""(s) => { const c = getComputedStyle(document.querySelector(s))
            return {bg: c.backgroundColor, radius: c.borderRadius, shadow: c.boxShadow} }""", sel)
        assert look("#scrubberFloatBar") == look("#legend")
        assert errors == []

    def test_playback_controls_still_work_while_floating(self, page):
        pg, errors, _ = page
        pg.click("#tabSettings"); pg.check("#floatScrubber")
        before = pg.inner_text("#playPause")
        pg.click("#playPause")
        assert pg.inner_text("#playPause") != before
        pg.click("#playPause")
        assert pg.inner_text("#playPause") == before
        assert errors == []

    def test_dragging_tracks_the_cursor_and_ignores_the_legend_size(self, page):
        pg, errors, _ = page
        pg.click("#tabSettings"); pg.check("#floatScrubber")
        for _ in range(4):                                                # then the legend to 2x
            _click(pg, "legendSizeUp", shift=True)                        # (via the DOM: a 2x legend can leave the window)
        assert _label(pg, "legendSizeLabel") == "2.00×"
        bar = self._rect(pg, "#scrubberFloatBar")
        assert bar["width"] == pytest.approx(min(760, self.VIEW_W - 32), abs=1)   # NOT scaled by the legend
        sc = self._rect(pg, "#timeScrubber")
        y = (sc["top"] + sc["bottom"]) / 2
        for frac in (0.25, 0.5, 0.75):
            x = sc["left"] + sc["width"] * frac
            pg.mouse.move(x, y); pg.mouse.down(); pg.mouse.move(x + 1, y); pg.mouse.up()
            pg.wait_for_timeout(200)
            head = pg.evaluate("(() => { const b = document.getElementById('scPlayhead').getBoundingClientRect(); return b.left + b.width / 2 })()")
            assert abs(head - x) < 14, (frac, head, x)
        assert errors == []

    def test_it_fits_a_narrow_viewport(self, page):
        pg, errors, _ = page
        pg.click("#tabSettings"); pg.check("#floatScrubber")
        pg.set_viewport_size({"width": 480, "height": 320})
        pg.wait_for_timeout(400)
        bar = self._rect(pg, "#scrubberFloatBar")
        assert bar["left"] >= 0 and bar["right"] <= 480 and bar["bottom"] <= 320
        assert bar["width"] == pytest.approx(480 - 32, abs=1)
        sc = self._rect(pg, "#timeScrubber")
        assert sc["width"] > 150                                            # still usable
        assert errors == []


class TestMapControls:
    """Scale bar top-right; the basemap credit button hidden unless asked for."""

    def _count(self, pg, selector):
        return pg.evaluate("(s) => document.querySelectorAll(s).length", selector)

    def test_scale_bar_is_top_right_and_nowhere_else(self, page):
        pg, errors, _ = page
        assert self._count(pg, ".maplibregl-ctrl-top-right .maplibregl-ctrl-scale") == 1
        assert self._count(pg, ".maplibregl-ctrl-bottom-left .maplibregl-ctrl-scale") == 0
        r = pg.evaluate("""() => { const b = document.querySelector('.maplibregl-ctrl-scale').getBoundingClientRect()
            return {right: b.right, top: b.top, vw: innerWidth} }""")
        assert r["vw"] - r["right"] < 30 and r["top"] < 30           # hugging the top-right corner
        assert errors == []

    def test_no_attribution_button_by_default(self, page):
        pg, errors, _ = page
        assert self._count(pg, ".maplibregl-ctrl-attrib") == 0
        assert self._count(pg, ".maplibregl-ctrl-attrib-button") == 0
        assert errors == []

    def test_settings_checkbox_brings_the_credit_button_back_collapsed_and_removes_it(self, page):
        pg, errors, _ = page
        pg.click("#tabSettings")
        pg.check("#showAttribution")
        assert self._count(pg, ".maplibregl-ctrl-bottom-right .maplibregl-ctrl-attrib-button") == 1
        assert "attribution=1" in pg.url
        assert not pg.evaluate("document.querySelector('.maplibregl-ctrl-attrib').classList.contains('maplibregl-compact-show')")
        pg.uncheck("#showAttribution")
        assert self._count(pg, ".maplibregl-ctrl-attrib") == 0 and "attribution" not in pg.url
        assert errors == []

    def test_the_credit_text_is_reachable_when_enabled(self, page):
        pg, errors, _ = page
        pg.click("#tabSettings"); pg.check("#showAttribution")
        pg.click(".maplibregl-ctrl-attrib-button")                    # expand
        pg.wait_for_function("document.querySelector('.maplibregl-ctrl-attrib-inner') && "
                             "/OpenStreetMap/.test(document.querySelector('.maplibregl-ctrl-attrib-inner').textContent)",
                             timeout=20000)
        assert errors == []

    def test_it_starts_enabled_collapsed_when_the_url_says_so(self, page):
        pg, errors, _ = page
        pg.goto(f"{ORIGIN}/map/models/{MODEL}/?{QUERY}&attribution=1")
        pg.wait_for_selector("#legend", state="visible", timeout=30000)
        pg.wait_for_selector(".maplibregl-ctrl-attrib-button", state="attached", timeout=15000)
        pg.wait_for_timeout(800)
        assert self._count(pg, ".maplibregl-ctrl-attrib") == 1
        assert not pg.evaluate("document.querySelector('.maplibregl-ctrl-attrib').classList.contains('maplibregl-compact-show')")
        pg.click("#tabSettings")
        assert pg.is_checked("#showAttribution")
        assert errors == []


class TestViewCenter:
    """?mapLat/?mapLon: where the map is LOOKING. Set from the URL, kept current as the
    map moves, and separate from the data area's center (lat/lon)."""

    BASE = "lat=48.145506&lon=-116.391332&width=201.90&height=183.69&dt=900"   # no pins/legend params: clear to drag

    def _open(self, pg, extra="", size=(1400, 900)):
        pg.goto(f"{ORIGIN}/map/models/{MODEL}/?{self.BASE}{extra}")
        pg.wait_for_function("(() => { try { return !!window.WRLive } catch (e) { return false } })()", timeout=30000)
        pg.wait_for_timeout(500)

    def _view(self, pg):
        return pg.evaluate("window.WRLive.view()")

    def _param(self, pg, name):
        return pg.evaluate("(n) => new URL(location.href).searchParams.get(n)", name)

    def _settle(self, pg, timeout=8000):
        """Wait until the map has stopped moving (a released drag glides on inertia) and the
        URL shows exactly where it stopped — the page writes it once more when movement ends."""
        pg.wait_for_function("""() => { const v = window.WRLive.view(), now = performance.now()
            const w = window.__settle || (window.__settle = {lat: v.lat, lon: v.lon, since: now})
            if (Math.abs(w.lat - v.lat) > 1e-9 || Math.abs(w.lon - v.lon) > 1e-9) { w.lat = v.lat; w.lon = v.lon; w.since = now }
            const p = new URL(location.href).searchParams
            const la = parseFloat(p.get('mapLat')), lo = parseFloat(p.get('mapLon'))
            return now - w.since > 400 && Math.abs(la - v.lat) < 1e-4 && Math.abs(lo - v.lon) < 1e-4 }""",
                             timeout=timeout, polling=100)
        pg.evaluate("delete window.__settle")

    def _drag(self, pg, dx, dy, hold_ms=0, release=True):
        pg.mouse.move(800, 600); pg.mouse.down()
        for i in range(1, 11):
            pg.mouse.move(800 + dx * i / 10, 600 + dy * i / 10)
        if hold_ms:
            pg.wait_for_timeout(hold_ms)
        if release:
            pg.mouse.up()

    def test_it_defaults_to_the_area_center_and_adds_no_params(self, page):
        pg, errors, _ = page
        self._open(pg)
        v = self._view(pg)
        assert (v["lat"], v["lon"]) == pytest.approx((48.145506, -116.391332), abs=1e-4)
        assert self._param(pg, "mapLat") is None and self._param(pg, "mapLon") is None
        assert errors == []

    def test_the_url_sets_where_the_map_looks_without_changing_the_area(self, page):
        pg, errors, _ = page
        self._open(pg, "&mapLat=47.9&mapLon=-115.9&zoom=9")
        v = self._view(pg)
        assert (v["lat"], v["lon"], v["zoom"]) == pytest.approx((47.9, -115.9, 9.0), abs=1e-3)
        assert self._param(pg, "lat") == "48.145506" and self._param(pg, "lon") == "-116.391332"   # the data area is untouched
        assert errors == []

    def test_one_axis_overrides_only_that_axis(self, page):
        pg, _, _ = page
        self._open(pg, "&mapLat=47.9")
        v = self._view(pg)
        assert (v["lat"], v["lon"]) == pytest.approx((47.9, -116.391332), abs=1e-3)

    @pytest.mark.parametrize("extra", ["&mapLat=999&mapLon=abc", "&mapLat=&mapLon=", "&mapLat=47.9&mapLon=-999", "&mapLat=NaN"])
    def test_invalid_values_are_ignored_instead_of_breaking_the_page(self, page, extra):
        pg, errors, _ = page
        self._open(pg, extra)
        v = self._view(pg)
        assert abs(v["lat"]) <= 90 and abs(v["lon"]) <= 180
        assert errors == []

    def test_panning_updates_the_url_while_you_drag_and_when_you_let_go(self, page):
        pg, errors, _ = page
        self._open(pg)
        self._drag(pg, 160, 100, hold_ms=500, release=False)                  # still holding the mouse
        assert self._param(pg, "mapLat") is not None                         # the URL followed DURING the drag
        pg.mouse.up()
        self._settle(pg)
        v = self._view(pg)
        assert float(self._param(pg, "mapLat")) == pytest.approx(v["lat"], abs=1e-4)
        assert float(self._param(pg, "mapLon")) == pytest.approx(v["lon"], abs=1e-4)
        assert abs(v["lat"] - 48.145506) > 0.01 or abs(v["lon"] + 116.391332) > 0.01           # it really moved
        assert self._param(pg, "lat") == "48.145506"                          # panning never changes the data area
        assert errors == []

    def test_a_reload_comes_back_to_the_same_view(self, page):
        pg, errors, _ = page
        self._open(pg)
        self._drag(pg, -140, 90)
        self._settle(pg)
        before = self._view(pg); url = pg.url
        pg.goto(url)
        pg.wait_for_function("(() => { try { return !!window.WRLive } catch (e) { return false } })()", timeout=30000)
        after = self._view(pg)
        assert (after["lat"], after["lon"]) == pytest.approx((before["lat"], before["lon"]), abs=2e-4)
        assert errors == []

    def test_moving_back_onto_the_area_center_removes_the_params(self, page):
        pg, _, _ = page
        self._open(pg, "&mapLat=47.9&mapLon=-115.9")
        assert self._param(pg, "mapLat") == "47.9"
        pg.evaluate("window.WRLive.center(48.145506, -116.391332)")
        pg.wait_for_timeout(300)
        assert self._param(pg, "mapLat") is None and self._param(pg, "mapLon") is None

    def test_the_view_survives_a_model_switch(self, page):
        pg, errors, _ = page
        self._open(pg, "&mapLat=47.9&mapLon=-115.9")
        with pg.expect_navigation():
            pg.select_option("#modelSelect", "temperature")
        pg.wait_for_function("(() => { try { return !!window.WRLive } catch (e) { return false } })()", timeout=30000)
        v = self._view(pg)
        assert (v["lat"], v["lon"]) == pytest.approx((47.9, -115.9), abs=1e-3)
        assert "/map/models/temperature/" in pg.url and self._param(pg, "mapLat") == "47.9"
        assert errors == []

    def test_recentering_the_area_moves_both_and_leaves_no_view_override(self, page):
        pg, errors, _ = page
        self._open(pg, "&mapLat=47.9&mapLon=-115.9")
        pg.mouse.click(700, 450, button="right")
        pg.click("#ctxCenterHere")
        pg.wait_for_timeout(1800)                                             # the 500 ms ease + reload
        v = self._view(pg)
        assert float(self._param(pg, "lat")) == pytest.approx(v["lat"], abs=1e-3)       # area follows the new center
        assert float(self._param(pg, "lon")) == pytest.approx(v["lon"], abs=1e-3)
        assert self._param(pg, "mapLat") is None and self._param(pg, "mapLon") is None
        assert errors == []

    def test_both_params_are_carried_by_every_model_switch_path(self, page):
        import re
        html = open(os.path.join(os.path.dirname(__file__), "..", "..", "server", "src", "weather_radar",
                                 "server", "app", "map.html"), encoding="utf-8").read()
        lists = re.findall(r'for \(const k of \[("lat", "lon".*?)\]\) \{', html, re.S)
        assert len(lists) == 2
        assert all('"mapLat"' in l and '"mapLon"' in l for l in lists)


class TestZoomHook:
    """WRLive.zoom: zoom the view in place (what ?zoom describes)."""

    URL = "lat=48.145506&lon=-116.391332&width=201.90&height=183.69&dt=900"

    def _open(self, pg, extra=""):
        pg.goto(f"{ORIGIN}/map/models/{MODEL}/?{self.URL}{extra}")
        pg.wait_for_function("(() => { try { return !!window.WRLive } catch (e) { return false } })()", timeout=30000)
        pg.wait_for_timeout(400)

    def _zoom_param(self, pg):
        return pg.evaluate("new URL(location.href).searchParams.get('zoom')")

    def test_zooms_in_place_and_updates_the_url(self, page):
        pg, errors, _ = page
        self._open(pg)
        pg.evaluate("window.__marker = 5")
        pg.evaluate("window.WRLive.zoom(9.25)")
        assert pg.evaluate("window.WRLive.view().zoom") == pytest.approx(9.25, abs=1e-6)
        assert self._zoom_param(pg) == "9.25"
        assert pg.evaluate("window.__marker") == 5                      # nothing reloaded
        assert errors == []

    def test_it_keeps_the_view_center_while_zooming(self, page):
        pg, _, _ = page
        self._open(pg, "&mapLat=47.9&mapLon=-115.9")
        pg.evaluate("window.WRLive.zoom(10)")
        v = pg.evaluate("window.WRLive.view()")
        assert (v["lat"], v["lon"]) == pytest.approx((47.9, -115.9), abs=1e-4)

    @pytest.mark.parametrize("bad", ["NaN", "Infinity", "99", "-3", "'abc'"])
    def test_invalid_zoom_is_ignored(self, page, bad):
        pg, errors, _ = page
        self._open(pg, "&zoom=8")
        pg.evaluate(f"window.WRLive.zoom({bad})")
        assert pg.evaluate("window.WRLive.view().zoom") == pytest.approx(8.0, abs=1e-6)
        assert errors == []

    @pytest.mark.parametrize("blank", ["null", "undefined", "''"])
    def test_blank_means_the_default_and_removes_the_parameter(self, page, blank):
        pg, _, _ = page
        self._open(pg, "&zoom=10")
        pg.evaluate(f"window.WRLive.zoom({blank})")
        assert pg.evaluate("window.WRLive.view().zoom") == pytest.approx(7.65, abs=1e-6)
        assert self._zoom_param(pg) is None

    def test_a_mouse_wheel_still_writes_the_zoom_to_the_url(self, page):
        pg, errors, _ = page
        self._open(pg)
        pg.mouse.move(800, 600)
        pg.mouse.wheel(0, -400)
        pg.wait_for_function("window.WRLive.view().zoom > 7.8", timeout=8000)
        pg.wait_for_function("(() => { const z = new URL(location.href).searchParams.get('zoom'); return !!z && Math.abs(parseFloat(z) - window.WRLive.view().zoom) < 0.01 })()", timeout=8000)
        assert errors == []


class TestLegendPanels:
    """Speed and Time live under the legend's Settings tab; Contour stays under Controls."""

    def _panel_of(self, pg, el_id):
        return pg.evaluate("(id) => { const p = document.getElementById(id).closest('#panelControls, #panelSettings'); return p && p.id }", el_id)

    def test_speed_and_time_are_in_settings_and_contour_stays_in_controls(self, page):
        pg, errors, _ = page
        for el in ("speedDown", "speedLabel", "speedUp", "dtDown", "dtLabel", "dtUp"):
            assert self._panel_of(pg, el) == "panelSettings", el
        for el in ("stepDown", "stepLabel", "stepUp", "playPause", "timeScrubber"):
            assert self._panel_of(pg, el) == "panelControls", el
        assert errors == []

    def test_they_show_with_the_settings_tab_and_not_with_controls(self, page):
        pg, _, _ = page
        assert not pg.is_visible("#speedUp") and not pg.is_visible("#dtUp")
        pg.click("#tabSettings")
        assert pg.is_visible("#speedUp") and pg.is_visible("#dtUp")
        assert not pg.is_visible("#playPause")
        pg.click("#tabControls")
        assert not pg.is_visible("#speedUp") and pg.is_visible("#playPause")

    def test_they_come_first_in_settings_above_the_sizes(self, page):
        pg, _, _ = page
        order = pg.evaluate("""() => [...document.querySelectorAll('#panelSettings span')]
            .map(s => s.textContent.trim()).filter(t => ['Speed', 'Time', 'Pin size', 'Legend size'].includes(t))""")
        assert order == ["Speed", "Time", "Pin size", "Legend size"]

    def test_the_buttons_still_work_from_their_new_place(self, page):
        pg, errors, _ = page
        pg.click("#tabSettings")
        s0, d0 = pg.inner_text("#speedLabel"), pg.inner_text("#dtLabel")
        pg.click("#speedUp"); pg.click("#dtUp")
        s1, d1 = pg.inner_text("#speedLabel"), pg.inner_text("#dtLabel")
        assert s1 != s0 and d1 != d0
        pg.click("#speedDown"); pg.click("#dtDown")
        assert (pg.inner_text("#speedLabel"), pg.inner_text("#dtLabel")) == (s0, d0)
        assert errors == []

    def test_they_still_work_with_the_floating_scrub_bar(self, page):
        pg, errors, _ = page
        pg.click("#tabSettings"); pg.check("#floatScrubber")
        assert self._panel_of(pg, "speedUp") == "panelSettings"
        before = pg.inner_text("#speedLabel"); pg.click("#speedUp")
        assert pg.inner_text("#speedLabel") != before
        assert errors == []

    def test_a_taller_tab_near_the_bottom_stays_fully_on_screen(self, page):
        pg, errors, _ = page
        # (this page's legend is anchored low on the map, so the taller Settings tab would run off the bottom)
        bottom = pg.evaluate("window.innerHeight")
        r = pg.evaluate("(() => { const b = document.getElementById('legend').getBoundingClientRect(); return [b.top, b.bottom] })()")
        assert r[1] <= bottom
        pg.click("#tabSettings")
        pg.wait_for_timeout(300)
        for el in ("showAttribution", "lockLayout", "floatScrubber"):                 # the LAST options in the panel
            box = pg.evaluate("(id) => { const b = document.getElementById(id).getBoundingClientRect(); return [b.top, b.bottom] }", el)
            assert 0 <= box[0] and box[1] <= bottom, (el, box, bottom)
        pg.click("#tabControls")
        pg.wait_for_timeout(300)
        assert pg.evaluate("document.getElementById('legend').getBoundingClientRect().bottom") <= bottom
        assert errors == []


class TestScaleBarSetting:
    URL = "lat=48.145506&lon=-116.391332&width=201.90&height=183.69&dt=900"

    def _scale_count(self, pg):
        return pg.evaluate("document.querySelectorAll('.maplibregl-ctrl-scale').length")

    def _param(self, pg):
        return pg.evaluate("new URL(location.href).searchParams.get('scaleBar')")

    def test_shown_by_default_with_a_ticked_box_in_settings(self, page):
        pg, errors, _ = page
        assert self._scale_count(pg) == 1
        pg.click("#tabSettings")
        assert pg.is_checked("#showScaleBar") and self._param(pg) is None
        assert errors == []

    def test_unticking_removes_it_and_records_that_in_the_url_and_ticking_restores_it(self, page):
        pg, errors, _ = page
        pg.click("#tabSettings")
        pg.uncheck("#showScaleBar")
        assert self._scale_count(pg) == 0 and self._param(pg) == "0"
        pg.check("#showScaleBar")
        assert self._scale_count(pg) == 1 and self._param(pg) is None
        assert errors == []

    def test_toggling_repeatedly_never_leaves_two_of_them(self, page):
        pg, _, _ = page
        pg.click("#tabSettings")
        for _ in range(4):
            pg.uncheck("#showScaleBar"); pg.check("#showScaleBar")
        assert self._scale_count(pg) == 1

    def test_the_url_can_start_it_hidden(self, page):
        pg, errors, _ = page
        pg.goto(f"{ORIGIN}/map/models/{MODEL}/?{self.URL}&scaleBar=0")
        pg.wait_for_function("(() => { try { return !!window.WRLive } catch (e) { return false } })()", timeout=30000)
        assert self._scale_count(pg) == 0
        pg.click("#tabSettings")
        assert not pg.is_checked("#showScaleBar")
        assert errors == []

    def test_the_live_hook_toggles_it_and_keeps_the_box_in_step(self, page):
        pg, _, _ = page
        pg.evaluate("window.WRLive.scaleBar(false)")
        pg.click("#tabSettings")
        assert self._scale_count(pg) == 0 and not pg.is_checked("#showScaleBar")
        pg.evaluate("window.WRLive.scaleBar(true)")
        assert self._scale_count(pg) == 1 and pg.is_checked("#showScaleBar")

    def test_it_survives_a_model_switch_on_both_paths(self, page):
        import re
        html = open(os.path.join(os.path.dirname(__file__), "..", "..", "server", "src", "weather_radar",
                                 "server", "app", "map.html"), encoding="utf-8").read()
        lists = re.findall(r'for \(const k of \[("lat", "lon".*?)\]\) \{', html, re.S)
        assert len(lists) == 2 and all('"scaleBar"' in l for l in lists)

    def test_hiding_it_leaves_no_empty_control_box_behind(self, page):
        pg, errors, _ = page
        pg.click("#tabSettings"); pg.uncheck("#showScaleBar")
        pg.wait_for_timeout(300)
        assert pg.evaluate("document.querySelectorAll('.maplibregl-ctrl-top-right .maplibregl-ctrl').length") == 0
        assert errors == []
