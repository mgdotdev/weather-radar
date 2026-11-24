"""The /graph-preview/ page: a control per graph-page parameter, a frame of the
chosen Home Assistant card size, the URL + card YAML, and "follow".

Real headless Chromium against the real preview page and the real graph page
inside its iframe; only the data API is stubbed. Skipped unless Playwright +
Chromium + the plotting-library CDN are available."""
import json
from urllib.parse import urlparse, parse_qs, quote

import pytest

pw = pytest.importorskip("playwright.sync_api")
pytest.importorskip("jinja2")
pytest.importorskip("fastapi")

from fixture_harness import serve_api                        # noqa: E402

ORIGIN = "http://localhost:8000"
PIN_BUTTON_URL = "/graph/?models=%7B%22aqi%22%3A%5B%22pm25%22%5D%7D&lat=48.188820&lon=-116.081840"


def _graph_html(url):
    import weather_radar.server.app as app
    q = parse_qs(urlparse(url).query)
    g = lambda k, d: q[k][0] if k in q else d
    return app.graph(models=g("models", "{}"), lat=float(g("lat", 0)), lon=float(g("lon", 0)),
                     dark_mode=g("dark_mode", "true").lower() not in ("false", "0"),
                     dt=int(g("dt", 450)), counts=int(g("counts", -1)),
                     ha=g("ha", "false").lower() in ("true", "1"))


@pytest.fixture(scope="module")
def preview_html():
    import weather_radar.server.app as app
    return app.graph_preview()


@pytest.fixture
def pv(preview_html):
    errors = []
    with pw.sync_playwright() as p:
        try:
            browser = p.chromium.launch(args=["--no-sandbox"])
        except Exception as e:
            pytest.skip(f"chromium unavailable: {e}")
        ctx = browser.new_context(viewport={"width": 1500, "height": 950})
        ctx.grant_permissions(["clipboard-read", "clipboard-write"], origin=ORIGIN)

        def handle(route):
            path = urlparse(route.request.url).path
            if path.startswith("/graph-preview"):
                return route.fulfill(body=preview_html, content_type="text/html; charset=utf-8")
            if path.startswith("/graph/"):
                return route.fulfill(body=_graph_html(route.request.url), content_type="text/html; charset=utf-8")
            if path.startswith("/api/models/"):
                return serve_api(route, open_preview.seen)
            return route.fulfill(status=404, body="")
        ctx.route(f"{ORIGIN}/**", handle)

        def open_preview(query="", hash_=""):
            pg = ctx.new_page()
            pg.on("pageerror", lambda e: errors.append(str(e)))
            pg.goto(f"{ORIGIN}/graph-preview/{query}{hash_}", wait_until="load")
            pg.wait_for_function("window.__graphPreview !== undefined")
            pg.evaluate("document.querySelectorAll('details').forEach(d => d.open = true)")
            return pg
        open_preview.seen = []
        yield open_preview, errors
        browser.close()


def _src(pg):
    src = pg.evaluate("document.getElementById('frame').getAttribute('src')")
    u = urlparse(src)
    return u.path, {k: v[0] for k, v in parse_qs(u.query, keep_blank_values=True).items()}


def _wait_src(pg, js, timeout=8000):
    pg.wait_for_function("(js) => { const s = document.getElementById('frame').getAttribute('src') || '';"
                         " return new Function('src', 'return ' + js)(s) }", arg=js, timeout=timeout)


def _traces(pg):
    """How many curves the REAL graph page has drawn inside the iframe."""
    return pg.evaluate("""() => { try { const g = document.getElementById('frame').contentDocument.getElementById('graph')
        return g && g.data ? g.data.length : -1 } catch (_) { return -1 } }""")


def _wait_traces(pg, n, timeout=30000):
    pg.wait_for_function("""(n) => { try { const g = document.getElementById('frame').contentDocument.getElementById('graph')
        return !!(g && g.data && g.data.length === n && g.data.every(t => t.x && t.x.length > 0)) } catch (_) { return false } }""",
                         arg=n, timeout=timeout)


def _models(pg):
    return json.loads(_src(pg)[1]["models"])


class TestRoute:
    def test_renders_with_catalog_defaults_and_navigation(self, preview_html):
        assert "{{" not in preview_html and "{%" not in preview_html
        boot = json.loads(preview_html.split("const BOOT = ")[1].split("\n")[0])
        values = {e["value"] for e in boot["catalog"]}
        assert {"observation|temperature", "accumulation|snowfall-amount", "aqi|pm25"} <= values
        assert boot["defaults"]["dt"] == 450 and boot["defaults"]["counts"] == -1
        assert 'href="/ha-preview/"' in preview_html and 'class="nav here" href="/graph-preview/"' in preview_html


class TestControls:
    def test_loads_cleanly_and_draws_the_default_graph(self, pv):
        open_preview, errors = pv
        pg = open_preview()
        path, q = _src(pg)
        assert path == "/graph/" and json.loads(q["models"]) == {"observation": ["temperature"]}
        assert {"lat", "lon"} <= set(q) and not ({"dt", "counts", "ha", "dark_mode", "title"} & set(q))
        _wait_traces(pg, 1)
        assert "type: iframe" in pg.inner_text("#yamlOut") and "/graph/" in pg.inner_text("#urlOut")
        assert errors == []

    @pytest.mark.parametrize("control,value,param,expected", [
        ("lat", "47.5", "lat", "47.5"), ("lon", "-116.25", "lon", "-116.25"),
        ("counts", "5", "counts", "5"), ("title", "Beehive Lake", "title", "Beehive Lake"),
        ("titleX", "120", "titleX", "120"), ("titleY", "40", "titleY", "40"), ("titleSize", "30", "titleSize", "30"),
    ])
    def test_each_field_writes_its_parameter(self, pv, control, value, param, expected):
        open_preview, _ = pv
        pg = open_preview()
        pg.fill(f"#{control}", value); pg.press(f"#{control}", "Tab")
        _wait_src(pg, f"src.includes('{param}=')")
        assert _src(pg)[1][param] == expected and param + "=" in pg.inner_text("#urlOut")

    def test_selects_and_checkboxes_and_back_to_defaults(self, pv):
        open_preview, errors = pv
        pg = open_preview()
        pg.select_option("#dt", "900"); pg.check("#ha"); pg.uncheck("#dark")
        _wait_src(pg, "src.includes('ha=true') && src.includes('dark_mode=false') && src.includes('dt=900')")
        q = _src(pg)[1]
        assert (q["dt"], q["ha"], q["dark_mode"]) == ("900", "true", "false")
        pg.select_option("#dt", "450"); pg.uncheck("#ha"); pg.check("#dark")
        _wait_src(pg, "!src.includes('ha=') && !src.includes('dark_mode') && !src.includes('dt=')")
        assert errors == []

    def test_an_unusual_time_step_is_kept_not_blanked(self, pv):
        open_preview, _ = pv
        pg = open_preview()
        pg.fill("#importUrl", f"{ORIGIN}/graph/?models=%7B%7D&lat=1&lon=2&dt=1200"); pg.click("#importBtn")
        assert pg.input_value("#dt") == "1200" and _src(pg)[1]["dt"] == "1200"

    def test_unknown_parameters_are_preserved(self, pv):
        open_preview, _ = pv
        pg = open_preview()
        pg.fill("#importUrl", f"{ORIGIN}/graph/?models=%7B%7D&lat=1&lon=2&foo=bar"); pg.click("#importBtn")
        pg.fill("#lat", "9"); pg.press("#lat", "Tab")
        _wait_src(pg, "src.includes('lat=9')")
        assert _src(pg)[1]["foo"] == "bar"


class TestCurves:
    def test_curves_drive_the_models_parameter_and_the_real_graph_draws_them(self, pv):
        open_preview, errors = pv
        pg = open_preview()
        _wait_traces(pg, 1)
        pg.click("#addCurve")                                          # a second curve (next unused layer)
        _wait_src(pg, "decodeURIComponent(src).includes('],')  || decodeURIComponent(src).split('\"').length > 6")
        total = sum(len(v) for v in _models(pg).values())
        assert total == 2
        _wait_traces(pg, 2)                                            # the REAL graph page plots both
        pg.locator("#curves .curve").nth(1).locator(".sel").select_option("aqi|pm25")
        _wait_src(pg, "decodeURIComponent(src).includes('aqi')")
        assert _models(pg) == {"observation": ["temperature"], "aqi": ["pm25"]}
        pg.locator("#curves .curve").nth(0).locator(".x").click()      # remove the first
        _wait_src(pg, "!decodeURIComponent(src).includes('observation')")
        assert _models(pg) == {"aqi": ["pm25"]}
        _wait_traces(pg, 1)
        assert errors == []

    def test_no_curves_gives_an_empty_models_object(self, pv):
        open_preview, _ = pv
        pg = open_preview()
        pg.locator("#curves .curve").nth(0).locator(".x").click()
        _wait_src(pg, "decodeURIComponent(src).includes('models={}')")
        assert "No curves" in pg.inner_text("#curves")

    def test_the_u_parameter_starts_from_a_full_graph_url(self, pv):
        open_preview, errors = pv
        full = (ORIGIN + "/graph/?models=" + quote('{"observation":["temperature","dewpoint"]}', safe="")
                + "&lat=47.77877&lon=-116.70459&ha=true&title=Home&dt=900")
        pg = open_preview("?u=" + quote(full, safe=""))
        assert pg.locator("#curves .curve").count() == 2
        assert pg.is_checked("#ha") and pg.input_value("#title") == "Home" and pg.input_value("#dt") == "900"
        assert float(pg.input_value("#lat")) == pytest.approx(47.77877)
        _wait_traces(pg, 2)
        assert pg.evaluate("document.getElementById('frame').contentDocument.getElementById('graph').layout.font.size") == 12
        assert errors == []

    def test_import_of_the_url_a_pin_button_opens(self, pv):
        open_preview, errors = pv
        pg = open_preview()
        pg.fill("#importUrl", ORIGIN + PIN_BUTTON_URL); pg.click("#importBtn")
        assert pg.locator("#curves .curve").count() == 1
        assert pg.locator("#curves .sel").input_value() == "aqi|pm25"
        assert float(pg.input_value("#lat")) == pytest.approx(48.18882)
        assert float(pg.input_value("#lon")) == pytest.approx(-116.08184)
        path, q = _src(pg)
        assert json.loads(q["models"]) == {"aqi": ["pm25"]} and q["lat"] == "48.18882"
        _wait_traces(pg, 1)
        assert errors == []

    def test_a_curve_not_in_the_catalog_is_kept(self, pv):
        open_preview, _ = pv
        pg = open_preview()
        pg.fill("#importUrl", ORIGIN + "/graph/?models=%7B%22observation%22%3A%5B%22made-up%22%5D%7D&lat=1&lon=2")
        pg.click("#importBtn")
        assert pg.locator("#curves .sel").input_value() == "observation|made-up"
        assert _models(pg) == {"observation": ["made-up"]}


class TestLook:
    def _style_text(self, pg):
        return pg.evaluate("[...document.getElementById('frame').contentDocument.querySelectorAll('style')]"
                           ".map(s => s.textContent).join(' ')")

    def test_dark_mode_toggles_the_pages_inversion_filter(self, pv):
        open_preview, _ = pv
        pg = open_preview()
        _wait_traces(pg, 1)
        assert "invert(75%)" in self._style_text(pg)
        pg.uncheck("#dark")
        _wait_src(pg, "src.includes('dark_mode=false')")
        _wait_traces(pg, 1)
        assert "invert(75%)" not in self._style_text(pg)

    def test_the_compact_layout_really_changes_the_graph(self, pv):
        open_preview, _ = pv
        pg = open_preview()
        _wait_traces(pg, 1)
        size = lambda: pg.evaluate("document.getElementById('frame').contentDocument.getElementById('graph').layout.font.size")
        assert size() == 24
        pg.check("#ha")
        _wait_src(pg, "src.includes('ha=true')")
        _wait_traces(pg, 1)
        assert size() == 12                                             # the Home-Assistant sized text
        assert pg.evaluate("document.getElementById('frame').contentDocument.getElementById('graph').layout.legend.orientation") == "h"


class TestPlotFollowsTheFrame:
    """The plot must re-lay-out when the frame is resized (a card reflowing, a
    tablet rotating, or changing the size in this preview) instead of keeping
    the size it had when it loaded."""

    def _plot_width(self, pg):
        return pg.evaluate("""() => { try { const d = document.getElementById('frame').contentDocument
            return d.querySelector('#graph .main-svg').getBoundingClientRect().width } catch (_) { return -1 } }""")

    def test_resizing_the_card_resizes_the_plot(self, pv):
        open_preview, errors = pv
        pg = open_preview()
        pg.select_option("#mode", "sections"); pg.uncheck("#fit")
        pg.fill("#fw", "480"); pg.fill("#fh", "320")
        _wait_traces(pg, 1)
        assert self._plot_width(pg) == pytest.approx(480, abs=2)
        pg.fill("#fw", "720")                                           # the iframe is resized, NOT reloaded
        pg.wait_for_function("""() => { try { const d = document.getElementById('frame').contentDocument
            return Math.abs(d.querySelector('#graph .main-svg').getBoundingClientRect().width - 720) < 3 } catch (_) { return false } }""",
                             timeout=8000)
        assert errors == []


class TestFollow:
    def _ready(self, pv):
        open_preview, errors = pv
        pg = open_preview()
        _wait_traces(pg, 1)
        frame = lambda: [f for f in pg.frames if f != pg.main_frame][-1]
        frame().evaluate("window.__marker = 7")
        return pg, frame, errors

    def test_editing_the_title_inside_the_preview_updates_the_controls(self, pv):
        pg, frame, errors = self._ready(pv)
        frame().click("#pageTitle")                                      # the graph page's own inline editor
        pg.keyboard.type("Beehive Lake")
        pg.keyboard.press("Enter")
        pg.wait_for_function("document.getElementById('title').value === 'Beehive Lake'", timeout=8000)
        assert "title=Beehive" in pg.inner_text("#urlOut")
        assert frame().evaluate("window.__marker") == 7                  # not reloaded
        assert errors == []

    def test_moving_and_resizing_the_title_updates_the_position_controls(self, pv):
        pg, frame, _ = self._ready(pv)
        frame().evaluate("""() => { const u = new URL(location.href)
            u.searchParams.set('titleX', '150'); u.searchParams.set('titleY', '60'); u.searchParams.set('titleSize', '32')
            history.replaceState(null, '', u.toString()) }""")
        pg.wait_for_function("document.getElementById('titleX').value === '150'", timeout=8000)
        assert pg.input_value("#titleY") == "60" and pg.input_value("#titleSize") == "32"
        assert frame().evaluate("window.__marker") == 7

    def test_an_edit_is_never_overwritten_by_a_poll_that_sees_the_old_frame_url(self, pv):
        pg, frame, _ = self._ready(pv)
        frame().evaluate("""() => { const u = new URL(location.href); u.searchParams.set('title', 'moved')
            history.replaceState(null, '', u.toString()) }""")           # the page changed its own URL
        pg.fill("#lat", "47.25")                                         # typing: the reload is DEBOUNCED
        pg.evaluate("window.__graphPreview.pollFrame()"); pg.evaluate("window.__graphPreview.pollFrame()")
        assert pg.evaluate("window.__graphPreview.state.lat") == 47.25   # not thrown away
        pg.wait_for_function("document.getElementById('frame').getAttribute('src').includes('lat=47.25')", timeout=6000)

    def test_with_follow_off_the_controls_are_left_alone(self, pv):
        pg, frame, _ = self._ready(pv)
        pg.uncheck("#follow")
        frame().evaluate("""() => { const u = new URL(location.href); u.searchParams.set('title', 'x')
            history.replaceState(null, '', u.toString()) }""")
        pg.wait_for_timeout(2000)
        assert pg.input_value("#title") == ""


class TestFrameAndOutput:
    @pytest.mark.parametrize("ar,expect", [("", (480, 240)), ("16:9", (480, 270)), ("75%", (480, 360))])
    def test_card_height_follows_home_assistants_aspect_ratio(self, pv, ar, expect):
        open_preview, _ = pv
        pg = open_preview()
        pg.select_option("#mode", "card"); pg.fill("#fw", "480"); pg.select_option("#ar", ar); pg.uncheck("#fit")
        got = pg.evaluate("(() => { const f = document.getElementById('frame'); return [parseInt(f.style.width), parseInt(f.style.height)] })()")
        assert tuple(got) == expect

    def test_yaml_copy_token_and_base(self, pv):
        open_preview, _ = pv
        pg = open_preview("?token=secret123")
        assert "token=secret123" in pg.evaluate("document.getElementById('frame').getAttribute('src')")
        assert "token=secret123" in pg.inner_text("#urlOut")
        pg.uncheck("#useToken")
        assert "token=" not in pg.inner_text("#urlOut")
        pg.fill("#base", "https://weather.example.com/"); pg.press("#base", "Tab")
        assert pg.inner_text("#urlOut").startswith("https://weather.example.com/graph/")
        pg.select_option("#mode", "card"); pg.select_option("#ar", "16:9")
        assert "aspect_ratio: 16:9" in pg.inner_text("#yamlOut")
        pg.click("#copyYaml")
        assert pg.evaluate("navigator.clipboard.readText()").startswith("type: iframe")

    def test_settings_survive_a_reload_of_the_preview(self, pv):
        open_preview, _ = pv
        pg = open_preview()
        pg.select_option("#mode", "sections"); pg.fill("#fw", "500"); pg.fill("#lat", "47.25"); pg.press("#lat", "Tab")
        pg.wait_for_timeout(600)
        url = pg.url
        pg2 = open_preview("", url[url.index("#"):])
        assert pg2.input_value("#mode") == "sections" and pg2.input_value("#fw") == "500" and pg2.input_value("#lat") == "47.25"

    def test_the_navigation_link_to_the_map_preview_keeps_the_token(self, pv):
        open_preview, _ = pv
        pg = open_preview("?token=abc")
        assert pg.get_attribute("a.nav >> nth=0", "href") == "/ha-preview/?token=abc"


class TestSampleData:
    def test_the_frame_uses_sample_data_and_the_outputs_never_mention_it(self, pv):
        open_preview, errors = pv
        pg = open_preview()
        assert pg.is_checked("#fixture")
        assert "fixture=1" in pg.evaluate("document.getElementById('frame').getAttribute('src')")
        assert "fixture" not in pg.inner_text("#urlOut") and "fixture" not in pg.inner_text("#yamlOut")
        _wait_traces(pg, 1)
        api = [(p, q) for p, q in open_preview.seen if p.startswith("/api/models/")]
        assert api and all(q.get("fixture") == "1" for _, q in api), api
        assert errors == []

    def test_the_curves_are_drawn_from_the_sample_series(self, pv):
        open_preview, _ = pv
        pg = open_preview()
        _wait_traces(pg, 1)
        n, uom = pg.evaluate("""() => { const g = document.getElementById('frame').contentDocument.getElementById('graph')
            return [g.data[0].x.length, g.data[0].name] }""")
        assert n == 72 * 8 + 1 and "degF" in uom                     # 72 h at the default 450 s step

    def test_turning_it_off_drops_the_flag_from_the_frame(self, pv):
        open_preview, _ = pv
        pg = open_preview()
        pg.uncheck("#fixture")
        pg.wait_for_function("!document.getElementById('frame').getAttribute('src').includes('fixture')")
