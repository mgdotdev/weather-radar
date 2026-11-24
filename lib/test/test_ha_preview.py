"""The /ha-preview/ page: one control per map URL parameter, a frame of the
chosen Home Assistant card size, the resulting URL + card YAML, and "follow"
(changes made inside the preview flow back into the controls).

Run in a REAL browser (headless Chromium via Playwright) against the real page
and the real map inside its iframe; only the data API is stubbed. Skipped unless
Playwright + Chromium + the map-library CDN are available."""
import json
import time
from urllib.parse import urlparse, parse_qs, quote

import pytest

pw = pytest.importorskip("playwright.sync_api")
pytest.importorskip("jinja2")
pytest.importorskip("fastapi")

import test_page_smoke as S                                   # noqa: E402  (shared constants)
from fixture_harness import serve_api, map_html              # noqa: E402

ORIGIN = S.ORIGIN
AQI_URL = f"/map/models/{S.MODEL}/?{S.QUERY}"                 # 6 pins, pinBoxes, legend position


@pytest.fixture(scope="module")
def preview_html():
    import weather_radar.server.app as app
    return app.ha_preview()


@pytest.fixture
def pv(preview_html):
    errors = []
    with pw.sync_playwright() as p:
        try:
            browser = p.chromium.launch(args=["--no-sandbox", "--use-gl=angle", "--use-angle=swiftshader",
                                              "--ignore-gpu-blocklist"])
        except Exception as e:
            pytest.skip(f"chromium unavailable: {e}")
        ctx = browser.new_context(viewport={"width": 1500, "height": 950})
        ctx.grant_permissions(["clipboard-read", "clipboard-write"], origin=ORIGIN)

        def handle(route):
            url = route.request.url
            path = urlparse(url).path
            if path.startswith("/ha-preview"):
                return route.fulfill(body=preview_html, content_type="text/html; charset=utf-8")
            if path.startswith("/map/models/"):
                return route.fulfill(body=map_html(url), content_type="text/html; charset=utf-8")
            return serve_api(route, open_preview.seen)
        ctx.route(f"{ORIGIN}/**", handle)

        def open_preview(query="", hash_=""):
            pg = ctx.new_page()
            pg.on("pageerror", lambda e: errors.append(str(e)))
            pg.goto(f"{ORIGIN}/ha-preview/{query}{hash_}", wait_until="load")
            pg.wait_for_function("window.__haPreview !== undefined")
            # (the advanced sections start collapsed; a user would open them)
            pg.evaluate("document.querySelectorAll('details').forEach(d => d.open = true)")
            return pg
        open_preview.seen = []                      # every API call the frames make
        yield open_preview, errors
        browser.close()


_FRAME_URL_JS = """(() => { const f = document.getElementById('frame')
    try { const l = f.contentWindow.location; if (/\\/map\\/models\\//.test(l.pathname)) return l.pathname + l.search } catch (_) {}
    return f.getAttribute('src') || '' })()"""


def _src_params(pg):
    """The URL the frame is SHOWING (a live change updates it without reloading, so the src
    attribute can be stale), without the sample-data flag."""
    src = pg.evaluate(_FRAME_URL_JS)
    u = urlparse(src)
    q = {k: v[0] for k, v in parse_qs(u.query, keep_blank_values=True).items()}
    q.pop("fixture", None)
    return u.path, q


def _wait_src(pg, predicate_js, timeout=8000):
    pg.wait_for_function("(a) => { const src = " + _FRAME_URL_JS + "; return new Function('src', 'return ' + a)(src) }",
                         arg=predicate_js, timeout=timeout)


class TestRoute:
    def test_page_renders_with_models_defaults_and_no_leftover_template_syntax(self, preview_html):
        assert "{{" not in preview_html and "{%" not in preview_html
        boot = json.loads(preview_html.split("const BOOT = ")[1].split("\n")[0])
        assert {"value": "air-quality-pm25", "label": "Air Quality (PM2.5)"} in boot["models"]
        assert boot["defaults"]["dt"] == 3600 and boot["dts"][0]["value"] == 300
        assert boot["defaultModel"] == "temperature"


class TestControls:
    def test_loads_cleanly_and_shows_the_default_map(self, pv):
        open_preview, errors = pv
        pg = open_preview()
        path, q = _src_params(pg)
        assert path == "/map/models/temperature/" and {"lat", "lon", "width", "height", "dt"} <= set(q)
        assert "speed" not in q and "pinSize" not in q and "locked" not in q   # defaults are omitted
        assert "type: iframe" in pg.inner_text("#yamlOut")
        assert errors == []

    @pytest.mark.parametrize("control,value,param,expected", [
        ("lat", "47.5", "lat", "47.5"), ("lon", "-116.25", "lon", "-116.25"),
        ("width", "150", "width", "150.00"), ("height", "120.5", "height", "120.50"),
        ("speed", "2", "speed", "2.00"), ("zoom", "7.5", "zoom", "7.50"),
        ("mapLat", "47.9", "mapLat", "47.9"), ("mapLon", "-115.9", "mapLon", "-115.9"),
        ("pinBoxes", ";;-10,5", "pinBoxes", ";;-10,5"),
        ("legendLat", "47.1", "legendLat", "47.1"), ("legendLon", "-115.5", "legendLon", "-115.5"),
        ("from", "1795200000", "from", "1795200000"), ("to", "1795400000", "to", "1795400000"),
        ("t", "1795300000", "t", "1795300000"),
    ])
    def test_each_number_or_text_control_writes_its_parameter(self, pv, control, value, param, expected):
        open_preview, errors = pv
        pg = open_preview()
        pg.fill(f"#{control}", value); pg.press(f"#{control}", "Tab")
        _wait_src(pg, f"src.includes('{param}=')")
        assert _src_params(pg)[1][param] == expected
        assert param + "=" in pg.inner_text("#urlOut")

    def test_selects_and_ranges_and_checkboxes(self, pv):
        open_preview, errors = pv
        pg = open_preview()
        pg.select_option("#model", "air-quality-pm25")
        pg.select_option("#dt", "900")
        pg.evaluate("""() => { for (const [id, v] of [['pinSize', 1.5], ['legendSize', 0.75]]) {
            const el = document.getElementById(id); el.value = v
            el.dispatchEvent(new Event('input', {bubbles: true})); el.dispatchEvent(new Event('change', {bubbles: true})) } }""")
        pg.uncheck("#areaBox"); pg.uncheck("#cursorBox"); pg.check("#locked"); pg.check("#floatScrubber"); pg.check("#attribution")
        _wait_src(pg, "src.includes('locked=1') && src.includes('legendSize') && src.includes('floatScrubber') && src.includes('attribution=1')")
        path, q = _src_params(pg)
        assert path == "/map/models/air-quality-pm25/"
        assert (q["dt"], q["pinSize"], q["legendSize"]) == ("900", "1.5", "0.75")
        assert (q["areaBox"], q["cursorBox"], q["locked"], q["floatScrubber"]) == ("0", "0", "1", "1")
        pg.check("#areaBox"); pg.check("#cursorBox"); pg.uncheck("#locked"); pg.uncheck("#floatScrubber"); pg.uncheck("#attribution")
        pg.evaluate("""() => { const el = document.getElementById('pinSize'); el.value = 1
            el.dispatchEvent(new Event('change', {bubbles: true})) }""")
        # wait for the WHOLE end state: with live apply each setting lands on its own
        _wait_src(pg, "!['areaBox', 'cursorBox', 'locked', 'pinSize', 'floatScrubber', 'attribution']"
                      ".some(k => new URL(src, location.origin).searchParams.has(k))")
        q = _src_params(pg)[1]
        assert not ({"areaBox", "cursorBox", "locked", "pinSize", "floatScrubber", "attribution"} & set(q))   # back to defaults: omitted
        assert errors == []

    def test_compact_preset(self, pv):
        open_preview, _ = pv
        pg = open_preview()
        pg.click("#compact")
        _wait_src(pg, "src.includes('locked=1')")
        q = _src_params(pg)[1]
        assert (q["legendSize"], q["pinSize"], q["locked"]) == ("0.5", "0.5", "1")

    def test_an_unusual_time_step_is_kept_not_blanked(self, pv):
        open_preview, _ = pv
        pg = open_preview()
        pg.fill("#importUrl", f"{ORIGIN}/map/models/temperature/?lat=1&lon=2&width=3&height=4&dt=450")
        pg.click("#importBtn")
        assert pg.input_value("#dt") == "450"
        assert _src_params(pg)[1]["dt"] == "450"

    def test_unknown_parameters_are_preserved(self, pv):
        open_preview, _ = pv
        pg = open_preview()
        pg.fill("#importUrl", f"{ORIGIN}/map/models/temperature/?lat=1&lon=2&width=3&height=4&dt=3600&render=heatmap&step=2")
        pg.click("#importBtn")
        pg.fill("#lat", "9"); pg.press("#lat", "Tab")
        _wait_src(pg, "src.includes('lat=9')")
        q = _src_params(pg)[1]
        assert q["render"] == "heatmap" and q["step"] == "2"


class TestPins:
    def test_pins_added_in_the_editor_are_restored_by_the_real_map(self, pv):
        open_preview, errors = pv
        pg = open_preview("?u=" + quote(ORIGIN + AQI_URL.split('&pins=')[0], safe=""))
        pg.click("#addPin"); pg.click("#addPin")
        rows = pg.locator("#pins .pin")
        assert rows.count() == 2
        rows.nth(0).locator(".name").fill("Coeur d'Alene"); rows.nth(0).locator(".plat").fill("47.67359")
        rows.nth(0).locator(".plon").fill("-116.78221")
        rows.nth(1).locator(".name").fill("Beehive Lake & More")
        rows.nth(1).locator(".plat").fill("48.65516"); rows.nth(1).locator(".plon").fill("-116.65461")
        pg.press("#pins .pin >> nth=1 >> .name", "Tab")
        _wait_src(pg, "src.includes('pins=')")
        # Wait for the CONDITION (the real map restored both pins, names intact):
        # several quick edits each reload the frame, and pins appear one by one.
        pg.wait_for_function("""() => { try {
            const d = document.getElementById('frame').contentDocument
            const t = [...d.querySelectorAll('#pins > div')].map(x => x.textContent).join(' ')
            return t.includes("Coeur d'Alene") && t.includes('Beehive Lake & More') } catch (_) { return false } }""",
            timeout=45000)
        assert errors == []

    def test_deleting_a_pin_drops_its_box_offset_slot(self, pv):
        open_preview, _ = pv
        pg = open_preview()
        pg.fill("#importUrl", f"{ORIGIN}/map/models/temperature/?lat=1&lon=2&width=3&height=4&dt=3600"
                              "&pins=1,2,A;3,4,B;5,6,C&pinBoxes=;;-10,5")
        pg.click("#importBtn")
        pg.locator("#pins .pin").nth(0).locator(".x").click()           # delete pin A
        assert pg.locator("#pins .pin").count() == 2
        assert pg.input_value("#pinBoxes") == ";-10,5"                  # C's offset stays with C

    def test_import_of_a_real_url_fills_every_control(self, pv):
        open_preview, errors = pv
        pg = open_preview()
        pg.fill("#importUrl", ORIGIN + AQI_URL); pg.click("#importBtn")
        assert pg.input_value("#model") == S.MODEL
        assert float(pg.input_value("#width")) == pytest.approx(201.9)
        assert pg.input_value("#dt") == "900"
        assert pg.locator("#pins .pin").count() == 6
        assert pg.locator("#pins .pin").nth(1).locator(".name").input_value() == "Beehive Lake"
        assert pg.input_value("#pinBoxes") == ";;-203,34;;-187,-27;-201,-9"
        assert float(pg.input_value("#legendLat")) == pytest.approx(47.79725)
        # and what it writes back means the same thing
        _, q = _src_params(pg)
        mine = json.loads(pg.evaluate("JSON.stringify(window.__haPreview.state.pins)"))
        assert [(round(p['lat'], 5), round(p['lon'], 5), p['name']) for p in mine][:2] == [
            (47.77877, -116.70459, "Home"), (48.65516, -116.65461, "Beehive Lake")]
        assert q["pinBoxes"] == ";;-203,34;;-187,-27;-201,-9" and q["legendLat"] == "47.79725"
        assert errors == []


class TestFollow:
    def _ready(self, pv, follow=True):
        open_preview, errors = pv
        pg = open_preview("?u=" + quote(ORIGIN + AQI_URL, safe=""))
        frame = lambda: [f for f in pg.frames if f != pg.main_frame][-1]
        frame().wait_for_selector("#legend", state="visible", timeout=30000)
        pg.wait_for_timeout(1500)
        assert pg.locator("#pins .pin").count() == 6               # the whole URL was loaded, not a truncation
        if not follow:
            pg.uncheck("#follow")
        frame().evaluate("window.__marker = 123")
        return pg, frame, errors

    def test_changes_inside_the_preview_update_the_controls_without_reloading(self, pv):
        pg, frame, errors = self._ready(pv)
        frame().evaluate("""() => { const u = new URL(location.href)
            u.searchParams.set('legendLat', '47.5'); u.searchParams.set('legendLon', '-115.5')
            u.searchParams.set('zoom', '9.25'); u.searchParams.set('pinSize', '2')
            history.replaceState(null, '', u.toString()) }""")
        pg.wait_for_function("document.getElementById('legendLat').value === '47.5'", timeout=6000)
        assert pg.input_value("#zoom") == "9.25" and float(pg.input_value("#pinSize")) == 2
        assert "legendLat=47.5" in pg.inner_text("#urlOut") and "zoom=9.25" in pg.inner_text("#urlOut")
        assert frame().evaluate("window.__marker") == 123          # same document: NOT reloaded
        assert errors == []

    def test_with_follow_off_the_controls_are_left_alone(self, pv):
        pg, frame, _ = self._ready(pv, follow=False)
        before = pg.input_value("#legendLat")
        frame().evaluate("""() => { const u = new URL(location.href); u.searchParams.set('legendLat', '41.5')
            history.replaceState(null, '', u.toString()) }""")
        pg.wait_for_timeout(2000)
        assert pg.input_value("#legendLat") == before

    def test_an_edit_is_never_overwritten_by_a_poll_that_sees_the_old_frame_url(self, pv):
        """The race: the map changes its own URL, you edit a control, and a poll
        runs BEFORE the debounced reload — it used to adopt the frame's old URL
        and silently drop the edit. Poll explicitly in that window."""
        pg, frame, errors = self._ready(pv)
        frame().evaluate("""() => { const u = new URL(location.href); u.searchParams.set('zoom', '8.5')
            history.replaceState(null, '', u.toString()) }""")           # the map moved on its own
        pg.fill("#lat", "47.25")              # typing: an `input` event -> the reload is DEBOUNCED
        pg.evaluate("window.__haPreview.pollFrame()")                     # a poll lands before it fires
        pg.evaluate("window.__haPreview.pollFrame()")
        assert pg.evaluate("window.__haPreview.state.lat") == 47.25       # the edit was not thrown away
        pg.wait_for_function("document.getElementById('frame').getAttribute('src').includes('lat=47.25')",
                             timeout=6000)
        assert errors == []

    def test_editing_a_control_while_following_wins_and_reloads_once(self, pv):
        pg, frame, _ = self._ready(pv)
        pg.fill("#lat", "47.0"); pg.press("#lat", "Tab")
        pg.wait_for_function("document.getElementById('frame').getAttribute('src').includes('lat=47')")
        pg.wait_for_timeout(2500)
        assert pg.input_value("#lat") == "47"                       # not clobbered by the old URL
        assert frame().evaluate("window.__marker") is None          # reloaded with the new URL


class TestFrameAndOutput:
    @pytest.mark.parametrize("mode,fw,ar,expect", [
        ("card", "480", "", (480, 240)), ("card", "480", "16:9", (480, 270)),
        ("card", "480", "75%", (480, 360)), ("card", "600", "1:1", (600, 600)),
    ])
    def test_card_height_follows_home_assistants_aspect_ratio_rules(self, pv, mode, fw, ar, expect):
        open_preview, _ = pv
        pg = open_preview()
        pg.select_option("#mode", mode); pg.fill("#fw", fw); pg.select_option("#ar", ar)
        pg.uncheck("#fit")
        got = pg.evaluate("(() => { const f = document.getElementById('frame'); return [parseInt(f.style.width), parseInt(f.style.height)] })()")
        assert tuple(got) == expect

    def test_other_card_types_and_yaml(self, pv):
        open_preview, _ = pv
        pg = open_preview()
        pg.select_option("#mode", "card"); pg.select_option("#ar", "16:9")
        assert "aspect_ratio: 16:9" in pg.inner_text("#yamlOut")
        pg.select_option("#mode", "sections")
        assert "aspect_ratio:" not in pg.inner_text("#yamlOut") and "grid" in pg.inner_text("#yamlOut")
        pg.select_option("#mode", "device"); pg.select_option("#dev", "390x844"); pg.uncheck("#fit")
        assert pg.evaluate("document.getElementById('frame').style.height") == "844px"
        pg.select_option("#mode", "panel")
        assert "Panel" in pg.inner_text("#yamlOut")

    def test_copy_buttons_put_the_output_on_the_clipboard(self, pv):
        open_preview, _ = pv
        pg = open_preview()
        pg.click("#copyUrl")
        assert pg.evaluate("navigator.clipboard.readText()") == pg.inner_text("#urlOut")
        pg.click("#copyYaml")
        assert pg.evaluate("navigator.clipboard.readText()").startswith("type: iframe")

    def test_token_is_forwarded_to_the_frame_and_optional_in_the_output(self, pv):
        open_preview, _ = pv
        pg = open_preview("?token=secret123")
        assert "token=secret123" in pg.evaluate("document.getElementById('frame').getAttribute('src')")
        assert "token=secret123" in pg.inner_text("#urlOut")            # Home Assistant needs it
        pg.uncheck("#useToken")
        assert "token=" not in pg.inner_text("#urlOut")
        assert "token=" in pg.evaluate("document.getElementById('frame').getAttribute('src')")  # frame keeps working

    def test_base_url_can_be_changed_for_the_dashboard(self, pv):
        open_preview, _ = pv
        pg = open_preview()
        pg.fill("#base", "https://weather.example.com/"); pg.press("#base", "Tab")
        assert pg.inner_text("#urlOut").startswith("https://weather.example.com/map/models/")

    def test_settings_survive_a_reload_of_the_preview(self, pv):
        open_preview, _ = pv
        pg = open_preview()
        pg.select_option("#mode", "sections"); pg.fill("#fw", "500"); pg.fill("#lat", "47.25"); pg.press("#lat", "Tab")
        pg.wait_for_timeout(600)
        url = pg.url
        pg2 = open_preview("", url[url.index("#"):])
        assert pg2.input_value("#mode") == "sections" and pg2.input_value("#fw") == "500"
        assert pg2.input_value("#lat") == "47.25"


def _ready_map(pv, url=None):
    """Open the preview on the full AQI map, wait for the map AND its live hooks, and
    plant a marker in the frame's window (it survives only if the frame is NOT reloaded)."""
    open_preview, errors = pv
    pg = open_preview("?u=" + quote(ORIGIN + (url or AQI_URL), safe=""))
    frame = lambda: [f for f in pg.frames if f != pg.main_frame][-1]
    frame().wait_for_selector("#legend", state="visible", timeout=30000)
    pg.wait_for_function("(() => { try { return !!document.getElementById('frame').contentWindow.WRLive } catch (e) { return false } })()",
                         timeout=30000)
    pg.wait_for_function("document.querySelectorAll('#pins .pin').length >= 6")          # (editor rows: the URL was loaded)
    pg.wait_for_timeout(600)                                                          # let the frame's load event settle
    frame().evaluate("window.__marker = 99")
    return pg, frame, errors, open_preview


def _slider(pg, element_id, value):
    pg.evaluate("""([id, v]) => { const el = document.getElementById(id); el.value = v
        el.dispatchEvent(new Event('input', {bubbles: true})); el.dispatchEvent(new Event('change', {bubbles: true})) }""",
                [element_id, value])


class TestSampleData:
    def test_the_frame_asks_for_sample_data_but_the_outputs_never_mention_it(self, pv):
        open_preview, errors = pv
        pg = open_preview()
        assert pg.is_checked("#fixture")
        assert "fixture=1" in pg.evaluate("document.getElementById('frame').getAttribute('src')")
        assert "fixture" not in pg.inner_text("#urlOut") and "fixture" not in pg.inner_text("#yamlOut")
        assert "fixture" not in json.dumps(pg.evaluate("window.__haPreview.state"))
        assert errors == []

    def test_every_api_call_the_frame_makes_is_a_sample_data_call(self, pv):
        open_preview, _ = pv
        pg = open_preview("?u=" + quote(ORIGIN + AQI_URL, safe=""))
        pg.wait_for_function("document.querySelectorAll('#pins .pin').length >= 6")
        pg.wait_for_function("window.__haPreview !== undefined")
        [f for f in pg.frames if f != pg.main_frame][-1].wait_for_selector("#legend", state="visible", timeout=30000)
        pg.wait_for_timeout(1500)
        api = [(p, q) for p, q in open_preview.seen if p.startswith("/api/models/")]
        assert api and all(q.get("fixture") == "1" for _, q in api), api

    def test_a_non_aqi_layer_renders_end_to_end_from_sample_data(self, pv):
        """The default (NBM-style) path — prefetch, range and the series stream — through the
        real map page, with nothing but sample data behind it."""
        open_preview, errors = pv
        pg = open_preview("?u=" + quote(ORIGIN + "/map/models/temperature/?lat=47.7&lon=-116.7&width=150&height=120&dt=3600"
                                       "&pins=47.7%2C-116.7%2CHome", safe=""))
        pg.wait_for_function("""() => { try { const d = document.getElementById('frame').contentDocument
            return [...d.querySelectorAll('#pins > div')].some(x => /Temperature: *-?[0-9]/.test(x.textContent)) } catch (e) { return false } }""",
                             timeout=45000)
        paths = {p for p, _ in open_preview.seen}
        assert {"/api/models/prefetch/stream", "/api/models/temperature/type/range",
                "/api/models/temperature/type/series/stream"} <= paths
        assert errors == []

    def test_turning_it_off_reloads_without_the_flag_and_it_is_remembered(self, pv):
        open_preview, _ = pv
        pg = open_preview()
        pg.uncheck("#fixture")
        pg.wait_for_function("!document.getElementById('frame').getAttribute('src').includes('fixture')")
        assert "fixture" not in pg.inner_text("#urlOut")
        pg.wait_for_timeout(500)
        url = pg.url
        pg2 = open_preview("", url[url.index("#"):])
        assert not pg2.is_checked("#fixture")
        pg2.check("#fixture")
        pg2.wait_for_function("document.getElementById('frame').getAttribute('src').includes('fixture=1')")

    def test_the_flag_never_leaks_into_the_state_through_follow(self, pv):
        pg, frame, errors, _ = _ready_map(pv)
        pg.wait_for_timeout(1500)                                    # a few follow polls
        assert "fixture" not in json.dumps(pg.evaluate("window.__haPreview.state"))
        assert "fixture" not in pg.inner_text("#urlOut")


class TestLiveApply:
    """Purely visual settings are applied INSIDE the loaded map — no reload, no data."""

    def _no_reload_no_data(self, pg, frame, open_preview, n_before):
        assert frame().evaluate("window.__marker") == 99                 # the same document: not reloaded
        assert len(open_preview.seen) == n_before                        # and not a single data request

    def test_pin_size_changes_instantly_with_no_reload_and_no_data_fetch(self, pv):
        pg, frame, errors, op = _ready_map(pv)
        n = len(op.seen)
        t0 = time.time()
        _slider(pg, "pinSize", 2)
        frame().wait_for_function("""() => { const b = [...document.querySelectorAll('#pins > div')].filter(
            d => getComputedStyle(d).width === '190px')[0]; return !!b && Math.abs(b.getBoundingClientRect().width - 380) < 2 }""",
                                  timeout=3000)
        assert time.time() - t0 < 3
        pg.wait_for_timeout(1200)                                         # long enough for any (wrong) debounced reload
        self._no_reload_no_data(pg, frame, op, n)
        assert _src_params(pg)[1]["pinSize"] == "2"                       # the frame's URL followed
        assert "pinSize=2" in pg.inner_text("#urlOut")
        assert errors == []

    def test_the_maps_own_url_rewrites_do_not_force_a_reload(self, pv):
        """The map adds `speed=1.00` (and other defaults) to its own URL after it loads;
        that must not count as a change that needs a reload."""
        pg, frame, _, op = _ready_map(pv)
        assert "speed=1.00" in frame().evaluate("location.search")
        n = len(op.seen)
        _slider(pg, "pinSize", 1.25)
        pg.wait_for_timeout(1500)
        self._no_reload_no_data(pg, frame, op, n)

    def test_legend_size_changes_instantly(self, pv):
        pg, frame, _, op = _ready_map(pv)
        n = len(op.seen)
        _slider(pg, "legendSize", 0.5)
        frame().wait_for_function("Math.abs(document.getElementById('legend').getBoundingClientRect().width - 180) < 2", timeout=3000)
        pg.wait_for_timeout(800)
        self._no_reload_no_data(pg, frame, op, n)

    def test_dragging_a_slider_applies_every_step_live(self, pv):
        pg, frame, errors, op = _ready_map(pv)
        n = len(op.seen)
        for v in (0.6, 0.7, 0.8, 0.9, 1.1, 1.3, 1.5):
            _slider(pg, "pinSize", v)
        frame().wait_for_function("""() => { const b = [...document.querySelectorAll('#pins > div')].filter(
            d => getComputedStyle(d).width === '190px')[0]; return !!b && Math.abs(b.getBoundingClientRect().width - 285) < 2 }""", timeout=3000)
        pg.wait_for_timeout(800)
        self._no_reload_no_data(pg, frame, op, n)
        assert errors == []

    @pytest.mark.parametrize("control,action,check_js", [
        ("locked", "check", "document.getElementById('lockLayout').checked === true"),
        ("cursorBox", "uncheck", "document.getElementById('showCursorBox').checked === false"),
        ("areaBox", "uncheck", "document.getElementById('showAreaBox').checked === false"),
        ("floatScrubber", "check", "!!document.getElementById('timeScrubber').closest('#scrubberFloatBar')"),
        ("attribution", "check", "!!document.querySelector('.maplibregl-ctrl-attrib')"),
        ("scaleBar", "uncheck", "!document.querySelector('.maplibregl-ctrl-scale') && document.getElementById('showScaleBar').checked === false"),
    ])
    def test_each_visual_toggle_applies_in_place_and_keeps_the_frames_own_controls_in_step(self, pv, control, action, check_js):
        pg, frame, errors, op = _ready_map(pv)
        n = len(op.seen)
        getattr(pg, action)(f"#{control}")
        frame().wait_for_function(check_js, timeout=3000)
        pg.wait_for_timeout(800)
        self._no_reload_no_data(pg, frame, op, n)
        assert errors == []

    def test_turning_a_toggle_back_off_also_works_live(self, pv):
        pg, frame, _, op = _ready_map(pv)
        n = len(op.seen)
        pg.check("#floatScrubber")
        frame().wait_for_function("!!document.getElementById('timeScrubber').closest('#scrubberFloatBar')", timeout=3000)
        pg.uncheck("#floatScrubber")
        frame().wait_for_function("!!document.getElementById('timeScrubber').closest('#legend')", timeout=3000)
        assert "floatScrubber" not in _src_params(pg)[1]
        self._no_reload_no_data(pg, frame, op, n)

    def test_the_compact_preset_applies_live(self, pv):
        pg, frame, _, op = _ready_map(pv)
        n = len(op.seen)
        pg.click("#compact")
        frame().wait_for_function("Math.abs(document.getElementById('legend').getBoundingClientRect().width - 180) < 2", timeout=3000)
        pg.wait_for_timeout(800)
        self._no_reload_no_data(pg, frame, op, n)
        q = _src_params(pg)[1]
        assert (q["legendSize"], q["pinSize"], q["locked"]) == ("0.5", "0.5", "1")

    def test_a_data_parameter_still_reloads(self, pv):
        pg, frame, _, op = _ready_map(pv)
        pg.fill("#lat", "47.0"); pg.press("#lat", "Tab")
        pg.wait_for_function("window.__haPreview.state.lat === 47")
        pg.wait_for_function("document.getElementById('frame').getAttribute('src').includes('lat=47')", timeout=8000)
        pg.wait_for_timeout(2500)
        assert frame().evaluate("window.__marker") is None               # a fresh document

    def test_a_mixed_change_reloads_so_nothing_is_half_applied(self, pv):
        pg, frame, _, op = _ready_map(pv)
        pg.fill("#importUrl", ORIGIN + AQI_URL.replace("dt=900", "dt=900&pinSize=1.5") .replace("lat=48.145506", "lat=47.9"))
        pg.click("#importBtn")
        pg.wait_for_timeout(3000)
        assert frame().evaluate("window.__marker") is None
        assert _src_params(pg)[1]["pinSize"] == "1.5" and _src_params(pg)[1]["lat"] == "47.9"

    def test_without_the_hooks_it_falls_back_to_a_reload(self, pv):
        pg, frame, _, op = _ready_map(pv)
        frame().evaluate("delete window.WRLive")                          # e.g. an older page
        _slider(pg, "pinSize", 2)
        pg.wait_for_function("document.getElementById('frame').getAttribute('src').includes('pinSize=2')", timeout=8000)
        pg.wait_for_timeout(2500)
        assert frame().evaluate("window.__marker") is None
        assert _src_params(pg)[1]["pinSize"] == "2"

    def test_follow_does_not_undo_a_live_change(self, pv):
        pg, frame, _, op = _ready_map(pv)
        _slider(pg, "pinSize", 1.75)
        frame().wait_for_function("""() => { const b = [...document.querySelectorAll('#pins > div')].filter(
            d => getComputedStyle(d).width === '190px')[0]; return !!b && Math.abs(b.getBoundingClientRect().width - 332.5) < 2 }""", timeout=3000)
        for _ in range(3):
            pg.evaluate("window.__haPreview.pollFrame()")
            pg.wait_for_timeout(400)
        assert pg.evaluate("window.__haPreview.state.pinSize") == 1.75
        assert float(pg.input_value("#pinSize")) == 1.75

    def test_a_change_made_inside_the_frame_still_flows_back_after_live_changes(self, pv):
        pg, frame, _, op = _ready_map(pv)
        _slider(pg, "pinSize", 2)
        frame().wait_for_function("""() => { const b = [...document.querySelectorAll('#pins > div')].filter(
            d => getComputedStyle(d).width === '190px')[0]; return !!b && Math.abs(b.getBoundingClientRect().width - 380) < 2 }""", timeout=3000)
        frame().evaluate("""() => { const u = new URL(location.href); u.searchParams.set('legendLat', '47.5')
            history.replaceState(null, '', u.toString()) }""")
        pg.wait_for_function("document.getElementById('legendLat').value === '47.5'", timeout=6000)
        assert frame().evaluate("window.__marker") == 99


class TestViewCenter:
    """The view center (mapLat/mapLon): moves the map in place, and follows panning."""

    NO_PINS = "/map/models/air-quality-pm25/?lat=48.145506&lon=-116.391332&width=201.90&height=183.69&dt=900"

    def _ready(self, pv):
        open_preview, errors = pv
        pg = open_preview("?u=" + quote(ORIGIN + self.NO_PINS, safe=""))
        pg.select_option("#mode", "sections"); pg.fill("#fw", "900"); pg.fill("#fh", "600"); pg.uncheck("#fit")
        frame = lambda: [f for f in pg.frames if f != pg.main_frame][-1]
        frame().wait_for_selector("#legend", state="visible", timeout=30000)
        pg.wait_for_function("(() => { try { return !!document.getElementById('frame').contentWindow.WRLive } catch (e) { return false } })()", timeout=30000)
        pg.wait_for_timeout(800)
        frame().evaluate("window.__marker = 99")
        return pg, frame, errors, open_preview

    def test_typing_a_view_center_moves_the_map_in_place(self, pv):
        pg, frame, errors, op = self._ready(pv)
        n = len(op.seen)
        pg.fill("#mapLat", "47.9"); pg.fill("#mapLon", "-115.9"); pg.press("#mapLon", "Tab")
        frame().wait_for_function("(() => { const v = window.WRLive.view(); return Math.abs(v.lat - 47.9) < 1e-3 && Math.abs(v.lon + 115.9) < 1e-3 })()", timeout=4000)
        pg.wait_for_timeout(1000)
        assert frame().evaluate("window.__marker") == 99 and len(op.seen) == n        # no reload, no data
        q = _src_params(pg)[1]
        assert (float(q["mapLat"]), float(q["mapLon"])) == (47.9, -115.9)             # (the map writes 47.90000; same number)
        assert "mapLat=47.9" in pg.inner_text("#urlOut") and q["lat"] == "48.145506"   # the data area is unchanged
        assert errors == []

    def test_one_field_moves_just_that_axis_and_blank_means_the_area_center(self, pv):
        pg, frame, _, _ = self._ready(pv)
        pg.fill("#mapLat", "47.9"); pg.press("#mapLat", "Tab")
        frame().wait_for_function("Math.abs(window.WRLive.view().lat - 47.9) < 1e-3", timeout=4000)
        assert abs(frame().evaluate("window.WRLive.view().lon") + 116.391332) < 1e-3
        pg.fill("#mapLat", ""); pg.press("#mapLat", "Tab")
        frame().wait_for_function("Math.abs(window.WRLive.view().lat - 48.145506) < 1e-3", timeout=4000)
        frame().wait_for_function("!new URL(location.href).searchParams.has('mapLat')", timeout=4000)   # back on the area center

    def test_panning_the_map_inside_the_preview_fills_the_controls_as_you_go(self, pv):
        pg, frame, errors, op = self._ready(pv)
        box = pg.evaluate("(() => { const b = document.getElementById('frame').getBoundingClientRect(); return {x: b.left, y: b.top} })()")
        x, y = box["x"] + 650, box["y"] + 420                                           # empty map area inside the frame
        pg.mouse.move(x, y); pg.mouse.down()
        for i in range(1, 11):
            pg.mouse.move(x - 12 * i, y + 8 * i)
        pg.wait_for_function("document.getElementById('mapLat').value !== ''", timeout=6000)   # while still dragging
        pg.mouse.up()
        # wait for the glide to end AND the control to show where it stopped (a fixed sleep races the inertia)
        pg.wait_for_function("""() => { const v = document.getElementById('frame').contentWindow.WRLive.view()
            return Math.abs(parseFloat(document.getElementById('mapLat').value) - v.lat) < 2e-4
                && Math.abs(parseFloat(document.getElementById('mapLon').value) - v.lon) < 2e-4 }""", timeout=10000, polling=150)
        pg.wait_for_timeout(700)                                                          # (and it stays put)
        v = frame().evaluate("window.WRLive.view()")
        assert float(pg.input_value("#mapLat")) == pytest.approx(v["lat"], abs=2e-4)
        assert float(pg.input_value("#mapLon")) == pytest.approx(v["lon"], abs=2e-4)
        assert "mapLat=" in pg.inner_text("#urlOut")                                     # ready to copy into Home Assistant
        assert frame().evaluate("window.__marker") == 99                                 # and nothing reloaded
        assert errors == []

    def test_reset_layout_clears_the_view_center(self, pv):
        pg, frame, _, _ = self._ready(pv)
        pg.fill("#mapLat", "47.9"); pg.fill("#mapLon", "-115.9"); pg.press("#mapLon", "Tab")
        frame().wait_for_function("Math.abs(window.WRLive.view().lat - 47.9) < 1e-3", timeout=4000)
        pg.click("#resetView")
        assert pg.input_value("#mapLat") == "" and pg.input_value("#mapLon") == ""
        frame().wait_for_function("Math.abs(window.WRLive.view().lat - 48.145506) < 1e-3", timeout=6000)

    def test_importing_a_url_with_a_view_center_fills_the_fields(self, pv):
        open_preview, _ = pv
        pg = open_preview()
        pg.fill("#importUrl", ORIGIN + self.NO_PINS + "&mapLat=47.9&mapLon=-115.9"); pg.click("#importBtn")
        assert (pg.input_value("#mapLat"), pg.input_value("#mapLon")) == ("47.9", "-115.9")
        assert _src_params(pg)[1]["mapLat"] == "47.9"


class TestZoom:
    """The zoom field zooms the map in place; zooming inside the preview fills the field."""

    NO_PINS = TestViewCenter.NO_PINS

    def _ready(self, pv):
        return TestViewCenter()._ready(pv)

    def test_typing_a_zoom_zooms_the_map_in_place(self, pv):
        pg, frame, errors, op = self._ready(pv)
        n = len(op.seen)
        pg.fill("#zoom", "9.5"); pg.press("#zoom", "Tab")
        frame().wait_for_function("Math.abs(window.WRLive.view().zoom - 9.5) < 1e-3", timeout=4000)
        pg.wait_for_timeout(1000)
        assert frame().evaluate("window.__marker") == 99 and len(op.seen) == n         # no reload, no data
        assert float(_src_params(pg)[1]["zoom"]) == 9.5
        assert "zoom=9.50" in pg.inner_text("#urlOut")
        assert errors == []

    def test_every_step_of_a_number_spinner_applies_live(self, pv):
        pg, frame, errors, op = self._ready(pv)
        n = len(op.seen)
        for z in ("8", "8.5", "9", "9.5", "10"):
            pg.fill("#zoom", z)
        frame().wait_for_function("Math.abs(window.WRLive.view().zoom - 10) < 1e-3", timeout=4000)
        pg.wait_for_timeout(800)
        assert frame().evaluate("window.__marker") == 99 and len(op.seen) == n
        assert errors == []

    def test_clearing_the_zoom_returns_to_the_default(self, pv):
        pg, frame, _, _ = self._ready(pv)
        pg.fill("#zoom", "9.5"); pg.press("#zoom", "Tab")
        frame().wait_for_function("Math.abs(window.WRLive.view().zoom - 9.5) < 1e-3", timeout=4000)
        pg.fill("#zoom", ""); pg.press("#zoom", "Tab")
        frame().wait_for_function("Math.abs(window.WRLive.view().zoom - 7.65) < 1e-3", timeout=4000)
        assert "zoom" not in _src_params(pg)[1] and pg.input_value("#zoom") == ""
        assert "zoom=" not in pg.inner_text("#urlOut")

    def test_zooming_with_the_mouse_wheel_inside_the_preview_fills_the_field(self, pv):
        pg, frame, errors, op = self._ready(pv)
        box = pg.evaluate("(() => { const b = document.getElementById('frame').getBoundingClientRect(); return {x: b.left, y: b.top} })()")
        pg.mouse.move(box["x"] + 650, box["y"] + 420)
        pg.mouse.wheel(0, -500)
        pg.wait_for_function("document.getElementById('zoom').value !== ''", timeout=8000)
        pg.wait_for_function("""() => { const v = document.getElementById('frame').contentWindow.WRLive.view()
            return Math.abs(parseFloat(document.getElementById('zoom').value) - v.zoom) < 0.02 }""", timeout=8000)
        assert frame().evaluate("window.__marker") == 99
        assert "zoom=" in pg.inner_text("#urlOut")
        assert errors == []

    def test_reset_layout_restores_zoom_and_center_in_place(self, pv):
        pg, frame, _, op = self._ready(pv)
        pg.fill("#zoom", "10"); pg.fill("#mapLat", "47.9"); pg.fill("#mapLon", "-115.9"); pg.press("#mapLon", "Tab")
        frame().wait_for_function("(() => { const v = window.WRLive.view(); return Math.abs(v.zoom - 10) < 1e-3 && Math.abs(v.lat - 47.9) < 1e-3 })()", timeout=5000)
        n = len(op.seen)
        pg.click("#resetView")
        frame().wait_for_function("(() => { const v = window.WRLive.view(); return Math.abs(v.zoom - 7.65) < 1e-3 && Math.abs(v.lat - 48.145506) < 1e-3 })()", timeout=6000)
        assert frame().evaluate("window.__marker") == 99 and len(op.seen) == n

    def test_the_hint_describes_what_blank_really_does(self, pv):
        open_preview, _ = pv
        pg = open_preview()
        text = pg.inner_text("body")
        assert "default zoom" in text and "fit the area" not in text


class TestWidthPresets:
    def test_a_phone_width_preset_sizes_the_frame_and_the_saved_state(self, pv):
        open_preview, errors = pv
        pg = open_preview()
        pg.select_option("#ar", "100%")
        pg.select_option("#wPreset", "377")
        assert pg.input_value("#fw") == "377" and pg.input_value("#wPreset") == ""     # applied, picker back to its prompt
        size = pg.evaluate("(() => { const f = document.getElementById('frame'); return [f.offsetWidth, f.offsetHeight] })()")
        assert size == [377, 377]
        assert "aspect_ratio: 100%" in pg.inner_text("#yamlOut")
        assert "fw" in pg.evaluate("location.hash") or "377" in pg.evaluate("decodeURIComponent(location.hash)")
        assert errors == []

    def test_the_presets_are_phone_card_widths_not_screen_widths(self, pv):
        open_preview, _ = pv
        pg = open_preview()
        vals = pg.evaluate("[...document.querySelectorAll('#wPreset option')].map(o => o.value).filter(Boolean).map(Number)")
        assert vals and all(v < 430 for v in vals) and 377 in vals


class TestScaleBarSetting:
    def test_it_is_on_by_default_and_the_url_only_says_so_when_it_is_off(self, pv):
        pg, frame, _, _ = _ready_map(pv)
        assert pg.is_checked("#scaleBar") and "scaleBar" not in pg.inner_text("#urlOut")
        assert frame().evaluate("document.querySelectorAll('.maplibregl-ctrl-scale').length") == 1
        pg.uncheck("#scaleBar")
        assert "scaleBar=0" in pg.inner_text("#urlOut")
        pg.check("#scaleBar")
        assert "scaleBar" not in pg.inner_text("#urlOut")

    def test_hiding_and_showing_it_again_is_live_and_never_duplicates_it(self, pv):
        pg, frame, errors, op = _ready_map(pv)
        n = len(op.seen)
        for _ in range(3):
            pg.uncheck("#scaleBar")
            frame().wait_for_function("document.querySelectorAll('.maplibregl-ctrl-scale').length === 0", timeout=3000)
            pg.check("#scaleBar")
            frame().wait_for_function("document.querySelectorAll('.maplibregl-ctrl-scale').length === 1", timeout=3000)
        pg.wait_for_timeout(800)
        assert frame().evaluate("window.__marker") == 99 and len(op.seen) == n
        assert errors == []

    def test_a_pasted_url_with_it_off_unticks_the_box_and_starts_hidden(self, pv):
        open_preview, errors = pv
        pg = open_preview()
        pg.fill("#importUrl", ORIGIN + AQI_URL + "&scaleBar=0"); pg.click("#importBtn")
        assert not pg.is_checked("#scaleBar")
        frame = lambda: [f for f in pg.frames if f != pg.main_frame][-1]
        frame().wait_for_selector("#legend", state="visible", timeout=30000)
        pg.wait_for_timeout(800)
        assert frame().evaluate("document.querySelectorAll('.maplibregl-ctrl-scale').length") == 0
        assert errors == []
