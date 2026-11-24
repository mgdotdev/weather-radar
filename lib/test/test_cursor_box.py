"""The cursor box: same look/size as a pin's info box, scaled by the pin-size
setting, placed so it never runs off-screen, and switchable off in Settings."""
import json
import os
import re
import shutil
import subprocess

import pytest

MAP_HTML = os.path.join(os.path.dirname(__file__), "..", "..", "server", "src",
                        "weather_radar", "server", "app", "map.html")
HTML = open(MAP_HTML, encoding="utf-8").read()


def _function_source(name):
    start = HTML.index(f"function {name}(")
    i = HTML.index("{", start)
    depth = 0
    for j in range(i, len(HTML)):
        depth += {"{": 1, "}": -1}.get(HTML[j], 0)
        if depth == 0:
            return HTML[start:j + 1]
    raise AssertionError(name)


class TestSharedLook:
    def test_pin_box_and_cursor_box_use_the_same_style_constant(self):
        assert "const PIN_INFO_CSS" in HTML
        info = HTML[HTML.index("info.style.cssText"):][:400]
        readout = HTML[HTML.index('readoutEl.style.cssText'):][:300]
        assert "PIN_INFO_CSS" in info and "PIN_INFO_CSS" in readout

    def test_the_old_independent_readout_styling_is_gone(self):
        tag = HTML[HTML.index('<div id="cursorReadout"'):][:260]
        assert "rgba(0,0,0,0.75)" not in tag and "padding: 4px 7px" not in tag

    def test_pin_box_dimensions_live_in_the_shared_constant(self):
        const = HTML[HTML.index("const PIN_INFO_CSS"):][:900]
        for piece in ("width:190px", "font-size:12px", "padding:3px 22px 5px 7px",
                      "rgba(30,30,30,0.85)", "border-radius:4px"):
            assert piece in const

    def test_cursor_box_scales_with_pin_size_from_its_corner(self):
        assert "transform-origin:0 0" in HTML[HTML.index("readoutEl.style.cssText"):][:200]
        assert "readoutEl.style.transform = `scale(${pinSize})`" in HTML

    def test_changing_pin_size_refreshes_the_cursor_box(self):
        assert "refreshReadout()   // the cursor box follows the pin size too" in HTML


class TestSetting:
    def test_checkbox_exists_and_defaults_on(self):
        assert re.search(r'<input type="checkbox" id="showCursorBox" checked', HTML)

    def test_hidden_only_when_param_is_zero(self):
        assert 'searchParams.get("cursorBox") !== "0"' in HTML

    def test_persists_and_survives_a_model_switch(self):
        assert 'u.searchParams.set("cursorBox", "0")' in HTML
        # both model-switch carry-over lists keep the cursor-box and floating-bar settings
        lists = re.findall(r'for \(const k of \[("lat", "lon".*?)\]\) \{', HTML, re.S)
        assert len(lists) == 2
        for lst in lists:
            assert all(f'"{k}"' in lst for k in ("cursorBox", "floatScrubber", "attribution"))

    def test_readout_is_hidden_when_off(self):
        assert 'if (!showCursorBox || !cursorPx)' in HTML


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
class TestPlacement:
    def _place(self, cases):
        js = _function_source("readoutPosition") + (
            "\nconst cs = JSON.parse(require('fs').readFileSync(0,'utf8'));"
            "console.log(JSON.stringify(cs.map(a => readoutPosition(...a))));")
        out = subprocess.run(["node", "-e", js], input=json.dumps(cases),
                             capture_output=True, text=True, check=True).stdout
        return json.loads(out)

    def test_normal_case_is_below_right_of_the_pointer(self):
        (p,) = self._place([[100, 200, 190, 40, 1000, 800, 12]])
        assert p == {"x": 112, "y": 212}

    def test_flips_left_near_the_right_edge(self):
        (p,) = self._place([[900, 200, 190, 40, 1000, 800, 12]])
        assert p["x"] == 900 - 12 - 190 and p["y"] == 212

    def test_flips_up_near_the_bottom_edge(self):
        (p,) = self._place([[100, 780, 190, 40, 1000, 800, 12]])
        assert p["y"] == 780 - 12 - 40 and p["x"] == 112

    def test_scaled_box_uses_its_on_screen_size(self):
        # at 3x the box is 570 wide: it fits at x=112 only if there's room
        fits, flips = self._place([[100, 100, 570, 120, 1000, 800, 12],
                                   [600, 100, 570, 120, 1000, 800, 12]])
        assert fits["x"] == 112
        assert flips["x"] == 600 - 12 - 570

    def test_never_leaves_the_viewport_origin(self):
        (p,) = self._place([[5, 5, 900, 900, 400, 300, 12]])      # box bigger than the viewport
        assert p["x"] >= 0 and p["y"] >= 0


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
class TestStartupOrder:
    """refreshReadout() is called while pins are restored from the URL
    (addPin -> applyPinSize), BEFORE later `let`s in the load handler (e.g.
    `drawing`) are initialized — and by pointer events that can arrive before
    setup finishes. Touching an uninitialized `let` throws (even via typeof),
    which aborted the whole handler and broke the page. Run the REAL function
    in exactly that situation."""

    def _run(self, body):
        js = ("const showCursorBox = true; const readoutEl = { style: {} };\n"
              "let cursorPx = %s;\n" % body["cursor"]
              + _function_source("refreshReadout") + "\n"
              "let threw = null;\n"
              "try { refreshReadout(); } catch (e) { threw = String(e); }\n"
              "let drawing = false;   // declared AFTER the call: TDZ at call time\n"
              "console.log(JSON.stringify({ threw, display: readoutEl.style.display }));")
        out = subprocess.run(["node", "-e", js], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        return json.loads(out.stdout)

    def test_startup_call_with_no_pointer_does_not_throw_and_hides_the_box(self):
        r = self._run({"cursor": "null"})
        assert r["threw"] is None and r["display"] == "none"

    def test_pointer_moving_before_setup_finishes_does_not_throw(self):
        # a pointer is over the map but `drawing` isn't initialized yet; it must
        # not throw on `drawing` (it may fail later on other not-yet-built
        # state, which is not what this guards — so only check the TDZ error)
        r = self._run({"cursor": "{ x: 5, y: 5 }"})
        assert r["threw"] is None or "before initialization" not in r["threw"]

    def test_the_cheap_early_exit_comes_before_anything_uninitialized(self):
        src = _function_source("refreshReadout")
        assert src.index("!showCursorBox || !cursorPx") < src.index("drawing")
