"""Scrubber dragging must track the cursor at every legend size.

The legend is scaled with CSS transform:scale(legendSize). Handles are DRAWN in
the SVG's unscaled units (so land at x*scale on screen), while pointer events
arrive in screen pixels — pointerToEpoch has to undo the scale. These tests run
the REAL functions from map.html under Node against a fake scaled SVG.
"""
import json
import os
import re
import shutil
import subprocess

import pytest

MAP_HTML = os.path.join(os.path.dirname(__file__), "..", "..", "server", "src",
                        "weather_radar", "server", "app", "map.html")

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def _function_source(html, name):
    start = html.index(f"function {name}(")
    i = html.index("{", start)
    depth = 0
    for j in range(i, len(html)):
        depth += {"{": 1, "}": -1}.get(html[j], 0)
        if depth == 0:
            return html[start:j + 1]
    raise AssertionError(name)


def _run(scenarios):
    html = open(MAP_HTML, encoding="utf-8").read()
    pad = re.search(r"const SC_PAD\s*=\s*(\d+)", html).group(1)
    funcs = "\n".join(_function_source(html, n) for n in
                      ("scrubberScale", "scWidth", "epochToX", "xToEpoch", "pointerToEpoch"))
    js = f"""
const SC_PAD = {pad};
let legendSize, floatScrubber = false, forecastStart, forecastEnd, scrubberEl;
{funcs}
const out = [];
for (const s of JSON.parse(require('fs').readFileSync(0, 'utf8'))) {{
  legendSize = s.legend; floatScrubber = !!s.floating; forecastStart = s.start; forecastEnd = s.end;
  // an SVG s.width unscaled px wide, drawn under transform:scale(s.scale) with
  // its top-left at screen x = s.left. (Docked, s.scale is the legend size;
  // floating, the bar is outside the legend and nothing scales it.)
  scrubberEl = {{ getBoundingClientRect: () => ({{ left: s.left, width: s.width * s.scale }}) }};
  const res = [];
  for (const e of s.epochs) {{
    const handleScreenX = s.left + epochToX(e) * s.scale;      // where the handle appears
    res.push(pointerToEpoch({{ clientX: handleScreenX }}));     // cursor on that handle
  }}
  out.push(res);
}}
console.log(JSON.stringify(out));
"""
    r = subprocess.run(["node", "-e", js], input=json.dumps(scenarios),
                       capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


START, END = 1_800_000_000, 1_800_000_000 + 11 * 24 * 3600


@pytest.mark.parametrize("scale", [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.75, 3.0])
def test_cursor_on_a_handle_maps_back_to_that_handles_epoch(scale):
    epochs = [START + int(f * (END - START)) for f in (0, 0.1, 0.25, 0.5, 0.77, 0.9, 1)]
    (got,) = _run([{"scale": scale, "legend": scale, "left": 137, "width": 260, "start": START,
                    "end": END, "epochs": epochs}])
    # a pixel is ~ (END-START)/(260-2*pad) seconds; the round trip is exact up to rounding
    assert all(abs(g - e) <= 1 for g, e in zip(got, epochs)), (scale, got, epochs)


@pytest.mark.parametrize("legend", [0.5, 1.0, 2.0, 3.0])
def test_a_floating_bar_ignores_the_legend_size(legend):
    """Floating, the scrubber is drawn at scale 1 whatever the legend size is."""
    epochs = [START + int(f * (END - START)) for f in (0, 0.2, 0.5, 0.8, 1)]
    (got,) = _run([{"scale": 1, "legend": legend, "floating": True, "left": 90, "width": 700,
                    "start": START, "end": END, "epochs": epochs}])
    assert all(abs(g - e) <= 1 for g, e in zip(got, epochs)), (legend, got, epochs)


def test_the_bug_is_visible_without_the_correction():
    """Guard the test itself: undoing the fix must make scale != 1 fail."""
    html = open(MAP_HTML, encoding="utf-8").read()
    assert "/ scrubberScale()" in _function_source(html, "pointerToEpoch")
