"""The map's "Add to Hot Cache" action must keep `pins` in the watched URL:
the builder pre-warms each pin's graph from it (server/pointseries.py)."""
import os
import re

MAP_HTML = os.path.join(os.path.dirname(__file__), "..", "..", "server", "src",
                        "weather_radar", "server", "app", "map.html")


def _stripped_params():
    html = open(MAP_HTML, encoding="utf-8").read()
    body = html[html.index('getElementById("ctxHotCache").addEventListener'):]
    m = re.search(r"for \(const k of \[([^\]]*)\]\) \{\s*u\.searchParams\.delete\(k\)", body)
    assert m, "hot-cache param-stripping loop not found"
    return re.findall(r'"([^"]+)"', m.group(1))


class TestHotCacheWatchUrl:
    def test_pins_are_kept(self):
        assert "pins" not in _stripped_params()

    def test_secrets_and_view_only_params_are_still_dropped(self):
        stripped = _stripped_params()
        for k in ("zoom", "speed", "pinSize"):
            assert k in stripped
        html = open(MAP_HTML, encoding="utf-8").read()
        body = html[html.index('getElementById("ctxHotCache").addEventListener'):]
        assert 'u.searchParams.delete("token")' in body.split("watchUrl")[0]

    def test_a_watched_url_with_pins_yields_graph_targets(self):
        import pytest
        pytest.importorskip("fastapi"); pytest.importorskip("jinja2")
        from weather_radar.server.pointseries import targets_for_url
        url = "/map/models/temperature/?lat=48.1&lon=-116.4&width=200&height=180&dt=900&pins=47.7,-116.7,Home"
        assert [t[2:4] for t in targets_for_url(url)] == [(47.7, -116.7)] * 2
