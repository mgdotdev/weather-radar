"""Every id in a page must be unique: getElementById silently returns the FIRST
match, so a duplicate makes some code operate on the wrong element (it once made
the floating-scrub-bar code move things into a checkbox)."""
import collections
import os
import re

import pytest

APP = os.path.join(os.path.dirname(__file__), "..", "..", "server", "src",
                   "weather_radar", "server", "app")


@pytest.mark.parametrize("page", ["map.html", "graph.html", "forecast.html"])
def test_ids_are_unique(page):
    html = open(os.path.join(APP, page), encoding="utf-8").read()
    body = re.sub(r"<script[\s\S]*?</script>", "", html)            # markup only, not JS strings
    ids = re.findall(r'\sid="([^"{}]+)"', body)
    dupes = {i: n for i, n in collections.Counter(ids).items() if n > 1}
    assert not dupes, dupes


def _rendered_ids(html):
    body = re.sub(r"<script[\s\S]*?</script>", "", html)
    return re.findall(r'\sid="([^"{}]+)"', body)


@pytest.mark.parametrize("route", ["ha_preview", "graph_preview"])
def test_rendered_preview_pages_have_unique_ids(route):
    """The preview pages are assembled from a page template + shared markup, so
    check the RENDERED page (a duplicate between the two halves would otherwise
    hide in the separate source files)."""
    pytest.importorskip("fastapi"); pytest.importorskip("jinja2")
    import weather_radar.server.app as app
    ids = _rendered_ids(getattr(app, route)())
    dupes = {i: n for i, n in collections.Counter(ids).items() if n > 1}
    assert not dupes, dupes
    for needed in ("frame", "urlOut", "yamlOut", "follow", "importUrl", "importBtn"):
        assert needed in ids, needed                 # the shared engine's required elements
