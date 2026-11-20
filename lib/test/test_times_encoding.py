"""The REAL client encoder (extracted from map.html and run under Node) must
produce compact `times` that the server's parser expands back to exactly the
same epochs in the same order."""
import json
import os
import re
import shutil
import subprocess

import pytest

from weather_radar.lib.frametimes import frame_times

MAP_HTML = os.path.join(os.path.dirname(__file__), "..", "..", "server", "src",
                        "weather_radar", "server", "app", "map.html")

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def _encoder_source():
    html = open(MAP_HTML, encoding="utf-8").read()
    start = html.index("function encodeTimes(list) {")
    depth, i = 0, html.index("{", start)
    for j in range(i, len(html)):
        depth += {"{": 1, "}": -1}.get(html[j], 0)
        if depth == 0:
            return html[start:j + 1]
    raise AssertionError("unbalanced encodeTimes")


def _encode_all(cases):
    js = _encoder_source() + (
        "\nconst cases = JSON.parse(require('fs').readFileSync(0, 'utf8'));\n"
        "console.log(JSON.stringify(cases.map(encodeTimes)));")
    out = subprocess.run(["node", "-e", js], input=json.dumps(cases),
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


STEP = 900
T0 = 1_795_330_800


def _first_occurrences(xs):
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _roundtrip(cases):
    enc = _encode_all(cases)
    for case, e in zip(cases, enc):
        # the server drops repeats (keeping the first), so compare against that
        assert frame_times(None, None, 0, e) == _first_occurrences(case), (case[:6], e[:80])
    return enc


class TestEncodeTimes:
    def test_the_request_that_motivated_this(self):
        batch = list(range(1795547700, 1795330800 - 1, -900))      # 242 descending
        (enc,) = _roundtrip([batch])
        assert enc == "1795547700:1795330800:900"

    def test_roundtrip_many_shapes(self):
        cases = [
            [], [5], [5, 9], [5, 6, 7], [7, 6, 5], [1, 3, 5, 7, 9],
            [1, 2, 4, 6, 8], [10, 20, 30, 15, 25, 35, 45],
            [100, 100], [3, 3, 3, 3], [1, 2, 3, 1, 2, 3],
            list(range(0, 90, 10)) + [5, 15, 25] + list(range(500, 400, -25)),
        ]
        _roundtrip(cases)

    def test_roundtrip_random(self):
        import random
        rng = random.Random(11)
        cases = []
        for _ in range(300):
            c = []
            for _ in range(rng.randint(0, 8)):
                kind = rng.choice("rrs")
                if kind == "r":
                    a, d, n = T0 + rng.randint(0, 400) * STEP, rng.choice([-1, 1]) * rng.choice([300, 900, 3600]), rng.randint(1, 60)
                    c += [a + d * k for k in range(n)]
                else:
                    c.append(T0 + rng.randint(0, 5000) * 300)
            # the server dedupes, so compare against the first-occurrence order
            seen, c2 = set(), []
            for x in c:
                if x not in seen:
                    seen.add(x); c2.append(x)
            cases.append(c2)
        _roundtrip(cases)

    def test_url_length_stays_small_at_every_dt(self):
        window = 11 * 24 * 3600
        for dt in (300, 900, 3600):
            back = list(range(T0 + window, T0, -dt))
            (enc,) = _roundtrip([back])
            assert len(enc) < 40, (dt, enc)
            assert len(",".join(map(str, back))) > 30 * len(enc) or dt == 3600

    def test_short_two_element_runs_are_left_plain(self):
        assert _encode_all([[10, 20]]) == ["10,20"]
