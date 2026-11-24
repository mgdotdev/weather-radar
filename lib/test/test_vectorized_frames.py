"""The live series stream and the bundle builder compute a frame for ALL points in one numpy pass.
Both must emit exactly the bytes the old per-point loop did — and must not fall back to that loop
(it cost ~300 ms a frame at 44,000 points, which made a full-window bundle take many minutes)."""
import asyncio
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))        # the synthetic-model helpers live in a sibling test module

pytest.importorskip("fastapi")

from weather_radar.lib import framing as F                                      # noqa: E402
from weather_radar.lib.models.ndfd_model import _NDFDPointModel                 # noqa: E402
from weather_radar.lib.models.utils import BoundsError                          # noqa: E402
from test_ndfd_frame_evaluator import KNOTS, T0, loop, point                    # noqa: E402

LAT, LON, W, H = 48.1, -116.2, 449.63, 397.11
HALF = 1800


def _models():
    return ([point(i, "cubic", nonneg=(i % 4 == 0)) for i in range(40)]
            + [point(40 + i, "pchip", nonneg=True) for i in range(10)]
            + [point(50, spline=None), point(51, end=T0 + 30 * 3600), point(52, start=T0 + 10 * 3600)])


@pytest.fixture
def env(tmp_path, monkeypatch):
    import weather_radar.server.bundle as b
    import weather_radar.server.api.models as m
    monkeypatch.setattr(b, "BUNDLE_DIR", str(tmp_path))
    monkeypatch.setattr(b, "_source_token", lambda model, kind: 700.0)
    monkeypatch.setattr(m, "_gzip_stream", lambda gen: gen)
    monkeypatch.setattr(b, "save_request_layer", lambda *a, **k: False)

    models = _models()

    class Ens:
        def __init__(self, area, model): pass
        def iter_points(self): return iter(models)
        def value_range(self):
            from datetime import datetime, timezone
            iso = lambda e: datetime.fromtimestamp(e, timezone.utc).isoformat()
            # the window runs PAST the points' horizon, so the last frames are wholly out of range
            return {"start": iso(T0), "end": iso(float(KNOTS[-1]) + 4 * 3600), "min": 0, "max": 100, "uom": "degF"}
    monkeypatch.setattr(m, "ENSEMBLE_KINDS", lambda: {"observation": Ens})
    monkeypatch.setattr(m, "_is_ndfd", lambda: True)

    # the slow per-point path must never run: make it fail loudly instead of quietly producing Nones
    def boom(self, e): raise AssertionError("per-point predict_epoch was called")
    return b, m, models, boom


def _expected(models, epoch, scale, offset):
    return F.encode_frame(epoch, loop(models, epoch), scale, offset)


def test_the_live_stream_emits_the_same_bytes_without_the_per_point_loop(env, monkeypatch):
    b, m, models, boom = env
    start, end = int(T0), int(KNOTS[-1]) + 3600                      # the sweep runs one hour past the horizon
    epochs = list(range(start, end + 1, HALF))
    ref = {e: _expected(models, e, 10.0, 0.0) for e in epochs}        # computed with the real per-point loop first
    present = {e for e in epochs if any(v is not None for v in loop(models, e))}   # frames with any value at all
    monkeypatch.setattr(_NDFDPointModel, "predict_epoch", boom)
    resp = m.ensemble_series_stream("temperature", LAT, LON, start, end, HALF, None, W, H, "observation")

    async def drain():
        got, first = {}, True
        async for chunk in resp.body_iterator:
            for rec in F.iter_records(chunk):
                if first:
                    first = False
                    continue
                got[F.decode_frame_epoch(rec)] = rec
        return got
    got = asyncio.run(drain())
    want = {e: p for e, p in ref.items() if e in present}             # all-missing frames are not sent
    assert len(want) > 200 and set(got) == set(want) and got == want
    assert max(got) <= KNOTS[-1] and len(got) < len(epochs)           # the out-of-horizon frames were skipped


def test_the_bundle_builder_writes_the_same_frames_without_the_per_point_loop(env, monkeypatch):
    b, m, models, boom = env
    captured = {}

    def fake_write(dirpath, header, frame_iter):
        captured["frames"] = list(frame_iter)
        captured["header"] = header
        return len(captured["frames"])
    monkeypatch.setattr(b, "_write_bundle", fake_write)
    want = {}
    for k in range(0, 3000):
        e = T0 + k * 300
        if e > KNOTS[-1] + 4 * 3600 + 1:
            break
        if any(v is not None for v in loop(models, e)):
            want[e] = _expected(models, e, 10.0, 0.0)
    monkeypatch.setattr(_NDFDPointModel, "predict_epoch", boom)

    out = b.build_bundle("temperature", LAT, LON, W, H, "observation", 300)
    assert out is not None
    frames = captured["frames"]
    assert frames and len(frames) == len(want)
    got = {rec["time"]: F.encode_frame(rec["time"], rec["vals"], 10.0, 0.0) for rec in frames}
    assert got == want                                                             # every frame, every byte
    assert len(captured["header"]["coords"]) == len(models)           # the lattice is fixed: a point with no spline keeps its slot
    assert max(got) <= KNOTS[-1]                                                   # frames past the horizon were not written


def test_the_json_source_still_uses_the_loop(env, monkeypatch):
    """Models that aren't NBM/NDFD point models (the JSON gridpoints source) keep the per-point path."""
    b, m, models, boom = env
    calls = []

    class P:
        def __init__(self, i): self.i = i; self.model = (0, 0, type("C", (), {"lon": -116 + i * .1, "lat": 48.0})(), None, "degF", 0)
        def predict_epoch(self, e): calls.append(e); return 40.0 + self.i

    class Ens:
        def __init__(self, area, model): pass
        def iter_points(self): return iter([P(0), P(1), P(2)])
    monkeypatch.setattr(m, "ENSEMBLE_KINDS", lambda: {"observation": Ens})
    epochs = [T0 + 3600, T0 + 7200]
    resp = m.ensemble_series_stream("temperature", LAT, LON, epochs[0], epochs[-1], 3600, epochs, W, H, "observation")

    async def drain():
        return [r async for chunk in resp.body_iterator for r in F.iter_records(chunk)]
    recs = asyncio.run(drain())
    assert len(recs) == 3 and len(calls) == 6                 # header + 2 frames; 3 points x 2 frames via the loop
