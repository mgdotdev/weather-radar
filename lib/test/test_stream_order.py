"""Stream ordering when the bundle holds only SOME of the requested frames:
stored frames go out first (instant disk reads, before any slow compute), then
the missing ones are computed in the order the client asked — nearest the
playhead first — and `times` may use compact `from:to:step` runs."""
import asyncio

import pytest

pytest.importorskip("fastapi")

from weather_radar.lib import aqi, framing as F          # noqa: E402

T0 = 1_800_000_000 - 1_800_000_000 % 3600
HOUR = 3600
HALF = 1800
PIVOT = T0 + 6 * HOUR
LAT, LON, W, H = 48.145506, -116.391332, 201.9, 183.7


@pytest.fixture
def env(tmp_path, monkeypatch):
    import weather_radar.server.bundle as b
    import weather_radar.server.api.models as m
    monkeypatch.setattr(b, "BUNDLE_DIR", str(tmp_path))
    monkeypatch.setattr(b, "_source_token", lambda model, kind: 700.0)
    monkeypatch.setattr(b, "_aqi_token_in_use", lambda p: 700.0)
    monkeypatch.setattr(m, "_gzip_stream", lambda gen: gen)       # observe raw records
    # keep request-created layers out of these ordering tests
    monkeypatch.setattr(b, "save_request_layer", lambda *a, **k: False)
    events = []
    return b, m, events


def _store_hourly(b, model, kind, uom, n=2):
    header = {"type": "range", "uom": uom, "coords": [[LON + i * .1, LAT] for i in range(n)],
              "range": {"min": 0, "max": 300, "uom": uom},
              "meta": {"source_mtime": 700.0}}
    frames = [{"time": T0 + k * HOUR, "vals": [1000 + k] * n} for k in range(13)]   # T0..T0+12h
    assert b._write_bundle(b._bundle_dir(model, LAT, LON, W, H, kind), header, iter(frames))


def _drain(resp, events):
    async def go():
        sent = []
        first = True
        async for chunk in resp.body_iterator:
            rec = next(F.iter_records(chunk))
            if first:                       # grid header
                first = False
                continue
            e = F.decode_frame_epoch(rec)
            sent.append(e)
            events.append(("send", e))
        return sent
    return asyncio.run(go())


def _request_order():
    """What the client sends at dt=30 min with the playhead at PIVOT: the frames
    behind it nearest-first as ONE compact run, then the sequence forward."""
    back = f"{PIVOT - HALF}:{T0}:{HALF}"                      # descending run
    return back, PIVOT, T0 + 12 * HOUR


def _expected(stored_hourly=True):
    back = list(range(PIVOT - HALF, T0 - 1, -HALF))
    fwd = list(range(PIVOT, T0 + 12 * HOUR + 1, HALF))
    wanted = back + fwd
    stored = sorted(e for e in wanted if (e - T0) % HOUR == 0)
    live = [e for e in wanted if (e - T0) % HOUR != 0]
    return wanted, stored, live


def _check(events, sent):
    wanted, stored, live = _expected()
    assert sorted(sent) == sorted(wanted) and len(sent) == len(wanted)
    # 1. every stored frame is sent before ANY computing starts
    first_compute = next(i for i, ev in enumerate(events) if ev[0] == "compute")
    stored_sends = [ev[1] for ev in events[:first_compute] if ev[0] == "send"]
    assert stored_sends == stored
    # 2. stored frames are in time order; the computed ones follow in REQUEST order
    assert sent[:len(stored)] == stored
    assert sent[len(stored):] == live
    # 3. so the frame nearest the playhead (just behind it) is computed first
    assert live[0] == PIVOT - HALF


class TestAqiStream:
    def test_stored_first_then_requested_order(self, env, monkeypatch):
        b, m, events = env
        _store_hourly(b, "pm25", "aqi", "AQI")
        monkeypatch.setattr(aqi, "fit_area_dense",
                            lambda area, p: {"start": T0, "end": T0 + 12 * HOUR})
        monkeypatch.setattr(aqi, "grid_points", lambda fit: [[LON, LAT], [LON + .1, LAT]])

        def predict(fit, epochs, fixed_grid=False):
            for e in epochs:
                events.append(("compute", e))
                yield {"epoch": e, "vals": [5, 5]}
        monkeypatch.setattr(aqi, "predict_area", predict)

        back, start, end = _request_order()
        resp = m.aqi_stream(LAT, LON, W, H, "pm25", start, end, HALF, back)
        _check(events, _drain(resp, events))

    def test_fully_stored_request_never_computes(self, env, monkeypatch):
        b, m, events = env
        _store_hourly(b, "pm25", "aqi", "AQI")
        monkeypatch.setattr(aqi, "fit_area_dense",
                            lambda *a: (_ for _ in ()).throw(AssertionError("fit ran")))
        resp = m.aqi_stream(LAT, LON, W, H, "pm25", T0, T0 + 12 * HOUR, HOUR, None)
        sent = _drain(resp, events)
        assert sent == list(range(T0, T0 + 12 * HOUR + 1, HOUR))


class _Center:
    def __init__(self, lon, lat):
        self.lon, self.lat = lon, lat


class _Point:
    def __init__(self, i, events):
        self.i, self.events = i, events
        self.model = (0, 0, _Center(LON + i * .1, LAT), None, "degF", 0)

    def predict_epoch(self, e):
        if self.i == 0:
            self.events.append(("compute", e))
        return 50.0


class TestSeriesStream:
    def test_stored_first_then_requested_order(self, env, monkeypatch):
        b, m, events = env
        _store_hourly(b, "temperature", "observation", "degF")

        class Ens:
            def __init__(self, area, model): pass
            def iter_points(self): return [_Point(0, events), _Point(1, events)]
        monkeypatch.setattr(m, "ENSEMBLE_KINDS", lambda: {"observation": Ens})
        monkeypatch.setattr(m, "_is_ndfd", lambda: True)

        back, start, end = _request_order()
        resp = m.ensemble_series_stream("temperature", LAT, LON, start, end, HALF,
                                        back, W, H, "observation")
        _check(events, _drain(resp, events))
