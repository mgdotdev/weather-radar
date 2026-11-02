"""Offline tests for the per-frame ensemble cache (framecache.py).

No network and no scipy: we exercise the cache primitives directly — area-key
determinism/rounding, write/read round-trip, per-frame independence (one frame
doesn't satisfy another), area-level freshness gating, and atomic writes with
no leftover temp files.

A fake `requests` module is installed before import so `cachewriter` (imported
transitively for CACHE_ROOT/is_fresh) never touches the network.
"""

import os
import sys
import json
import types
import tempfile
import importlib.util

import pytest


FUTURE = "Wed, 01 Jan 2099 00:00:00 GMT"
PAST = "Thu, 01 Jan 1970 00:00:00 GMT"


def _load(cache_dir, monkeypatch):
    monkeypatch.setenv("CACHE_DIR", cache_dir)
    fake = types.ModuleType("requests")

    class _FakeSession:
        def __init__(self):
            self.headers = {}

    fake.session = lambda: _FakeSession()
    monkeypatch.setitem(sys.modules, "requests", fake)

    here = os.path.dirname(__file__)
    base = os.path.abspath(os.path.join(here, "..", "src", "weather_radar", "lib"))

    # Load cachewriter under the package path framecache's relative import needs.
    cw_src = os.path.join(base, "cachewriter.py")
    spec = importlib.util.spec_from_file_location("weather_radar.lib.cachewriter", cw_src)
    cw = importlib.util.module_from_spec(spec)
    pkg = types.ModuleType("weather_radar.lib"); pkg.__path__ = [base]
    sys.modules.setdefault("weather_radar", types.ModuleType("weather_radar"))
    sys.modules["weather_radar.lib"] = pkg
    sys.modules["weather_radar.lib.cachewriter"] = cw
    spec.loader.exec_module(cw)

    fc_src = os.path.join(base, "framecache.py")
    spec = importlib.util.spec_from_file_location("weather_radar.lib.framecache", fc_src)
    fc = importlib.util.module_from_spec(spec)
    sys.modules["weather_radar.lib.framecache"] = fc
    spec.loader.exec_module(fc)
    return fc, cw


def _write_gridpoint(cw, path, expires):
    _url, d, cache_file = cw._tempfile_for(path)
    os.makedirs(d, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump({"headers": {"Expires": expires}, "body": {}}, f)


@pytest.fixture
def fc(monkeypatch):
    cache_dir = tempfile.mkdtemp()
    mod, cw = _load(cache_dir, monkeypatch)
    return types.SimpleNamespace(mod=mod, cw=cw, dir=cache_dir)


def test_area_key_deterministic_and_rounds(fc):
    k1 = fc.mod.area_key("temperature", "observation", 47.68, -116.78, 40, 10)
    k2 = fc.mod.area_key("temperature", "observation", 47.680001, -116.780004, 40, 10)
    k3 = fc.mod.area_key("temperature", "observation", 47.68, -116.78, 40, 20)
    assert k1 == k2          # trivial float diffs round to the same area
    assert k1 != k3          # different size => different area


def test_write_then_read_roundtrip(fc):
    akey = fc.mod.area_key("m", "k", 1, 2, 3, 4)
    frame = {"type": "FeatureCollection", "features": [{"v": 1}]}
    fc.mod.write_frame(akey, 1000, frame)
    assert fc.mod.read_frame(akey, 1000) == frame
    assert fc.mod.read_frame(akey, 2000) is None   # other timestamp = miss


def test_frames_independent_across_time(fc):
    akey = fc.mod.area_key("m", "k", 1, 2, 3, 4)
    fc.mod.write_frame(akey, 100, {"features": ["a"]})
    fc.mod.write_frame(akey, 200, {"features": ["b"]})
    assert fc.mod.read_frame(akey, 100)["features"] == ["a"]
    assert fc.mod.read_frame(akey, 200)["features"] == ["b"]


def test_area_freshness_gates(fc):
    fresh = "/gridpoints/ABC/1,2"
    stale = "/gridpoints/ABC/9,9"
    _write_gridpoint(fc.cw, fresh, FUTURE)
    _write_gridpoint(fc.cw, stale, PAST)
    assert fc.mod.area_is_fresh([fresh]) is True
    assert fc.mod.area_is_fresh([fresh, stale]) is False
    assert fc.mod.area_is_fresh([]) is True   # no inputs => trivially fresh


def test_write_leaves_no_tmp(fc):
    akey = fc.mod.area_key("m", "k", 5, 6, 7, 8)
    fc.mod.write_frame(akey, 42, {"features": []})
    area_dir = fc.mod._area_dir(akey)
    entries = os.listdir(area_dir)
    assert "42.json" in entries
    assert not any(e.endswith(".tmp") for e in entries)


def test_version_mismatch_invalidates(fc):
    """A frame stamped with one data version must not be served once the
    underlying NOAA data (and thus the version) changes."""
    gp = "/gridpoints/ABC/1,2"
    _write_gridpoint(fc.cw, gp, FUTURE)
    v1 = fc.mod.data_version([gp])

    akey = fc.mod.area_key("m", "k", 1, 2, 3, 4)
    fc.mod.write_frame(akey, 100, {"features": ["v1"]}, version=v1)

    # Same version -> hit.
    assert fc.mod.read_frame(akey, 100, version=v1) == {"features": ["v1"]}

    # NOAA data refreshed: different Expires -> different version -> miss.
    _write_gridpoint(fc.cw, gp, "Thu, 01 Jan 2099 00:00:00 GMT")
    v2 = fc.mod.data_version([gp])
    assert v2 != v1
    assert fc.mod.read_frame(akey, 100, version=v2) is None
    # Without a version arg, the raw frame is still returned (back-compat).
    assert fc.mod.read_frame(akey, 100) == {"features": ["v1"]}


def test_clear_area_removes_frames(fc):
    akey = fc.mod.area_key("m", "k", 9, 9, 1, 1)
    fc.mod.write_frame(akey, 1, {"features": []}, version="x")
    fc.mod.write_frame(akey, 2, {"features": []}, version="x")
    assert fc.mod.read_frame(akey, 1) is not None
    removed = fc.mod.clear_area(akey)
    assert removed == 2
    assert fc.mod.read_frame(akey, 1) is None
    assert fc.mod.read_frame(akey, 2) is None


# --- Writer compute_frame integration (hit path, no scipy) ------------------
# compute_frame lazily imports `area` and the model ensembles. We stub those
# in sys.modules so we can exercise the writer's cache-hit + dedupe logic
# in-process without scipy or a real area.

def _install_stubs(monkeypatch, base, gridpoints, predict_fc):
    """Install fake weather_radar.lib.area and models.* so compute_frame can
    import them. `predict_fc` is returned by the stub ensemble's predict()."""
    # area stub
    area_mod = types.ModuleType("weather_radar.lib.area")
    class MapCoordinate:
        def __init__(self, lat, lon): self.lat = lat; self.lon = lon
    class CoordinateArea:
        def __init__(self, coord, w, h): self.coord = coord
        @property
        def gridpoints(self): return gridpoints
    area_mod.MapCoordinate = MapCoordinate
    area_mod.CoordinateArea = CoordinateArea
    monkeypatch.setitem(sys.modules, "weather_radar.lib.area", area_mod)

    # models package + ensembles
    models_pkg = types.ModuleType("weather_radar.lib.models")
    models_pkg.__path__ = [os.path.join(base, "models")]
    monkeypatch.setitem(sys.modules, "weather_radar.lib.models", models_pkg)

    class _Ens:
        def __init__(self, area, model): pass
        def predict(self, dtime, verbose=False): return predict_fc
    for name in ("accumulation", "observation"):
        m = types.ModuleType(f"weather_radar.lib.models.{name}")
        m.AccumulationEnsemble = _Ens
        m.ObservationEnsemble = _Ens
        monkeypatch.setitem(sys.modules, f"weather_radar.lib.models.{name}", m)


def test_compute_frame_hit_skips_compute(fc, monkeypatch):
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                        "src", "weather_radar", "lib"))
    # Mark the area's gridpoint fresh and pre-write the frame WITH the matching
    # data version (so compute_frame trusts it as current).
    gp = "/gridpoints/ABC/1,2"
    _write_gridpoint(fc.cw, gp, FUTURE)
    version = fc.mod.data_version([gp])
    params = dict(model="m", kind="observation", lat=1, lon=2, width=3, height=4, time=1000)
    akey = fc.mod.area_key("m", "observation", 1, 2, 3, 4)
    fc.mod.write_frame(akey, 1000, {"type": "FeatureCollection", "features": ["cached"]}, version=version)

    # Stub area (returns our fresh gp) and an ensemble that would raise if used.
    _install_stubs(monkeypatch, base, [gp.split("/gridpoints/")[1]],
                   predict_fc={"features": ["SHOULD_NOT_BE_USED"]})

    w = fc.cw._Writer()
    out = json.loads(w.compute_frame(json.dumps(params)))
    assert out["features"] == ["cached"]   # served from cache, not computed


def test_compute_frame_miss_computes_and_caches(fc, monkeypatch):
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                        "src", "weather_radar", "lib"))
    gp = "/gridpoints/ABC/3,4"
    _write_gridpoint(fc.cw, gp, FUTURE)
    params = dict(model="m", kind="observation", lat=5, lon=6, width=7, height=8, time=2000)
    _install_stubs(monkeypatch, base, [gp.split("/gridpoints/")[1]],
                   predict_fc={"type": "FeatureCollection", "features": ["computed"]})

    w = fc.cw._Writer()
    out = json.loads(w.compute_frame(json.dumps(params)))
    assert out["features"] == ["computed"]
    # And it should now be cached on disk.
    akey = fc.mod.area_key("m", "observation", 5, 6, 7, 8)
    assert fc.mod.read_frame(akey, 2000)["features"] == ["computed"]
