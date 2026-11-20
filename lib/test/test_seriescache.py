"""Disk cache of finished point series (lib/seriescache.py)."""
import os
import threading
import time

import pytest

from weather_radar.lib import seriescache as sc

DATA = {"type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {"temperature": 41.0, "time": 1}}]}


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "SERIES_ROOT", str(tmp_path))
    sc._READ_CACHE.clear()
    return tmp_path


def _put(root, token, managed=False, data=DATA, built=None):
    path = sc.path_for("observation", "temperature", 47.7, -116.7, 450)
    blob = sc.pack(data, token, managed)
    if built is not None:                      # backdate for staleness tests
        import gzip, json
        e = json.loads(gzip.decompress(blob)); e["built_ts"] = built
        blob = gzip.compress(json.dumps(e).encode())
    assert sc.save(path, blob, token, managed) == "ok"
    return path


class TestKeys:
    def test_dt_zero_and_hourly_are_one_entry(self, root):
        a = sc.path_for("observation", "temperature", 1.0, 2.0, 0)
        assert a == sc.path_for("observation", "temperature", 1.0, 2.0, 3600)
        assert a == sc.path_for("observation", "temperature", 1.0, 2.0, -5)

    def test_different_step_kind_attribute_or_point_differ(self, root):
        base = sc.path_for("observation", "temperature", 1.0, 2.0, 450)
        for other in (sc.path_for("observation", "temperature", 1.0, 2.0, 900),
                      sc.path_for("accumulation", "temperature", 1.0, 2.0, 450),
                      sc.path_for("observation", "dewpoint", 1.0, 2.0, 450),
                      sc.path_for("observation", "temperature", 1.1, 2.0, 450),
                      sc.path_for("observation", "temperature", 1.0, 2.1, 450)):
            assert other != base

    def test_coordinates_round_to_six_decimals(self, root):
        assert (sc.path_for("k", "a", 47.7787812, -116.7045912, 450)
                == sc.path_for("k", "a", 47.778781, -116.704591, 450))

    def test_format_version_invalidates_old_entries(self, root, monkeypatch):
        a = sc.path_for("k", "a", 1, 2, 450)
        monkeypatch.setattr(sc, "FORMAT_VERSION", sc.FORMAT_VERSION + 1)
        assert sc.path_for("k", "a", 1, 2, 450) != a

    def test_lives_under_the_series_root(self, root):
        assert sc.path_for("k", "a", 1, 2, 3).startswith(str(root))


class TestGet:
    def test_current_token_is_served(self, root):
        p = _put(root, 100.0)
        assert sc.get(p, 100.0) == DATA

    def test_newer_entry_than_our_view_is_served(self, root):
        p = _put(root, 200.0)
        assert sc.get(p, 100.0) == DATA

    def test_older_unmanaged_entry_is_a_miss(self, root):
        p = _put(root, 100.0, managed=False)
        assert sc.get(p, 200.0) is None

    def test_older_managed_entry_served_while_recent(self, root):
        p = _put(root, 100.0, managed=True)
        assert sc.get(p, 200.0) == DATA

    def test_older_managed_entry_expires(self, root):
        p = _put(root, 100.0, managed=True, built=time.time() - sc.STALE_SECONDS - 10)
        assert sc.get(p, 200.0) is None

    def test_unknown_current_token_serves_nothing(self, root):
        p = _put(root, 100.0, managed=True)
        assert sc.get(p, None) is None

    def test_missing_or_corrupt_file(self, root):
        assert sc.get(str(root / "nope.json.gz"), 1.0) is None
        bad = root / "bad.json.gz"
        bad.write_bytes(b"not gzip")
        assert sc.get(str(bad), 1.0) is None

    def test_is_fresh(self, root):
        p = _put(root, 100.0)
        assert sc.is_fresh(p, 100.0) and sc.is_fresh(p, 99.0)
        assert not sc.is_fresh(p, 101.0) and not sc.is_fresh(p, None)


class TestSave:
    def test_never_replaces_newer_data(self, root):
        p = _put(root, 200.0)
        assert sc.save(p, sc.pack({"x": 1}, 100.0), 100.0) == "stale"
        assert sc.get(p, 200.0) == DATA

    def test_same_data_is_a_dup(self, root):
        p = _put(root, 100.0)
        assert sc.save(p, sc.pack(DATA, 100.0), 100.0) == "dup"

    def test_newer_data_replaces(self, root):
        p = _put(root, 100.0)
        assert sc.save(p, sc.pack({"new": 1}, 200.0), 200.0) == "ok"
        assert sc.get(p, 200.0) == {"new": 1}

    def test_managed_upgrade_of_same_data_is_written(self, root):
        p = _put(root, 100.0, managed=False)
        assert sc.save(p, sc.pack(DATA, 100.0, True), 100.0, True) == "ok"
        assert sc.read(p)["managed"] is True

    def test_unmanaged_never_downgrades_managed(self, root):
        p = _put(root, 100.0, managed=True)
        assert sc.save(p, sc.pack(DATA, 100.0, False), 100.0, False) == "dup"
        assert sc.read(p)["managed"] is True

    def test_no_temp_files_left(self, root):
        _put(root, 100.0)
        assert [n for n in os.listdir(root) if ".tmp" in n] == []

    def test_read_cache_sees_a_rewritten_file(self, root):
        p = _put(root, 100.0)
        assert sc.get(p, 100.0) == DATA
        sc.save(p, sc.pack({"v": 2}, 300.0), 300.0)
        assert sc.get(p, 300.0) == {"v": 2}

    def test_concurrent_saves_leave_a_valid_newest_entry(self, root):
        p = sc.path_for("k", "a", 1, 2, 450)
        def go(t):
            sc.save(p, sc.pack({"t": t}, t), t)
        ts = [threading.Thread(target=go, args=(float(i),)) for i in range(1, 17)]
        [t.start() for t in ts]; [t.join() for t in ts]
        assert sc.read(p)["token"] == 16.0
