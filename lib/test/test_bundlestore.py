"""Tests for the layered, append-only bundle store (lib/bundlestore.py)."""

import os
import struct
import time

import pytest

from weather_radar.lib import bundlestore as bs
from weather_radar.lib import framing as F

COORDS = [[-116.9, 47.2], [-116.8, 47.2]]


def _items(epochs, base=0):
    """(epoch, payload) pairs whose first value encodes the epoch, so a read
    can verify WHICH layer's frame it got."""
    scale, offset = F.choose_scale("AQI")
    return [(e, F.encode_frame(e, [e % 300 + base, 1], scale, offset)) for e in epochs]


def _layer(epochs, token=100.0, base=0, built=None, coords=COORDS, with_range=False):
    scale, offset = F.choose_scale("AQI")
    meta = {"type": "range", "uom": "AQI", "coords": coords,
            "enc": F.ENC_I16, "scale": scale, "offset": offset,
            "sentinel": F.SENTINEL,
            "meta": {"source_mtime": token}}
    if built is not None:
        meta["meta"]["built_ts"] = built
    if with_range:
        meta["range"] = {"min": 0, "max": 300}
    return bs.pack_layer(meta, _items(epochs, base))


def _save(d, epochs, **kw):
    token = kw.get("token", 100.0)
    data, eps = _layer(epochs, **kw)
    return bs.save_layer(str(d), data, token, eps)


def _read(d, wanted, token=100.0):
    ls = bs.usable_layers(str(d), token)
    out = {}
    scale, _off = F.choose_scale("AQI")
    for e, payload in bs.merged_frames(ls, wanted):
        out[e] = struct.unpack_from("<2h", payload, 8)[0] / scale
    return out, ls


class TestLayerFiles:
    def test_pack_sorts_dedupes_and_records_epochs(self):
        data, eps = bs.pack_layer({"uom": "AQI", "coords": COORDS, "meta": {}},
                                  _items([30, 0, 60, 30]))
        assert eps == [0, 30, 60]

    def test_header_is_readable_without_frames(self, tmp_path):
        _save(tmp_path, [0, 30, 60])
        (seq, path), = bs.layer_paths(str(tmp_path))
        meta = bs.read_meta(path)
        assert seq == 1 and meta["epochs"] == [0, 30, 60]
        assert meta["coords"] == COORDS and bs.token_of(meta) == 100.0

    def test_frames_in_file_are_ascending(self, tmp_path):
        _save(tmp_path, [60, 0, 30])
        (_, path), = bs.layer_paths(str(tmp_path))
        got = [e for e, _ in bs.iter_frames(path, {0, 30, 60})]
        assert got == [0, 30, 60]


class TestMergingLayers:
    """The scenario from the design: three requests, three files."""

    def _three(self, tmp_path):
        assert _save(tmp_path, [0, 30, 60], base=1000) == "ok"          # dt=30
        assert _save(tmp_path, [10, 20, 40], base=2000) == "ok"         # dt=10
        assert _save(tmp_path, [5, 15, 25, 35], base=3000) == "ok"      # dt=5

    def test_each_epoch_comes_from_the_layer_that_holds_it(self, tmp_path):
        self._three(tmp_path)
        got, ls = _read(tmp_path, [0, 5, 10, 15, 20, 25, 30, 35, 40, 60])
        assert len(ls) == 3
        # value = (epoch % 300) + base, base = 1000 / 2000 / 3000 for layers 1/2/3
        src = {e: int(v) // 1000 for e, v in got.items()}
        assert src == {0: 1, 30: 1, 60: 1,
                       10: 2, 20: 2, 40: 2,
                       5: 3, 15: 3, 25: 3, 35: 3}

    def test_output_is_one_ascending_stream(self, tmp_path):
        self._three(tmp_path)
        ls = bs.usable_layers(str(tmp_path), 100.0)
        order = [e for e, _ in bs.merged_frames(ls, [60, 5, 0, 40, 25, 10, 35])]
        assert order == sorted(order) == [0, 5, 10, 25, 35, 40, 60]

    def test_coarse_request_touches_only_the_layers_it_needs(self, tmp_path):
        self._three(tmp_path)
        ls = bs.usable_layers(str(tmp_path), 100.0)
        assigned = bs.plan(ls, [0, 30, 60])               # dt=30 -> layer 1 only
        assert len(assigned) == 1
        assigned = bs.plan(ls, [0, 10, 20, 30])           # dt=10 -> layers 1 + 2
        assert len(assigned) == 2

    def test_epoch_in_no_layer_is_simply_absent(self, tmp_path):
        self._three(tmp_path)
        got, _ = _read(tmp_path, [0, 7, 30])
        assert sorted(got) == [0, 30]                     # 7 is for the live path

    def test_first_layer_wins_when_epochs_overlap(self, tmp_path):
        _save(tmp_path, [0, 30], base=1000)
        _save(tmp_path, [0, 10], base=2000)               # 0 again, plus new 10
        ls = bs.usable_layers(str(tmp_path), 100.0)
        plan = bs.plan(ls, [0, 10, 30])
        by_layer = sorted(len(v) for v in plan.values())
        assert by_layer == [1, 2]                         # 0,30 from #1; 10 from #2

    def test_stops_reading_when_everything_wanted_is_found(self, tmp_path, monkeypatch):
        _save(tmp_path, list(range(0, 3000, 30)))
        (_, path), = bs.layer_paths(str(tmp_path))
        reads = []
        real = F.decode_frame_epoch
        monkeypatch.setattr(F, "decode_frame_epoch",
                            lambda p: (reads.append(1), real(p))[1])
        list(bs.iter_frames(path, {0, 30}))
        assert len(reads) == 2                            # not all 100 frames


class TestSaveRules:
    def test_sequence_numbers_increase(self, tmp_path):
        _save(tmp_path, [0])
        _save(tmp_path, [1])
        assert [s for s, _ in bs.layer_paths(str(tmp_path))] == [1, 2]

    def test_duplicate_layer_is_not_written(self, tmp_path):
        assert _save(tmp_path, [0, 30, 60]) == "ok"
        assert _save(tmp_path, [0, 60]) == "dup"          # nothing new
        assert len(bs.layer_paths(str(tmp_path))) == 1

    def test_partially_new_layer_is_written(self, tmp_path):
        _save(tmp_path, [0, 30])
        assert _save(tmp_path, [30, 60]) == "ok"

    def test_older_token_is_refused(self, tmp_path):
        _save(tmp_path, [0], token=200.0)
        assert _save(tmp_path, [10], token=100.0) == "stale"
        assert len(bs.layer_paths(str(tmp_path))) == 1

    def test_newer_token_replaces_the_old_generation(self, tmp_path):
        _save(tmp_path, [0, 30], token=100.0)
        _save(tmp_path, [10], token=100.0)
        assert _save(tmp_path, [0, 30], token=200.0) == "ok"
        ls = bs.layers(str(tmp_path))
        assert len(ls) == 1 and bs.token_of(ls[0][2]) == 200.0

    def test_layer_cap(self, tmp_path):
        for i in range(4):
            data, eps = _layer([i])
            assert bs.save_layer(str(tmp_path), data, 100.0, eps, max_layers=4) == "ok"
        data, eps = _layer([99])
        assert bs.save_layer(str(tmp_path), data, 100.0, eps, max_layers=4) == "full"

    def test_unversioned_layer_cannot_join_a_versioned_bundle(self, tmp_path):
        _save(tmp_path, [0], token=100.0)
        data, eps = _layer([5])
        assert bs.save_layer(str(tmp_path), data, None, eps) == "stale"

    def test_no_temp_files_left_behind(self, tmp_path):
        _save(tmp_path, [0])
        assert [n for n in os.listdir(tmp_path) if n.startswith(".tmp")] == []

    def test_concurrent_saves_get_distinct_slots(self, tmp_path):
        import threading
        out = []

        def go(i):
            out.append(_save(tmp_path, [i * 10, i * 10 + 1]))

        ts = [threading.Thread(target=go, args=(i,)) for i in range(8)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        assert out.count("ok") == 8
        assert sorted(s for s, _ in bs.layer_paths(str(tmp_path))) == list(range(1, 9))


class TestGenerations:
    def test_matching_token_layers_are_used(self, tmp_path):
        _save(tmp_path, [0], token=100.0)
        assert len(bs.usable_layers(str(tmp_path), 100.0)) == 1

    def test_stale_generation_served_when_recent(self, tmp_path):
        _save(tmp_path, [0], token=100.0, built=time.time())
        # current data moved on (token 200) but nothing rebuilt yet
        assert len(bs.usable_layers(str(tmp_path), 200.0)) == 1

    def test_stale_generation_dropped_when_old(self, tmp_path):
        _save(tmp_path, [0], token=100.0, built=time.time() - 10_000)
        assert bs.usable_layers(str(tmp_path), 200.0) == []

    def test_unknown_current_token_uses_newest_generation(self, tmp_path):
        _save(tmp_path, [0], token=100.0, built=time.time() - 10_000)
        assert len(bs.usable_layers(str(tmp_path), None)) == 1

    def test_layers_with_different_point_counts_are_not_mixed(self, tmp_path):
        _save(tmp_path, [0, 30])
        _save(tmp_path, [10], coords=COORDS[:1])          # 1 point vs 2
        assert len(bs.usable_layers(str(tmp_path), 100.0)) == 1

    def test_missing_dir_is_empty(self, tmp_path):
        assert bs.usable_layers(str(tmp_path / "nope"), 1.0) == []
