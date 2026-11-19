"""Tests for the pyproj-based Lambert grid index (lib/gridindex.py).

The golden cells below were read from a REAL NBM CONUS ASNOW record's own
latitude/longitude arrays (eccodes), together with the real grid parameters, so
test_real_nbm_grid_golden_cells checks our projection against the actual
product — not against our own projection code. The remaining tests use small
synthetic grids to cover scan modes, clamping, validation, and the fallbacks.
"""

import sys
import types

import numpy as np
import pytest

pyproj = pytest.importorskip("pyproj")

from weather_radar.lib import gridindex
from weather_radar.lib.gridindex import LambertGridIndex


# Real NBM CONUS message keys (as eccodes reports them).
REAL_NBM = {
    "gridType": "lambert",
    "Nx": 2345, "Ny": 1597,
    "DxInMetres": 2539.703, "DyInMetres": 2539.703,
    "latitudeOfFirstGridPointInDegrees": 19.229,
    "longitudeOfFirstGridPointInDegrees": 233.7234,   # 0..360, as published
    "LaDInDegrees": 25.0, "LoVInDegrees": 265.0,
    "Latin1InDegrees": 25.0, "Latin2InDegrees": 25.0,
    "iScansNegatively": 0, "jScansPositively": 1, "jPointsAreConsecutive": 0,
    "shapeOfTheEarth": 1,
    "scaleFactorOfRadiusOfSphericalEarth": 0,
    "scaledValueOfRadiusOfSphericalEarth": 6371200,
}

# (flat index, lat, lon) read from the real message's own arrays.
GOLDEN = [
    (0,       19.229000, -126.276600),   # first cell
    (2344,    20.331746,  -69.208228),   # last column, first row
    (3742620, 53.050563, -138.373268),   # first column, last row
    (3744964, 54.372791,  -59.042190),   # last cell
    (1872482, 40.616690,  -98.355780),   # center
    (2959832, 48.777811, -121.813233),   # Mt. Baker area
    (2828678, 48.241682, -116.372800),   # North Idaho area
]


def _real_index():
    return LambertGridIndex.from_message(REAL_NBM.__getitem__)


class TestRealNBMGrid:
    def test_real_nbm_grid_golden_cells(self):
        idx = _real_index()
        assert idx is not None
        pts = np.array([[la, lo] for _, la, lo in GOLDEN])
        dist, got = idx.query(pts)
        assert list(got) == [i for i, _, _ in GOLDEN]
        # Goldens carry 6 decimals (~0.1 m); allow generous float/proj noise.
        assert np.all(dist < 50.0), dist

    def test_between_cells_picks_nearest(self):
        idx = _real_index()
        _, la, lo = GOLDEN[4]
        # Nudge ~0.3 cell east/north of the center cell: still the same cell.
        _, got = idx.query(np.array([[la + 0.006, lo + 0.008]]))
        assert got[0] == GOLDEN[4][0]

    def test_outside_domain_clamps_to_valid_index(self):
        idx = _real_index()
        far = np.array([[0.0, 0.0], [80.0, -120.0], [40.0, 20.0], [-45.0, 170.0]])
        _, got = idx.query(far)
        assert np.all((got >= 0) & (got < 2345 * 1597))

    def test_non_finite_points_do_not_crash(self):
        idx = _real_index()
        dist, got = idx.query(np.array([[np.nan, -100.0], [40.0, np.inf]]))
        assert np.all(np.isinf(dist))
        assert np.all((got >= 0) & (got < 2345 * 1597))

    def test_k_other_than_1_unsupported(self):
        with pytest.raises(NotImplementedError):
            _real_index().query(np.array([[40.0, -100.0]]), k=2)

    def test_longitude_convention_0_360_equals_negative(self):
        idx = _real_index()
        _, a = idx.query(np.array([[40.0, -98.0]]))
        _, b = idx.query(np.array([[40.0, 262.0]]))
        assert a[0] == b[0]


def _synthetic(nx, ny, *, i_negative, j_positive, j_consecutive, radius=6371200.0):
    """Build (index, lats, lons) for a small grid with the given scan mode. The
    lat/lon arrays are generated INDEPENDENTLY of query(): by walking the grid
    in the scan order the GRIB flags describe, then inverse-projecting."""
    from pyproj import Transformer
    kw = dict(nx=nx, ny=ny, dx=2539.703, dy=2539.703, lat1=30.0, lon1=-110.0,
              lad=25.0, lov=-95.0, latin1=25.0, latin2=25.0, radius=radius,
              i_negative=i_negative, j_positive=j_positive,
              j_consecutive=j_consecutive)
    idx = LambertGridIndex(**kw)
    lcc = ("+proj=lcc +lat_1=25 +lat_2=25 +lat_0=25 +lon_0=-95 +x_0=0 +y_0=0 "
           f"+R={radius} +units=m +no_defs")
    geo = f"+proj=longlat +R={radius} +no_defs"
    fwd = Transformer.from_crs(geo, lcc, always_xy=True)
    inv = Transformer.from_crs(lcc, geo, always_xy=True)
    x0, y0 = fwd.transform(-110.0, 30.0)
    sx = -1 if i_negative else 1
    sy = 1 if j_positive else -1
    lats = np.empty(nx * ny)
    lons = np.empty(nx * ny)
    for row in range(ny):
        for col in range(nx):
            lon, lat = inv.transform(x0 + sx * col * 2539.703,
                                     y0 + sy * row * 2539.703)
            flat = col * ny + row if j_consecutive else row * nx + col
            lats[flat], lons[flat] = lat, lon
    return idx, lats, lons


class TestScanModes:
    @pytest.mark.parametrize("i_neg", [False, True])
    @pytest.mark.parametrize("j_pos", [False, True])
    @pytest.mark.parametrize("j_cons", [False, True])
    def test_every_cell_maps_to_itself(self, i_neg, j_pos, j_cons):
        idx, lats, lons = _synthetic(12, 9, i_negative=i_neg, j_positive=j_pos,
                                     j_consecutive=j_cons)
        dist, got = idx.query(np.column_stack([lats, lons]))
        assert np.array_equal(got, np.arange(12 * 9))
        assert np.all(dist < 1.0)
        assert idx.validate(lats, lons)


class TestValidation:
    def test_wrong_earth_radius_is_detected(self):
        # Arrays generated for a 6378137 m sphere; index built with 6371200 m.
        # Over a ~750 km grid the 0.11% scale error drifts > 0.1 cell.
        _, lats, lons = _synthetic(300, 300, i_negative=False, j_positive=True,
                                   j_consecutive=False, radius=6378137.0)
        wrong, _, _ = _synthetic(300, 300, i_negative=False, j_positive=True,
                                 j_consecutive=False, radius=6371200.0)
        assert wrong.validate(lats, lons) is False

    def test_wrong_scan_mode_is_detected(self):
        _, lats, lons = _synthetic(20, 15, i_negative=False, j_positive=True,
                                   j_consecutive=False)
        flipped, _, _ = _synthetic(20, 15, i_negative=False, j_positive=False,
                                   j_consecutive=False)
        assert flipped.validate(lats, lons) is False

    def test_size_mismatch_fails_validation(self):
        idx, lats, lons = _synthetic(10, 10, i_negative=False, j_positive=True,
                                     j_consecutive=False)
        assert idx.validate(lats[:-1], lons[:-1]) is False


class TestFromMessage:
    def test_non_lambert_grid_unsupported(self):
        assert LambertGridIndex.from_message(
            {**REAL_NBM, "gridType": "regular_ll"}.__getitem__) is None

    def test_ellipsoidal_earth_unsupported(self):
        assert LambertGridIndex.from_message(
            {**REAL_NBM, "shapeOfTheEarth": 2}.__getitem__) is None

    def test_sphere_code_6_uses_standard_radius(self):
        m = {**REAL_NBM, "shapeOfTheEarth": 6}
        assert gridindex._earth_radius(m.__getitem__) == 6371229.0

    def test_scaled_radius_for_code_1(self):
        m = {**REAL_NBM, "scaleFactorOfRadiusOfSphericalEarth": 2,
             "scaledValueOfRadiusOfSphericalEarth": 637120000}
        assert gridindex._earth_radius(m.__getitem__) == pytest.approx(6371200.0)


class _FakeEccodes(types.ModuleType):
    """Minimal eccodes stand-in so index_from_message can be exercised without
    the real library: a tiny 3x3 'message'. Counts lat/lon array reads."""
    array_reads = 0

    def codes_get_array(self, gid, key):
        type(self).array_reads += 1
        lats = np.repeat([40.0, 40.1, 40.2], 3)
        lons = np.tile([-100.0, -99.9, -99.8], 3)
        return lats if key == "latitudes" else lons

    def codes_get(self, gid, key):
        raise KeyError(key)   # from_message won't be reached (patched)


class TestFallback:
    @pytest.fixture
    def fake_ec(self, monkeypatch):
        _FakeEccodes.array_reads = 0
        monkeypatch.setitem(sys.modules, "eccodes", _FakeEccodes("eccodes"))
        monkeypatch.setattr(gridindex, "_VALIDATED", {})   # isolate the cache
        return _FakeEccodes

    def test_unsupported_grid_falls_back_to_kdtree(self, fake_ec, monkeypatch):
        monkeypatch.setattr(LambertGridIndex, "from_message",
                            classmethod(lambda cls, get: None))
        idx = gridindex.index_from_message(object())
        _, got = idx.query(np.array([[40.2, -99.8]]), k=1)
        assert int(got[0]) == 8                       # nearest = last cell
        assert type(idx).__name__ == "cKDTree"

    def test_failed_validation_falls_back_to_kdtree(self, fake_ec, monkeypatch):
        class Bad:
            def validate(self, lats, lons):
                return False
        monkeypatch.setattr(LambertGridIndex, "from_message",
                            classmethod(lambda cls, get: Bad()))
        idx = gridindex.index_from_message(object())
        assert type(idx).__name__ == "cKDTree"

    def test_exception_building_index_falls_back_to_kdtree(self, fake_ec, monkeypatch):
        def boom(cls, get):
            raise RuntimeError("projection exploded")
        monkeypatch.setattr(LambertGridIndex, "from_message", classmethod(boom))
        idx = gridindex.index_from_message(object())
        assert type(idx).__name__ == "cKDTree"

    def test_exception_during_validation_falls_back_to_kdtree(self, fake_ec, monkeypatch):
        class Explodes:
            def validate(self, lats, lons):
                raise ValueError("bad arrays")
        monkeypatch.setattr(LambertGridIndex, "from_message",
                            classmethod(lambda cls, get: Explodes()))
        idx = gridindex.index_from_message(object())
        assert type(idx).__name__ == "cKDTree"

    def test_valid_index_is_used_when_it_validates(self, fake_ec, monkeypatch):
        class Good:
            def validate(self, lats, lons):
                return True
        good = Good()
        monkeypatch.setattr(LambertGridIndex, "from_message",
                            classmethod(lambda cls, get: good))
        assert gridindex.index_from_message(object()) is good


class TestValidatedIndexCache:
    """One NBM grid is shared by every element, so after the first message
    validates, later messages on the same grid must skip the (slow, ~1 s) full
    lat/lon array reads entirely."""

    @pytest.fixture(autouse=True)
    def _fresh(self, monkeypatch):
        _FakeEccodes.array_reads = 0
        monkeypatch.setitem(sys.modules, "eccodes", _FakeEccodes("eccodes"))
        monkeypatch.setattr(gridindex, "_VALIDATED", {})

    def _patch(self, monkeypatch, sig, validates=True):
        made = []

        class Idx:
            signature = sig
            def validate(self, lats, lons):
                return validates

        def factory(cls, get):
            i = Idx()
            made.append(i)
            return i

        monkeypatch.setattr(LambertGridIndex, "from_message", classmethod(factory))
        return made

    def test_second_message_on_same_grid_skips_array_reads(self, monkeypatch):
        made = self._patch(monkeypatch, sig=("gridA",))
        first = gridindex.index_from_message(object())
        reads_after_first = _FakeEccodes.array_reads
        assert reads_after_first == 2                 # latitudes + longitudes
        second = gridindex.index_from_message(object())
        assert second is first                        # cached, validated index
        assert _FakeEccodes.array_reads == reads_after_first   # no new reads

    def test_different_grid_is_validated_separately(self, monkeypatch):
        self._patch(monkeypatch, sig=("gridA",))
        gridindex.index_from_message(object())
        self._patch(monkeypatch, sig=("gridB",))
        gridindex.index_from_message(object())
        assert _FakeEccodes.array_reads == 4          # each grid read once

    def test_failed_validation_is_not_cached(self, monkeypatch):
        self._patch(monkeypatch, sig=("gridA",), validates=False)
        assert type(gridindex.index_from_message(object())).__name__ == "cKDTree"
        assert gridindex._VALIDATED == {}

    def test_real_grid_signature_is_stable(self):
        a = LambertGridIndex.from_message(REAL_NBM.__getitem__)
        b = LambertGridIndex.from_message(REAL_NBM.__getitem__)
        assert a.signature == b.signature
        other = LambertGridIndex.from_message(
            {**REAL_NBM, "DxInMetres": 3000.0}.__getitem__)
        assert other.signature != a.signature


# --- Box query (cells covering a lat/lon box) --------------------------------

def _box_around(lat, lon, half_deg):
    return lat - half_deg, lat + half_deg, lon - half_deg, lon + half_deg


class TestBoxCells:
    def test_block_contains_golden_cells_at_their_real_coordinates(self):
        """Inverse projection checked against the REAL message's lat/lon."""
        idx = _real_index()
        for flat, la, lo in GOLDEN[4:]:          # interior goldens
            s, n, w, e = _box_around(la, lo, 0.05)
            blk = idx.box_cells(s, n, w, e)
            assert blk is not None
            pos = np.where(blk.flat == flat)[0]
            assert pos.size == 1, f"golden cell {flat} missing from block"
            assert blk.lats[pos[0]] == pytest.approx(la, abs=1e-5)
            assert blk.lons[pos[0]] == pytest.approx(lo, abs=1e-5)

    def test_every_point_in_box_maps_into_block(self):
        """Coverage: the nearest cell of ANY point inside the box is in the
        block (the grid is rotated, so this checks the edge projection)."""
        idx = _real_index()
        s, n, w, e = 47.9, 48.5, -116.9, -115.9   # ~North Idaho, ~70 x 70 km
        blk = idx.box_cells(s, n, w, e)
        rng = np.random.default_rng(1)
        pts = np.column_stack([rng.uniform(s, n, 5000), rng.uniform(w, e, 5000)])
        corners = np.array([[s, w], [s, e], [n, w], [n, e]])
        _, got = idx.query(np.vstack([pts, corners]))
        assert np.isin(got, blk.flat).all()

    def test_block_cells_round_trip_through_query(self):
        idx = _real_index()
        blk = idx.box_cells(48.0, 48.4, -116.6, -116.1)
        _, got = idx.query(np.column_stack([blk.lats, blk.lons]))
        assert np.array_equal(got, blk.flat)

    def test_block_is_regular_and_row_major(self):
        idx = _real_index()
        blk = idx.box_cells(48.0, 48.4, -116.6, -116.1)
        n_r, n_c = blk.shape
        assert blk.flat.size == n_r * n_c == blk.lats.size == blk.lons.size
        grid = blk.flat.reshape(n_r, n_c)
        assert np.all(np.diff(grid, axis=1) == 1)          # next col
        assert np.all(np.diff(grid, axis=0) == idx.nx)      # next row
        assert len(np.unique(blk.flat)) == blk.flat.size    # no duplicates

    def test_box_outside_grid_returns_none(self):
        idx = _real_index()
        assert idx.box_cells(-10.0, -5.0, 10.0, 15.0) is None   # S. Atlantic

    def test_unprojectable_box_returns_none(self):
        # A whole-globe box includes points Lambert can't project (non-finite).
        assert _real_index().box_cells(-90.0, 90.0, -180.0, 179.99) is None

    def test_box_partly_outside_is_clipped(self):
        idx = _real_index()
        _, la, lo = GOLDEN[0]                     # first cell (grid corner)
        blk = idx.box_cells(la - 2.0, la + 0.2, lo - 2.0, lo + 0.2)
        assert blk is not None
        assert blk.rows[0] == 0 and blk.cols[0] == 0
        assert 0 in blk.flat

    def test_non_finite_box_returns_none(self):
        assert _real_index().box_cells(np.nan, 48.0, -117.0, -116.0) is None

    def test_antimeridian_box_does_not_crash(self):
        assert _real_index().box_cells(50.0, 51.0, 179.0, -179.0) is None

    def test_fractional_latlon_lies_between_cells(self):
        idx = _real_index()
        blk = idx.box_cells(48.0, 48.4, -116.6, -116.1)
        la0, lo0 = blk.latlon(0.0, 0.0)
        la1, lo1 = blk.latlon(1.0, 1.0)
        lam, lom = blk.latlon(0.5, 0.5)
        assert min(la0, la1) < lam < max(la0, la1)
        assert min(lo0, lo1) < lom < max(lo0, lo1)
        assert la0 == pytest.approx(blk.lats[0]) and lo0 == pytest.approx(blk.lons[0])

    @pytest.mark.parametrize("i_neg", [False, True])
    @pytest.mark.parametrize("j_pos", [False, True])
    @pytest.mark.parametrize("j_cons", [False, True])
    def test_scan_modes_round_trip(self, i_neg, j_pos, j_cons):
        idx, lats, lons = _synthetic(30, 25, i_negative=i_neg, j_positive=j_pos,
                                     j_consecutive=j_cons)
        # A box around the middle of the synthetic grid.
        mid = (30 * 25) // 2
        s, n, w, e = _box_around(lats[mid], lons[mid], 0.08)
        blk = idx.box_cells(s, n, w, e)
        assert blk is not None and mid in blk.flat
        # Block coordinates agree with the independently generated arrays.
        assert np.allclose(blk.lats, lats[blk.flat], atol=1e-7)
        assert np.allclose(blk.lons, lons[blk.flat], atol=1e-7)


# --- Alternating-row scanning (GRIB scanningMode bit 0x10) -------------------

def _keys(alt=1, nx=4, ny=3, consecutive=0):
    return {"alternativeRowScanning": alt, "Nx": nx, "Ny": ny,
            "jPointsAreConsecutive": consecutive}.__getitem__


def _boustrophedon(grid):
    """Store a uniform (n_lines, length) grid the way GRIB does with
    alternating-row scanning: every other line reversed."""
    s = np.array(grid, dtype=float).copy()
    s[1::2] = s[1::2, ::-1]
    return s.ravel()


class TestUniformScanValues:
    TRUTH = np.arange(12, dtype=float).reshape(3, 4) * 1.0   # ny=3 rows, nx=4

    def test_alternate_rows_are_restored(self):
        stored = _boustrophedon(self.TRUTH)
        assert not np.array_equal(stored, self.TRUTH.ravel())
        got = gridindex.uniform_scan_values(stored, _keys())
        assert np.array_equal(got, self.TRUTH.ravel())

    def test_flag_off_is_unchanged(self):
        v = self.TRUTH.ravel()
        assert gridindex.uniform_scan_values(v, _keys(alt=0)) is v

    def test_columns_consecutive_reverses_alternate_columns(self):
        truth = np.arange(12, dtype=float).reshape(4, 3)      # nx=4 lines of ny=3
        stored = _boustrophedon(truth)
        got = gridindex.uniform_scan_values(stored, _keys(consecutive=1))
        assert np.array_equal(got, truth.ravel())

    def test_size_mismatch_is_unchanged(self):
        v = np.arange(11, dtype=float)
        assert gridindex.uniform_scan_values(v, _keys()) is v

    def test_missing_keys_is_unchanged(self):
        v = np.arange(12, dtype=float)
        assert gridindex.uniform_scan_values(v, {}.__getitem__) is v

    def test_input_not_modified(self):
        stored = _boustrophedon(self.TRUTH)
        before = stored.copy()
        gridindex.uniform_scan_values(stored, _keys())
        assert np.array_equal(stored, before)

    def test_real_nbm_layout_fixes_row_stripes(self):
        """A smooth field stored with NBM's flags: decoded naively it's striped
        row to row; restored it's smooth again."""
        ny, nx = 40, 60
        yy, xx = np.mgrid[0:ny, 0:nx]
        truth = 280.0 + 0.05 * xx + 0.03 * yy
        stored = _boustrophedon(truth)
        naive = stored.reshape(ny, nx)
        fixed = gridindex.uniform_scan_values(stored, _keys(nx=nx, ny=ny)).reshape(ny, nx)
        assert np.abs(np.diff(naive, axis=0)).mean() > 1.0
        assert np.abs(np.diff(fixed, axis=0)).mean() == pytest.approx(0.03)


class TestUniformScanLatLon:
    def _uniform_geometry(self, nx=4, ny=3):
        lats = np.repeat(np.linspace(40.0, 40.2, ny), nx)
        lons = np.tile(np.linspace(-100.0, -99.7, nx), ny)
        return lats, lons

    def test_uniform_geometry_left_alone(self):
        lats, lons = self._uniform_geometry()
        la, lo = gridindex.uniform_scan_latlon(lats, lons, _keys())
        assert np.array_equal(la, lats) and np.array_equal(lo, lons)

    def test_alternating_geometry_is_made_uniform(self):
        lats, lons = self._uniform_geometry()
        alt_lats = _boustrophedon(lats.reshape(3, 4))
        alt_lons = _boustrophedon(lons.reshape(3, 4))
        la, lo = gridindex.uniform_scan_latlon(alt_lats, alt_lons, _keys())
        assert np.array_equal(la, lats) and np.array_equal(lo, lons)

    def test_flag_off_left_alone(self):
        lats, lons = self._uniform_geometry()
        alt_lons = _boustrophedon(lons.reshape(3, 4))
        la, lo = gridindex.uniform_scan_latlon(lats, alt_lons, _keys(alt=0))
        assert np.array_equal(lo, alt_lons)
