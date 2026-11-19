"""Tests for interp.densify_block: densifying a rows x cols block of native
grid cells in grid (row, col) space."""

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from weather_radar.lib import interp


EPOCHS = np.array([0.0, 3600.0, 7200.0, 10800.0])


class _Const:
    """A 'spline' that returns a constant value at every epoch."""
    def __init__(self, v):
        self.v = float(v)

    def __call__(self, e):
        return np.full(np.shape(e), self.v, dtype=float)


def _fit_fn(vals, epochs):
    v = np.asarray(vals, dtype=float)
    if np.isfinite(v).sum() < 2:
        return None
    return CubicSpline(epochs, v)


def _identity_latlon(rows, cols):
    """Use (row, col) directly as (lat, lon) so positions are easy to check."""
    return np.asarray(rows, dtype=float), np.asarray(cols, dtype=float)


def _block(n_r, n_c, field, holes=()):
    """Splines + positions for an n_r x n_c block with value field(r, c);
    `holes` positions produce no spline (absent from the lists)."""
    splines, positions = [], []
    for r in range(n_r):
        for c in range(n_c):
            pos = r * n_c + c
            if pos in holes:
                continue
            splines.append(_Const(field(r, c)))
            positions.append(pos)
    return splines, np.asarray(positions)


class TestDensifyBlock:
    def test_dense_count_and_native_values_preserved(self):
        n_r, n_c, k = 5, 6, 3
        field = lambda r, c: 2.0 * r + 0.5 * c
        sp, pos = _block(n_r, n_c, field)
        lats, lons, dense = interp.densify_block(
            (n_r, n_c), pos, EPOCHS, sp, _fit_fn, k, _identity_latlon)
        n_dr, n_dc = (n_r - 1) * k + 1, (n_c - 1) * k + 1
        assert len(dense) == lats.size == lons.size == n_dr * n_dc
        # At native cell positions the densified value equals the native value
        # (the field is linear, so cubic resampling reproduces it exactly).
        for r in range(n_r):
            for c in range(n_c):
                i = (r * k) * n_dc + (c * k)
                assert lats[i] == pytest.approx(r) and lons[i] == pytest.approx(c)
                assert float(dense[i](0.0)) == pytest.approx(field(r, c), abs=1e-6)

    def test_points_between_cells_are_interpolated(self):
        n_r, n_c, k = 5, 5, 2
        field = lambda r, c: 10.0 * c
        sp, pos = _block(n_r, n_c, field)
        lats, lons, dense = interp.densify_block(
            (n_r, n_c), pos, EPOCHS, sp, _fit_fn, k, _identity_latlon)
        n_dc = (n_c - 1) * k + 1
        i = 0 * n_dc + 1                       # row 0, half-way to col 1
        assert lons[i] == pytest.approx(0.5)
        assert float(dense[i](0.0)) == pytest.approx(5.0, abs=1e-6)

    def test_holes_are_not_invented(self):
        """Dense points nearest a cell with no data get no spline."""
        n_r, n_c, k = 5, 5, 2
        hole = 2 * n_c + 2                     # center cell
        sp, pos = _block(n_r, n_c, lambda r, c: 1.0, holes={hole})
        lats, lons, dense = interp.densify_block(
            (n_r, n_c), pos, EPOCHS, sp, _fit_fn, k, _identity_latlon)
        n_dc = (n_c - 1) * k + 1
        center = (2 * k) * n_dc + (2 * k)      # dense point on the hole
        assert dense[center] is None
        corner = 0
        assert dense[corner] is not None
        # Only dense points rounding to the hole are dropped.
        near_hole = (np.rint(lats) == 2) & (np.rint(lons) == 2)
        assert all((d is None) == bool(h) for d, h in zip(dense, near_hole))

    def test_too_small_block_returns_none(self):
        sp, pos = _block(1, 4, lambda r, c: 1.0)
        assert interp.densify_block((1, 4), pos, EPOCHS, sp, _fit_fn, 3,
                                    _identity_latlon) is None

    def test_all_holes_returns_none(self):
        assert interp.densify_block((3, 3), np.asarray([]), EPOCHS, [], _fit_fn,
                                    2, _identity_latlon) is None

    def test_small_block_linear_fallback_is_exact_at_cells(self):
        """A 3x3 block can't use cubic; the linear fallback must keep every
        native cell's value exactly (no smoothing) and interpolate between."""
        k = 9
        field = lambda r, c: 5.0 if (r, c) == (1, 1) else 0.0
        sp, pos = _block(3, 3, field)
        lats, lons, dense = interp.densify_block(
            (3, 3), pos, EPOCHS, sp, _fit_fn, k, _identity_latlon)
        vals = np.array([float(d(0.0)) for d in dense])
        assert np.all(np.isfinite(vals))
        n_dc = (3 - 1) * k + 1
        for r in range(3):
            for c in range(3):
                assert vals[(r * k) * n_dc + (c * k)] == pytest.approx(field(r, c))
        # Half-way from the peak to an edge cell: linear -> exactly half.
        assert vals[(1 * k) * n_dc + (k + k // 2)] == pytest.approx(
            5.0 * (1 - (k // 2) / k))
