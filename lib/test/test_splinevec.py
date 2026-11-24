"""SplineBank evaluates many scipy splines at once — and must give EXACTLY what calling
each spline gives (the bundles and live streams depend on the numbers not changing)."""
import numpy as np
import pytest
from scipy.interpolate import CubicSpline, PchipInterpolator

from weather_radar.lib.splinevec import SplineBank, UnsupportedSpline

RNG = np.random.default_rng(7)
KNOTS = np.concatenate([np.arange(0, 48 * 3600, 3600), np.arange(48 * 3600, 120 * 3600, 3 * 3600)]).astype(float) + 1.7e9


def _series(knots=KNOTS, kind=CubicSpline):
    y = np.cumsum(RNG.normal(0, 1, len(knots))) + 40
    return kind(knots, np.maximum.accumulate(y) if kind is PchipInterpolator else y)


def _epochs(knots=KNOTS):
    inside = RNG.uniform(knots[0], knots[-1], 25)
    return [*knots[:4], knots[-1], *inside, knots[0] - 5000.0, knots[-1] + 7200.0]   # on knots, between, outside


@pytest.mark.parametrize("kind", [CubicSpline, PchipInterpolator])
def test_every_value_is_bit_identical_to_calling_the_spline(kind):
    splines = [_series(kind=kind) for _ in range(50)]
    bank = SplineBank(splines)
    for t in _epochs():
        got = bank.evaluate(t)
        want = np.array([float(sp(t)) for sp in splines])
        assert got.tolist() == want.tolist(), t              # exact, not approx


def test_mixed_knot_vectors_and_missing_splines():
    short = KNOTS[:60]
    splines = [_series(), None, _series(short), _series(), None, _series(short), _series(KNOTS[::2])]
    bank = SplineBank(splines)
    for t in _epochs():
        got = bank.evaluate(t)
        for i, sp in enumerate(splines):
            if sp is None:
                assert np.isnan(got[i])
            else:
                assert got[i] == float(sp(t)), (i, t)


def test_an_empty_or_all_none_bank_is_all_nan():
    assert SplineBank([]).evaluate(1.7e9).shape == (0,)
    b = SplineBank([None, None])
    assert not b.has_any and np.isnan(b.evaluate(1.7e9)).all()


def test_compacting_keeps_every_splines_own_values_and_stops_duplicating_the_numbers():
    splines = [_series() for _ in range(20)]
    held = sum(sp.c.nbytes for sp in splines)
    splines = [_series() for _ in range(20)]                       # fresh (scipy un-compacts a spline it evaluates)
    want = {t: [float(sp(t)) for sp in splines] for t in _epochs()}
    bank = SplineBank(splines, compact=True)
    stacked = bank._groups[0][2]
    # checked BEFORE evaluating any spline individually: scipy copies a non-contiguous c back
    # to a contiguous array the first time a spline is called, which is harmless (the hot paths
    # use the bank, and only a few pins' points are ever evaluated one at a time).
    assert all(np.shares_memory(sp.c, stacked) for sp in splines)  # views, not copies
    assert all(sp.x is splines[0].x for sp in splines)             # one shared knot array
    assert held == stacked.nbytes                                   # the same bytes, held once
    for t, w in want.items():
        assert bank.evaluate(t).tolist() == w
        assert [float(sp(t)) for sp in splines] == w                # and each spline still answers identically


@pytest.mark.parametrize("make", [
    lambda: CubicSpline(KNOTS[:10], np.random.default_rng(1).normal(size=(10, 3))),   # vector-valued
    lambda: CubicSpline(KNOTS[:10], np.arange(10.0), extrapolate=False),
])
def test_splines_it_cannot_evaluate_are_refused_not_silently_wrong(make):
    with pytest.raises(UnsupportedSpline):
        SplineBank([make()])
