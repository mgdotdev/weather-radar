"""Evaluate MANY scipy piecewise-cubic splines at one epoch in a single numpy pass.

An animated layer needs one value per lattice point per frame. Each point owns a
fitted scipy spline (`CubicSpline`, or `PchipInterpolator` for accumulations), and
calling them one at a time costs a Python round trip per point per frame — about
300 ms for the ~44,000 points of a 450 x 400 km area, which made a full-window
bundle take many minutes.

`SplineBank` stacks the splines' polynomial coefficients once and evaluates every
point's polynomial with array arithmetic. It uses the SAME arithmetic, in the SAME
order, as scipy's own evaluator (ascending powers of `t - x[i]`; outside the knots
the end interval's polynomial simply continues, as scipy's default extrapolation
does), so the values are bit-identical to calling each spline.

Splines are grouped by knot vector (almost all points of a fit share one), so a
frame is a handful of array operations however many points there are.
"""

import numpy as np


class UnsupportedSpline(ValueError):
    """A spline this module does not evaluate (callers fall back to per-point calls)."""


def _check(sp):
    c = getattr(sp, "c", None)
    x = getattr(sp, "x", None)
    if c is None or x is None or getattr(c, "ndim", 0) != 2:
        raise UnsupportedSpline("expected a scalar-valued piecewise polynomial")
    if not getattr(sp, "extrapolate", True):
        raise UnsupportedSpline("non-extrapolating spline")


class SplineBank:
    """A fixed list of splines (entries may be None) evaluated together.

    `evaluate(t)` returns a float array aligned to the input list, NaN where an
    entry is None. With `compact=True` each spline is re-pointed at a view of the
    stacked coefficients (and one shared knot array), so the numbers are stored
    once instead of twice; every spline still evaluates to exactly the same value.
    (scipy copies a non-contiguous `c` back to a contiguous array the first time a
    spline is called on its own, so only splines evaluated individually — a few pins
    — give the saving back.)
    """

    def __init__(self, splines, compact=False):
        self.n = len(splines)
        groups = []                      # [indices, x, [c, ...], [spline, ...]]
        by_key = {}
        for i, sp in enumerate(splines):
            if sp is None:
                continue
            _check(sp)
            x = np.asarray(sp.x)
            key = (len(x), float(x[0]), float(x[-1]))
            for gi in by_key.get(key, ()):
                if np.array_equal(groups[gi][1], x):
                    groups[gi][0].append(i)
                    groups[gi][2].append(sp.c)
                    groups[gi][3].append(sp)
                    break
            else:
                by_key.setdefault(key, []).append(len(groups))
                groups.append([[i], x, [sp.c], [sp]])
        self._groups = []
        for ix, x, cs, sps in groups:
            stacked = np.stack(cs, axis=-1)          # (4, n_intervals, m): a frame reads c[k, j, :]
            self._groups.append((np.array(ix, dtype=np.intp), x, stacked))
            if compact:
                for k, sp in enumerate(sps):
                    sp.c = stacked[:, :, k]
                    sp.x = x
        self.has_any = bool(self._groups)

    def evaluate(self, t):
        out = np.full(self.n, np.nan)
        for ix, x, c in self._groups:
            j = int(np.clip(np.searchsorted(x, t, side="right") - 1, 0, len(x) - 2))
            s = t - x[j]
            res = np.zeros(c.shape[2])
            z = 1.0
            for kp in range(c.shape[0]):             # res += c[k-kp-1] * s**kp, scipy's order
                res = res + c[c.shape[0] - kp - 1, j] * z
                z = z * s
            out[ix] = res
        return out
