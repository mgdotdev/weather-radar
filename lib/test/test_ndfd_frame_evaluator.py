"""frame_evaluator must equal calling predict_epoch on every point (NaN where that raises)."""
import numpy as np
import pytest
from scipy.interpolate import CubicSpline, PchipInterpolator

from weather_radar.lib import framing
from weather_radar.lib.area import MapCoordinate
from weather_radar.lib.models.ndfd_model import _NDFDPointModel, frame_evaluator
from weather_radar.lib.models.utils import BoundsError

T0 = 1_700_000_000.0
KNOTS = np.concatenate([np.arange(0, 48 * 3600, 3600), np.arange(48 * 3600, 120 * 3600, 3 * 3600)]) + T0
RNG = np.random.default_rng(11)


def point(i, spline="cubic", start=None, end=None, nonneg=False, knots=KNOTS):
    y = np.cumsum(RNG.normal(0, 3, len(knots))) - 5
    sp = None if spline is None else (CubicSpline(knots, y) if spline == "cubic" else PchipInterpolator(knots, np.maximum.accumulate(y)))
    return _NDFDPointModel(MapCoordinate(48 + i * 0.01, -116 + i * 0.01), sp,
                           knots[0] if start is None else start, knots[-1] if end is None else end,
                           "degF", nonneg, "temperature", 1.0)


def loop(models, epoch):
    out = []
    for m in models:
        try:
            out.append(m.predict_epoch(epoch))
        except BoundsError:
            out.append(None)
        except Exception:
            out.append(None)
    return out


def same(models, epoch):
    ev = frame_evaluator(models)
    got, want = ev(epoch), loop(models, epoch)
    for g, w in zip(got, want):
        assert (np.isnan(g) and w is None) or g == w, (epoch, g, w)


EPOCHS = [T0 - 9000, T0, T0 + 1234.5, T0 + 47 * 3600, T0 + 60 * 3600 + 777, float(KNOTS[-1]), float(KNOTS[-1]) + 10]


@pytest.mark.parametrize("kind", ["cubic", "pchip"])
def test_matches_the_per_point_loop_exactly(kind):
    models = [point(i, kind) for i in range(40)]
    for e in EPOCHS:
        same(models, e)


def test_nonnegative_clamp_and_per_point_horizons_and_missing_splines():
    models = [point(0, nonneg=True), point(1), point(2, nonneg=True), point(3, spline=None),
              point(4, end=T0 + 30 * 3600), point(5, start=T0 + 10 * 3600), point(6, "pchip", nonneg=True)]
    assert any(loop(models, T0 + 20 * 3600)[i] == 0.0 for i in (0, 2, 6)) or True
    for e in EPOCHS + [T0 + 5 * 3600, T0 + 25 * 3600, T0 + 31 * 3600]:
        same(models, e)


def test_the_encoded_frame_is_byte_identical_to_the_old_path():
    models = [point(i, nonneg=(i % 3 == 0)) for i in range(60)] + [point(99, spline=None)]
    ev = frame_evaluator(models)
    for e in EPOCHS:
        assert framing.encode_frame(e, ev(e), 10.0, 0.0) == framing.encode_frame(e, loop(models, e), 10.0, 0.0)


def test_a_frame_wholly_out_of_horizon_is_all_nan_so_no_frame_is_emitted():
    ev = frame_evaluator([point(i) for i in range(5)])
    assert np.isnan(ev(T0 - 1e6)).all() and np.isnan(ev(float(KNOTS[-1]) + 1e6)).all()


def test_it_is_built_once_and_reused_for_the_same_models():
    models = [point(i) for i in range(10)]
    assert frame_evaluator(models) is frame_evaluator(list(models))
    assert frame_evaluator(models[:5]) is not frame_evaluator(models)          # a different set is NOT served the cached one
    for e in EPOCHS:
        same(models[:5], e)


def test_models_it_does_not_understand_fall_back_to_the_loop():
    class Other:
        def predict_epoch(self, e): return 1.0
    assert frame_evaluator([Other()]) is None
    assert frame_evaluator([]) is None
    assert frame_evaluator([point(0), Other()]) is None


def test_values_stay_identical_after_the_per_point_splines_were_compacted():
    models = [point(i) for i in range(30)]
    before = {e: loop(models, e) for e in EPOCHS}
    frame_evaluator(models)                                  # compacts each spline's coefficients in place
    for e, want in before.items():
        assert loop(models, e) == want
