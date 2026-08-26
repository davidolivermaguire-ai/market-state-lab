"""Walk-forward engine: correctness of the bounded warm-up approximation.

`run_estimator(warmup=N)` replays only N rows before each scored block instead of the
whole prefix. That is an approximation, justified by the fact that every estimator
here is a filter whose dependence on the distant past decays geometrically. These
tests *measure* the resulting difference against full replay rather than assuming it
is negligible.
"""
from __future__ import annotations

import numpy as np
import pytest

from msl.engine.synthetic import make_regime_series
from msl.engine.walkforward import run_estimator
from msl.estimators.base import PROB_COLUMNS, get_estimator
from msl.features.core import build_features


@pytest.fixture(scope="module")
def features():
    px, _ = make_regime_series(n=1100, seed=17)
    return build_features(px)


def test_stateless_estimators_ignore_warmup(features) -> None:
    """Stateless rules never refit, so warm-up cannot change their output at all."""
    a = run_estimator(features, get_estimator("ma_cross"), min_train=600, warmup=250)
    b = run_estimator(features, get_estimator("ma_cross"), min_train=600, warmup=0)
    x = a[PROB_COLUMNS].astype(float).to_numpy()
    y = b[PROB_COLUMNS].astype(float).to_numpy()
    ok = ~(np.isnan(x).any(axis=1) | np.isnan(y).any(axis=1))
    np.testing.assert_allclose(x[ok], y[ok], atol=1e-12)


def _compare(features, name: str, warmup: int):
    kw = dict(min_train=600, refit_every=150, max_train=600)
    full = run_estimator(features, get_estimator(name), warmup=0, **kw)
    windowed = run_estimator(features, get_estimator(name), warmup=warmup, **kw)
    a = full[PROB_COLUMNS].astype(float).to_numpy()
    b = windowed[PROB_COLUMNS].astype(float).to_numpy()
    ok = ~(np.isnan(a).any(axis=1) | np.isnan(b).any(axis=1))
    assert ok.sum() > 100
    mean_abs = float(np.mean(np.abs(a[ok] - b[ok])))
    agree = float(np.mean(full["map_state"][ok].to_numpy() == windowed["map_state"][ok].to_numpy()))
    return mean_abs, agree


@pytest.mark.parametrize("name", ["hmm_gaussian", "bocpd"])
def test_bounded_warmup_matches_full_replay(features, name: str) -> None:
    """Filters that forget geometrically must be near-identical under a bounded run-up."""
    mean_abs, agree = _compare(features, name, warmup=750)
    assert mean_abs < 0.02, f"{name}: mean |Δprob| {mean_abs:.4f} too large"
    assert agree > 0.97, f"{name}: MAP agreement {agree:.3f} too low"


def test_full_replay_estimators_are_exact(features) -> None:
    """An estimator declaring `full_replay` must be unaffected by the warmup setting.

    The Kalman local-linear-trend filter is the case: maximum likelihood drives its
    slope variance toward zero, which turns the slope into an integral of the entire
    history rather than a fading average. Windowing it changed the answer materially,
    so it opts out — and this test pins that opt-out in place.
    """
    assert get_estimator("kalman_trend").full_replay is True
    mean_abs, agree = _compare(features, "kalman_trend", warmup=750)
    assert mean_abs == pytest.approx(0.0, abs=1e-12)
    assert agree == pytest.approx(1.0)
