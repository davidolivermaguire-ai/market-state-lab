"""Significance machinery: HAC standard errors, DM test, and multiple-testing control."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from msl.engine.synthetic import make_regime_series
from msl.estimators.base import get_estimator
from msl.features.core import build_features
from msl.metrics.decision import (
    _nw_var_of_mean,
    calibration_gain,
    deflate,
    diebold_mariano,
    instability,
)


def test_hac_variance_exceeds_iid_under_serial_correlation() -> None:
    """Overlapping labels inflate the true standard error — HAC must reflect that.

    An iid standard error on serially-correlated differentials is precisely how noise
    gets published as a finding.
    """
    rng = np.random.default_rng(0)
    e = rng.normal(size=4000)
    x = pd.Series(e).rolling(5).mean().dropna().to_numpy()   # induces MA(4) dependence

    iid = float(np.var(x) / len(x))
    hac = _nw_var_of_mean(x, lag=5)
    assert hac > 2.0 * iid, f"HAC {hac:.3e} failed to exceed iid {iid:.3e}"


def test_dm_detects_a_genuine_loss_reduction() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(1.0, 0.1, 3000)
    better = base - 0.05                       # uniformly lower loss
    t, p = diebold_mariano(better, base, lag=5)
    assert t < -5 and p < 1e-6


def test_dm_has_approximately_correct_size_under_the_null() -> None:
    """Check the *rate* of false positives, not one draw.

    A single sample proves nothing about a test — with enough seeds some will reject
    by chance, which is the whole point of a 5% level. What must hold is that the
    rejection rate under the null is close to nominal. If it is materially above, the
    HAC lag is too short and every 'finding' downstream is inflated.
    """
    rejections = 0
    trials = 60
    for seed in range(trials):
        rng = np.random.default_rng(1000 + seed)
        a = rng.normal(1.0, 0.1, 2000)
        b = rng.normal(1.0, 0.1, 2000)
        t, _ = diebold_mariano(a, b, lag=5)
        rejections += abs(t) > 1.96
    size = rejections / trials
    assert size < 0.20, f"empirical size {size:.2f} is far above the nominal 0.05"


def test_deflated_threshold_grows_with_the_number_of_trials() -> None:
    """Searching more often should raise the bar, not leave it at 1.96."""
    small = deflate(pd.DataFrame({"p_value": [0.01] * 5, "t_stat": [-2.6] * 5,
                                  "brier_delta": [-0.01] * 5}))
    large = deflate(pd.DataFrame({"p_value": [0.01] * 200, "t_stat": [-2.6] * 200,
                                  "brier_delta": [-0.01] * 200}))
    assert large["t_threshold"].iloc[0] > small["t_threshold"].iloc[0]
    assert large["t_threshold"].iloc[0] == pytest.approx(np.sqrt(2 * np.log(200)), rel=1e-9)
    # a t of -2.6 clears a 5-trial search but not a 200-trial one
    assert small["survives_deflated"].all()
    assert not large["survives_deflated"].any()


def test_instability_is_causal_and_bounded() -> None:
    px, _ = make_regime_series(n=1200, seed=9)
    feats = build_features(px)
    est = get_estimator("ewma_slope")
    states = est.filter(feats)

    full = instability(states)
    prefix = instability(states.iloc[:900])
    a, b = full.iloc[:900].to_numpy(), prefix.to_numpy()
    ok = ~(np.isnan(a) | np.isnan(b))
    np.testing.assert_allclose(a[ok], b[ok], atol=1e-12)     # no look-ahead
    assert full.dropna().between(0, 1).all()


def test_controlling_instability_changes_the_comparison() -> None:
    """The control must actually bind: adding flip rate should move the measured gain."""
    px, _ = make_regime_series(n=2000, seed=12)
    feats = build_features(px)
    states = get_estimator("ewma_slope").filter(feats)

    plain = calibration_gain(states, feats, horizon=5)
    controlled = calibration_gain(states, feats, horizon=5, control_instability=True)
    assert not plain.empty and not controlled.empty
    merged = plain.merge(controlled, on="target", suffixes=("_plain", "_ctrl"))
    assert (merged["brier_delta_plain"] != merged["brier_delta_ctrl"]).any()
