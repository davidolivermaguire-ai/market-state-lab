"""Tests for the recovery suite and the state report."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from msl.engine.synthetic import make_regime_series
from msl.estimators.base import get_estimator
from msl.features.core import build_features
from msl.metrics.recovery import recovery_metrics, run_recovery
from msl.report.state_report import state_report


def test_perfect_oracle_scores_perfectly() -> None:
    """An oracle that emits the true state must score ~1 — the metric's own sanity check."""
    from msl.estimators.base import OUTPUT_COLUMNS, STATES

    _, true_state = make_regime_series(n=800, seed=5)
    probs = np.zeros((len(true_state), 3))
    probs[np.arange(len(true_state)), true_state.to_numpy()] = 1.0
    oracle = pd.DataFrame(probs, index=true_state.index, columns=["p_down", "p_range", "p_up"])
    oracle["map_state"] = [STATES[i] for i in true_state]
    oracle["score"] = 0.0

    m = recovery_metrics(oracle[OUTPUT_COLUMNS], true_state)
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["balanced_accuracy"] == pytest.approx(1.0)
    assert m["brier"] == pytest.approx(0.0, abs=1e-9)
    assert m["detection_rate"] == pytest.approx(1.0)


def test_null_model_is_no_better_than_chance() -> None:
    """always_range never claims a trend, so it cannot recover the regime."""
    px, true_state = make_regime_series(n=1200, seed=6)
    out = get_estimator("always_range").filter(build_features(px))
    m = recovery_metrics(out, true_state)
    assert m["balanced_accuracy"] < 0.45
    assert m["ari"] == pytest.approx(0.0, abs=1e-6)


def test_a_real_method_beats_the_null_on_discrimination() -> None:
    """A trend rule must *discriminate* the simulated regime better than the null.

    Note the deliberate separation: discrimination (balanced accuracy, ARI) and
    calibration (Brier) are different axes, and a method can win one while losing
    the other. The moving-average rule takes confident positions, so when it is
    wrong it is punished harder than the hedging null — its Brier can be worse even
    though it identifies the regime far more often. That is a property to measure and
    report, not to hide, which is why the harness scores both.
    """
    px, true_state = make_regime_series(n=2000, seed=8)
    feats = build_features(px)
    good = recovery_metrics(get_estimator("ma_cross").filter(feats), true_state)
    null = recovery_metrics(get_estimator("always_range").filter(feats), true_state)

    assert good["balanced_accuracy"] > null["balanced_accuracy"]
    assert good["ari"] > null["ari"]
    assert good["detection_rate"] > 0.0        # the null never detects a change at all


def test_run_recovery_shape() -> None:
    per_seed, agg = run_recovery(methods=["ma_cross", "always_range"], seeds=(0, 1), n=900)
    assert len(per_seed) == 4                      # 2 methods x 2 seeds
    assert set(agg["method"]) == {"ma_cross", "always_range"}
    assert agg["balanced_accuracy"].is_monotonic_decreasing


def test_state_report_multi_asset() -> None:
    px_a, _ = make_regime_series(n=900, seed=21)
    px_b, _ = make_regime_series(n=900, seed=22)
    rep = state_report({"A": px_a, "B": px_b}, ["ma_cross", "return_sign", "ewma_slope"])

    assert list(rep["symbol"]) == ["A", "B"]
    assert {"consensus", "agreement", "disagreement", "p_up", "p_down"} <= set(rep.columns)
    assert rep["agreement"].between(0, 1).all()
    assert rep["disagreement"].between(0, 1).all()
    assert rep["p_consensus"].between(0, 1).all()
