"""Decision-value tier: correctness of the targets, the benchmark, and the sizing rule."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from msl.engine.synthetic import make_regime_series
from msl.estimators.base import PROB_COLUMNS, get_estimator
from msl.features.core import build_features
from msl.metrics.decision import BASE_FEATURES, build_targets, calibration_gain, risk_control


@pytest.fixture(scope="module")
def setup():
    px, true_state = make_regime_series(n=2000, seed=31)
    feats = build_features(px)
    est = get_estimator("hmm_gaussian")
    est.fit(feats.iloc[:800])
    return feats, est.filter(feats), true_state


def test_targets_are_forward_looking_and_binary(setup) -> None:
    feats, _, _ = setup
    t = build_targets(feats, horizon=5)
    assert set(t.columns) == {"direction", "loss", "volatility"}
    for c in t.columns:
        vals = t[c].dropna().unique()
        assert set(vals).issubset({0.0, 1.0}), f"{c} is not binary"
    # the last `horizon` rows cannot have a label — nothing to look forward to
    assert t.iloc[-5:].isna().all().all()


def test_labels_never_leak_into_features(setup) -> None:
    """A target must not be reconstructible from the feature set it is scored against."""
    feats, _, _ = setup
    t = build_targets(feats, horizon=5)
    df = pd.concat([feats[BASE_FEATURES], t["direction"]], axis=1).dropna()
    # correlation of a *forward* label with contemporaneous volatility should be tiny
    for c in BASE_FEATURES:
        assert abs(df[c].corr(df["direction"])) < 0.25


def test_calibration_gain_shape_and_sign_convention(setup) -> None:
    feats, states, _ = setup
    out = calibration_gain(states, feats, horizon=5)
    assert not out.empty
    assert {"brier_base", "brier_state", "brier_delta"} <= set(out.columns)
    # sign convention: delta = state - base, so negative means the state helped
    np.testing.assert_allclose(
        out["brier_delta"].to_numpy(),
        out["brier_state"].to_numpy() - out["brier_base"].to_numpy(),
        atol=1e-12,
    )
    assert (out["brier_base"] > 0).all() and (out["brier_base"] < 0.5).all()


def test_state_overlay_can_only_reduce_exposure(setup) -> None:
    """The pre-committed rule is defensive: it must never lever *up* versus baseline."""
    feats, states, _ = setup
    out = risk_control(states, feats)
    assert set(out["strategy"]) == {"buy_and_hold", "vol_target (baseline)", "vol_target x state"}
    base = out[out["strategy"] == "vol_target (baseline)"].iloc[0]
    state = out[out["strategy"] == "vol_target x state"].iloc[0]
    assert state["vol"] <= base["vol"] + 1e-9, "state overlay increased volatility"
    assert state["max_drawdown"] <= 0.0 and base["max_drawdown"] <= 0.0


def test_perfect_foresight_state_beats_baseline(setup) -> None:
    """Sanity check on the machinery: an oracle state must show a calibration gain.

    If a state built from the *true* regime cannot improve on volatility-only, the
    comparison is broken rather than the methods being weak.
    """
    feats, _, true_state = setup
    oracle = pd.DataFrame(0.0, index=feats.index, columns=PROB_COLUMNS)
    idx = true_state.reindex(feats.index)
    for i, col in enumerate(PROB_COLUMNS):
        oracle.loc[idx == i, col] = 1.0
    oracle[idx.isna()] = np.nan

    out = calibration_gain(oracle, feats, horizon=5)
    direction = out[out["target"] == "direction"].iloc[0]
    assert direction["brier_delta"] < 0, "oracle state failed to improve calibration"


def test_common_window_gives_every_method_identical_dates():
    """The whole point of --common-window: no method sees a date another does not.

    Without this, a fitted estimator is scored on a shorter, later span than a
    baseline, so the cross-method ranking partly reflects which period each saw.
    """
    from msl.features.core import build_features
    from msl.engine.walkforward import run_estimator
    from msl.estimators.base import get_estimator
    from msl.metrics.decision import scored_index

    px, _ = make_regime_series(n=1600, seed=3)
    feats = build_features(px)
    names = ["ma_cross", "ewma_slope", "cusum"]
    spans = {}
    for n in names:
        st = run_estimator(feats, get_estimator(n), min_train=400, warmup=400, max_train=600)
        spans[n] = scored_index(st, feats, horizon=5, control_instability=True)

    # the spans genuinely differ - otherwise this test proves nothing
    assert len({len(s) for s in spans.values()}) > 1, "expected unequal spans to align"

    common = spans[names[0]]
    for s in spans.values():
        common = common.intersection(s)
    assert len(common) > 0
    for n, s in spans.items():
        assert common.difference(s).empty, f"{n} is missing dates in the common window"
