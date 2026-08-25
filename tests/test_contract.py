"""Contract tests: every estimator must honour the shared output schema."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from msl.engine.synthetic import make_regime_series
from msl.engine.walkforward import run_sweep, state_summary
from msl.estimators.base import OUTPUT_COLUMNS, PROB_COLUMNS, STATES, get_estimator, list_estimators
from msl.features.core import build_features


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    px, _ = make_regime_series(n=1000, seed=11)
    return build_features(px)


def test_registry_not_empty() -> None:
    assert list_estimators(), "no estimators registered"


@pytest.mark.parametrize("name", list_estimators())
def test_output_schema(features: pd.DataFrame, name: str) -> None:
    out = get_estimator(name).filter(features)

    assert list(out.columns) == OUTPUT_COLUMNS, f"{name}: wrong columns"
    assert out.index.equals(features.index), f"{name}: index not aligned to input"

    probs = out[PROB_COLUMNS].astype(float).dropna()
    assert len(probs) > 0, f"{name}: produced no estimates"
    np.testing.assert_allclose(probs.sum(axis=1).to_numpy(), 1.0, atol=1e-6)
    assert (probs.to_numpy() >= 0).all(), f"{name}: negative probability"

    assert set(out["map_state"].dropna()).issubset(set(STATES)), f"{name}: unknown state label"
    score = out["score"].astype(float).dropna()
    assert score.between(-1.0, 1.0).all(), f"{name}: score outside [-1, 1]"


def test_map_state_matches_argmax(features: pd.DataFrame) -> None:
    out = get_estimator("ma_cross").filter(features).dropna()
    expected = [STATES[i] for i in out[PROB_COLUMNS].astype(float).to_numpy().argmax(axis=1)]
    assert list(out["map_state"]) == expected


def test_sweep_is_tidy_and_multi_asset() -> None:
    """The harness must handle N assets x M methods and return one tidy frame."""
    px_a, _ = make_regime_series(n=900, seed=1)
    px_b, _ = make_regime_series(n=900, seed=2)
    res = run_sweep({"SYNTH_A": px_a, "SYNTH_B": px_b},
                    ["ma_cross", "return_sign", "always_range"], verbose=False)

    assert {"date", "symbol", "method", *OUTPUT_COLUMNS} <= set(res.columns)
    assert set(res["symbol"]) == {"SYNTH_A", "SYNTH_B"}
    assert set(res["method"]) == {"ma_cross", "return_sign", "always_range"}

    summ = state_summary(res)
    assert len(summ) == 6                      # 2 assets x 3 methods
    assert summ["flip_rate"].between(0, 1).all()
    # the null model never claims a trend
    null = summ[summ["method"] == "always_range"]
    assert (null["pct_range"] == 1.0).all()
