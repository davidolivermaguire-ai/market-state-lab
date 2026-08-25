"""The look-ahead guard — the most important test in the repo.

A filtered estimate at time t must not change when future data arrives. If it does,
the estimator is smoothing (or the features are), and every downstream result is
contaminated. This test runs against *every registered estimator* automatically, so
a new method cannot be added without passing it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from msl.engine.synthetic import make_regime_series
from msl.estimators.base import PROB_COLUMNS, get_estimator, list_estimators
from msl.features.core import build_features


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    px, _ = make_regime_series(n=1200, seed=3)
    return build_features(px)


@pytest.mark.parametrize("name", list_estimators())
def test_filter_is_causal(features: pd.DataFrame, name: str) -> None:
    """Estimates on a prefix must equal estimates on the full series, over the overlap."""
    est = get_estimator(name)
    cut = 900

    full = est.filter(features)
    prefix = est.filter(features.iloc[:cut])

    a = full.iloc[:cut][PROB_COLUMNS].astype(float).to_numpy()
    b = prefix[PROB_COLUMNS].astype(float).to_numpy()

    # compare only where both are defined (warm-up produces NaN in both)
    both = ~(np.isnan(a).any(axis=1) | np.isnan(b).any(axis=1))
    assert both.sum() > 100, f"{name}: too few comparable rows ({both.sum()})"
    np.testing.assert_allclose(
        a[both], b[both], atol=1e-8,
        err_msg=f"{name}: estimate at time t changed when future rows were appended (look-ahead)",
    )


def test_features_are_causal(features: pd.DataFrame) -> None:
    """The shared feature set must itself be free of look-ahead."""
    px, _ = make_regime_series(n=1200, seed=3)
    cut = 900
    full = build_features(px)
    prefix = build_features(px.iloc[:cut])

    cols = [c for c in full.columns if c != "close"]
    a = full.iloc[:cut][cols].to_numpy()
    b = prefix[cols].to_numpy()
    both = ~(np.isnan(a).any(axis=1) | np.isnan(b).any(axis=1))
    np.testing.assert_allclose(
        a[both], b[both], atol=1e-10,
        err_msg="a feature column peeks at future data",
    )
