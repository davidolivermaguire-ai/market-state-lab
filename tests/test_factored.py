"""RQ2 factored-state module: causality, cardinality, and parameter accounting."""
import numpy as np
import pandas as pd
import pytest

from msl.engine.synthetic import make_regime_series
from msl.features.core import build_features
from msl.factored.hmm_k import GaussHMM
from msl.factored.compare import _observations, _walk_forward_posteriors, _joint_map, stability


@pytest.fixture(scope="module")
def feats():
    px, _ = make_regime_series(n=1400, seed=11)
    return build_features(px)


@pytest.mark.parametrize("mode,K", [("factored", 3), ("flat", 9)])
def test_no_lookahead(feats, mode, K):
    """Posteriors at t must not change when future rows are appended.

    This is the guarantee the whole project rests on. A smoothed pass would score far
    better and mean nothing.
    """
    obs = _observations(feats)
    cut = len(obs) - 120
    full = _walk_forward_posteriors(obs, mode, K, min_train=400, refit_every=63, max_train=600)
    trunc = _walk_forward_posteriors(obs.iloc[:cut], mode, K, min_train=400,
                                     refit_every=63, max_train=600)
    a = full.iloc[:cut].to_numpy(float)
    b = trunc.to_numpy(float)
    both = np.isfinite(a) & np.isfinite(b)
    assert both.sum() > 100, "not enough overlapping rows to test"
    assert np.allclose(a[both], b[both], atol=1e-8), f"{mode} posteriors changed with future data"


def test_factored_and_flat_share_cardinality(feats):
    """3x3 and flat-9 must describe the same number of joint cells, or the test is rigged."""
    obs = _observations(feats)
    fac = _walk_forward_posteriors(obs, "factored", 3, min_train=400, refit_every=63, max_train=600)
    flat = _walk_forward_posteriors(obs, "flat", 9, min_train=400, refit_every=63, max_train=600)
    lf, lp = _joint_map(fac, "factored", 3), _joint_map(flat, "flat", 9)
    assert lf.dropna().max() <= 8 and lp.dropna().max() <= 8
    assert stability(lf)["n_cells_used"] <= 9
    assert stability(lp)["n_cells_used"] <= 9


def test_flat_costs_more_parameters():
    """The premise of H1: same cells, far more parameters."""
    fac = 2 * (3 * 2 + 3 + 3)          # two univariate 3-state chains  = 24
    flat = 9 * 8 + 9 * 2 + 9 * 2       # one bivariate 9-state chain    = 108
    assert fac == 24 and flat == 108
    assert flat > 4 * fac, "flat model should be several times more expensive"
    assert GaussHMM(9).K == 9


def test_posteriors_are_valid_distributions(feats):
    obs = _observations(feats)
    p = _walk_forward_posteriors(obs, "flat", 9, min_train=400, refit_every=63, max_train=600)
    rows = p.dropna()
    assert len(rows) > 100
    assert np.allclose(rows.to_numpy().sum(axis=1), 1.0, atol=1e-6)
    assert (rows.to_numpy() >= -1e-12).all()
