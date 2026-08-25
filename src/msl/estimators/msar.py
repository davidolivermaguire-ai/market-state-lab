"""Markov-switching regression — a regression whose coefficients switch with the regime.

Where the HMM lets only the *mean and variance* of returns switch, this lets the
**relationship** switch: each regime has its own intercept, its own autoregressive
coefficient on the previous return, and its own variance. Momentum in one regime and
mean-reversion in another is a hypothesis a switching mean cannot express.

Built on `statsmodels.tsa.regime_switching.MarkovRegression`, which also serves as a
reference implementation check against the hand-rolled EM in `hmm.py`.

Two look-ahead traps, both avoided
----------------------------------
1. `results.smoothed_marginal_probabilities` uses the **whole sample** to describe
   time t. It looks far better and is invalid here. Only
   `filtered_marginal_probabilities` — the forward pass — is used.
2. Subtler: even *filtered* probabilities are contaminated if the **parameters** were
   estimated on a sample that includes the future. So `fit()` estimates parameters on
   the training slice only, and `filter()` rebuilds the model on the new data and
   calls `.filter(stored_params)` — applying past-estimated parameters forward,
   never re-estimating.

Regimes are relabelled by fitted intercept (lowest = "down") by permuting the output
columns rather than the parameter vector, which avoids error-prone surgery on
statsmodels' packed transition parameters.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from msl.estimators.base import OUTPUT_COLUMNS, PROB_COLUMNS, STATES, StateEstimator, register

K = len(STATES)
_DEFAULT_TRAIN = 500        # rows used if filter() is called before fit() (kept fixed: see below)


def _build(y: pd.Series, x: pd.Series | None):
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    kw = dict(k_regimes=K, trend="c", switching_variance=True)
    if x is not None:
        kw.update(exog=x, switching_exog=True)
    return MarkovRegression(y, **kw)


def _regime_means(model, params: np.ndarray) -> np.ndarray:
    """Fitted intercept per regime, read off statsmodels' parameter names."""
    means = np.zeros(K)
    for i, nm in enumerate(model.param_names):
        if nm.startswith("const["):
            k = int(nm.split("[")[1].rstrip("]"))
            if 0 <= k < K:
                means[k] = params[i]
    return means


@register("ms_regression")
class MarkovSwitchingRegression(StateEstimator):
    """3-regime Markov-switching AR(1) on daily returns; filtered probabilities only."""

    requires_fit = True

    def __init__(self, order: int = 1, em_iter: int = 20, max_iter: int = 60, seed: int = 0):
        super().__init__(order=order, em_iter=em_iter, max_iter=max_iter, seed=seed)
        self.order, self.em_iter, self.max_iter, self.seed = order, em_iter, max_iter, seed
        self.params_: np.ndarray | None = None
        self.col_order_: np.ndarray = np.arange(K)

    # ------------------------------------------------------------- helpers
    def _prep(self, features: pd.DataFrame):
        r = features["ret"].astype(float)
        r = r[np.isfinite(r)]
        if self.order >= 1:
            x = r.shift(1).iloc[1:]
            y = r.iloc[1:]
            return y, x
        return r, None

    # ---------------------------------------------------------------- fit
    def fit(self, features: pd.DataFrame) -> "MarkovSwitchingRegression":
        y, x = self._prep(features)
        if len(y) < 200:
            return self
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mod = _build(y, x)
                res = mod.fit(em_iter=self.em_iter, maxiter=self.max_iter, disp=False)
            self.params_ = np.asarray(res.params, dtype=float)
            self.col_order_ = np.argsort(_regime_means(mod, self.params_))
        except Exception:
            self.params_ = None          # a failed fit must not kill a sweep
        return self

    # ------------------------------------------------------------- filter
    def filter(self, features: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(np.nan, index=features.index, columns=OUTPUT_COLUMNS, dtype=object)
        y, x = self._prep(features)
        if len(y) < 50:
            return out

        # If filter() is called before fit(), self-fit on a FIXED-length prefix. Fixed,
        # not proportional, so a prefix and the full series produce identical parameters —
        # which keeps the estimate causal and the look-ahead guard meaningful.
        if self.params_ is None:
            if len(y) < _DEFAULT_TRAIN:
                return out
            self.fit(features.iloc[: _DEFAULT_TRAIN + self.order])
            if self.params_ is None:
                return out

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mod = _build(y, x)
                res = mod.filter(self.params_)          # apply past params forward; no re-estimation
                fp = res.filtered_marginal_probabilities
        except Exception:
            return out

        probs = np.asarray(fp, dtype=float)
        if probs.ndim != 2 or probs.shape[1] != K:
            return out
        probs = probs[:, self.col_order_]               # relabel: down, range, up
        idx = fp.index if isinstance(fp, pd.DataFrame) else y.index[-len(probs):]

        out.loc[idx, PROB_COLUMNS] = probs
        out.loc[idx, "map_state"] = [STATES[i] for i in probs.argmax(axis=1)]
        out.loc[idx, "score"] = probs[:, STATES.index("up")] - probs[:, STATES.index("down")]
        return out

    def params_summary_(self) -> dict:
        return {"fitted": self.params_ is not None, "regime_order": self.col_order_.tolist()}
