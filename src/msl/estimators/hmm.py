"""Gaussian hidden Markov model — latent regimes with an explicit switching process.

Unlike the local-linear-trend filter, which assumes the drift *wanders* smoothly, an
HMM assumes it *switches* between a small number of discrete regimes, each with its
own mean and volatility, governed by a transition matrix. That is much closer to how
market regimes are usually described — and to the process the synthetic markets are
generated from — so it is the natural competitor.

Why this is hand-rolled rather than `hmmlearn`
----------------------------------------------
`hmmlearn`'s inference APIs are look-ahead for our purposes: `predict` runs Viterbi
over the whole sequence, and `predict_proba` returns forward-**backward** (smoothed)
posteriors. Both use future observations to describe time t. Using either would post
a beautiful, invalid result and would fail the look-ahead guard.

Inference here is the **forward recursion only**:

    alpha_t(k) ∝ P(state_t = k | y_1 … y_t)

which is the filtered posterior — the honest real-time estimate. Training (Baum-Welch)
does use forward-backward, but only over the *training slice*, which is past data by
construction.

State labels are made semantic after fitting by sorting the fitted means, so state 0
is the lowest-drift regime ("down") and state 2 the highest ("up"). The recovery
scorer's permutation matching then acts as a cross-check rather than a crutch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from msl.estimators.base import OUTPUT_COLUMNS, PROB_COLUMNS, STATES, StateEstimator, register

K = len(STATES)
_VAR_FLOOR = 1e-12
_EPS = 1e-300


def _occupancy(A: np.ndarray) -> np.ndarray:
    """Stationary distribution of the transition matrix — long-run time in each state."""
    try:
        w, v = np.linalg.eig(A.T)
        k = int(np.argmin(np.abs(w - 1.0)))
        p = np.real(v[:, k])
        p = np.abs(p)
        s = p.sum()
        return p / s if s > 0 else np.full(K, 1.0 / K)
    except Exception:
        return np.full(K, 1.0 / K)


def _log_emissions(y: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """log N(y_t | mu_k, var_k) for every t, k."""
    d = y[:, None] - mu[None, :]
    return -0.5 * (np.log(2.0 * np.pi * var)[None, :] + d * d / var[None, :])


def _forward(logB: np.ndarray, A: np.ndarray, pi: np.ndarray):
    """Scaled forward recursion. Returns (filtered posteriors, scale factors, loglik).

    This is the causal part: alpha_t uses observations up to t only.
    """
    T = logB.shape[0]
    alpha = np.empty((T, K))
    c = np.empty(T)
    loglik = 0.0

    m = logB.max(axis=1)
    B = np.exp(logB - m[:, None])          # per-row scaling for numerical stability

    a = pi * B[0]
    c[0] = a.sum() + _EPS
    alpha[0] = a / c[0]
    loglik += np.log(c[0]) + m[0]

    for t in range(1, T):
        a = (alpha[t - 1] @ A) * B[t]
        c[t] = a.sum() + _EPS
        alpha[t] = a / c[t]
        loglik += np.log(c[t]) + m[t]

    return alpha, c, float(loglik)


def _backward(logB: np.ndarray, A: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Scaled backward pass — training only, never used for the reported estimate."""
    T = logB.shape[0]
    m = logB.max(axis=1)
    B = np.exp(logB - m[:, None])
    beta = np.ones((T, K))
    for t in range(T - 2, -1, -1):
        beta[t] = A @ (B[t + 1] * beta[t + 1]) / c[t + 1]
    return beta


@register("hmm_gaussian")
class GaussianHMM(StateEstimator):
    """3-state Gaussian HMM on daily returns, filtered (forward-only) at inference."""

    requires_fit = True

    def __init__(self, n_iter: int = 120, tol: float = 1e-6, seed: int = 0, n_init: int = 6):
        """
        n_init : int
            EM restarts. Baum-Welch converges to a *local* optimum, and a single
            initialisation with equal starting variances reliably lands in a poor one
            on regime data whose states differ mostly in volatility. Restarting from
            varied means and variances and keeping the highest-likelihood solution is
            what brings this implementation into agreement with statsmodels'
            MarkovRegression on the same model class — the cross-check that validates
            the hand-rolled EM.
        """
        super().__init__(n_iter=n_iter, tol=tol, seed=seed, n_init=n_init)
        self.n_iter, self.tol, self.seed, self.n_init = n_iter, tol, seed, n_init
        self.min_occupancy = 0.08
        self.degenerate_ = False
        # defaults so filter() works before fit(): persistent regimes, ordered means
        self.mu_ = np.array([-1.0e-3, 0.0, 1.0e-3])
        self.var_ = np.array([4.0e-4, 6.4e-5, 1.0e-4])
        self.A_ = np.full((K, K), 0.01) + np.eye(K) * 0.97
        self.pi_ = np.full(K, 1.0 / K)

    # ---------------------------------------------------------------- fit
    def fit(self, features: pd.DataFrame) -> "GaussianHMM":
        y = features["ret"].to_numpy(dtype=float)
        y = y[np.isfinite(y)]
        if len(y) < 200:
            return self

        rng = np.random.default_rng(self.seed)
        best = None      # best solution with all states non-degenerate
        fallback = None  # best overall, used only if every solution is degenerate
        for attempt in range(max(1, self.n_init)):
            mu, var, A, pi, ll = self._em_once(y, rng, attempt)
            if not np.isfinite(ll):
                continue
            if fallback is None or ll > fallback[-1]:
                fallback = (mu, var, A, pi, ll)
            # Reject degenerate fits: a state that occupies almost no time can capture a
            # short quiet episode with an extreme mean, which scrambles mean-ordered
            # labels even though the likelihood is higher. Interpretability of the state
            # labels is a requirement here, not a nicety - the state report says "up".
            if _occupancy(A).min() >= self.min_occupancy:
                if best is None or ll > best[-1]:
                    best = (mu, var, A, pi, ll)
        chosen = best if best is not None else fallback
        if chosen is None:
            return self
        mu, var, A, pi, _ = chosen
        self.degenerate_ = best is None

        # make labels semantic: sort states by fitted mean -> down, range, up
        order = np.argsort(mu)
        self.mu_, self.var_, self.pi_ = mu[order], var[order], pi[order]
        self.A_ = A[np.ix_(order, order)]
        self.A_ /= np.maximum(self.A_.sum(axis=1, keepdims=True), _EPS)
        return self

    def _em_once(self, y: np.ndarray, rng: np.random.Generator, attempt: int):
        """One Baum-Welch run from a randomised start. Returns (mu, var, A, pi, loglik)."""
        base_var = max(float(y.var()), _VAR_FLOOR)
        if attempt == 0:
            # data-driven start: quantile means, variances spread low/mid/high
            mu = np.quantile(y, [0.15, 0.5, 0.85])
            var = base_var * np.array([2.0, 0.5, 1.0])
        else:
            mu = np.quantile(y, np.sort(rng.uniform(0.05, 0.95, K)))
            var = base_var * np.exp(rng.normal(0.0, 0.7, K))
        var = np.maximum(var, _VAR_FLOOR)
        diag = rng.uniform(0.90, 0.985)
        A = np.full((K, K), (1.0 - diag) / (K - 1)) + np.eye(K) * (diag - (1.0 - diag) / (K - 1))
        A /= A.sum(axis=1, keepdims=True)
        pi = np.full(K, 1.0 / K)

        prev, ll = -np.inf, -np.inf
        for _ in range(self.n_iter):
            logB = _log_emissions(y, mu, var)
            alpha, c, ll = _forward(logB, A, pi)
            if not np.isfinite(ll):
                break
            beta = _backward(logB, A, c)

            gamma = alpha * beta
            gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), _EPS)

            # xi summed over t, for the transition update
            m = logB.max(axis=1)
            B = np.exp(logB - m[:, None])
            xi = np.zeros((K, K))
            for t in range(len(y) - 1):
                x = (alpha[t][:, None] * A) * (B[t + 1] * beta[t + 1])[None, :] / c[t + 1]
                xi += x
            A = xi / np.maximum(xi.sum(axis=1, keepdims=True), _EPS)

            w = np.maximum(gamma.sum(axis=0), _EPS)
            mu = (gamma * y[:, None]).sum(axis=0) / w
            var = np.maximum(((y[:, None] - mu[None, :]) ** 2 * gamma).sum(axis=0) / w, _VAR_FLOOR)
            pi = gamma[0] / max(gamma[0].sum(), _EPS)

            if abs(ll - prev) < self.tol * max(1.0, abs(prev)):
                break
            prev = ll

        return mu, var, A, pi, float(ll)

    # ------------------------------------------------------------- filter
    def filter(self, features: pd.DataFrame) -> pd.DataFrame:
        r = features["ret"].astype(float)
        y = r.to_numpy()
        ok = np.isfinite(y)

        out = pd.DataFrame(np.nan, index=features.index, columns=OUTPUT_COLUMNS, dtype=object)
        if ok.sum() < 5:
            return out

        logB = _log_emissions(y[ok], self.mu_, self.var_)
        alpha, _, _ = _forward(logB, self.A_, self.pi_)      # filtered: no future data

        probs = pd.DataFrame(alpha, index=features.index[ok], columns=PROB_COLUMNS)
        out.loc[probs.index, PROB_COLUMNS] = probs.to_numpy()
        out.loc[probs.index, "map_state"] = [STATES[i] for i in alpha.argmax(axis=1)]
        # continuous trend score: how much more probable "up" is than "down"
        out.loc[probs.index, "score"] = alpha[:, STATES.index("up")] - alpha[:, STATES.index("down")]
        return out

    def params_(self) -> dict:
        return {"mu": self.mu_, "var": self.var_, "diag_A": np.diag(self.A_)}
