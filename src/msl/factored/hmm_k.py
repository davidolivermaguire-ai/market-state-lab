"""A K-state diagonal-covariance Gaussian HMM, filtered at inference.

Generalises the fixed 3-state univariate model in `msl.estimators.hmm` so the RQ2
comparison can run a 9-state bivariate flat model against two 3-state univariate axes.
Kept separate rather than refactoring the existing estimator, because that one is load
bearing for the published RQ1 result and its behaviour should not move.

Causality: `fit` sees only the training window; `filter` runs the **forward** recursion
only. No smoothing, so the posterior at time t never uses data after t.
"""
from __future__ import annotations

import numpy as np

LOG2PI = float(np.log(2.0 * np.pi))
VAR_FLOOR = 1e-10


def _log_emissions(Y: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """log N(y_t | mu_k, diag(var_k)) for every t, k.  Y (n,d), mu/var (K,d)."""
    d = Y.shape[1]
    v = np.maximum(var, VAR_FLOOR)                       # (K,d)
    diff = Y[:, None, :] - mu[None, :, :]                # (n,K,d)
    quad = np.sum(diff * diff / v[None, :, :], axis=2)   # (n,K)
    norm = np.sum(np.log(v), axis=1) + d * LOG2PI        # (K,)
    return -0.5 * (quad + norm[None, :])


def _forward(logB: np.ndarray, A: np.ndarray, pi: np.ndarray):
    """Scaled forward pass. Returns filtered posteriors, scale factors, log-likelihood."""
    n, K = logB.shape
    B = np.exp(logB - logB.max(axis=1, keepdims=True))    # stabilise, scale cancels
    shift = logB.max(axis=1)
    alpha = np.empty((n, K))
    c = np.empty(n)

    a = pi * B[0]
    c[0] = a.sum() or 1e-300
    alpha[0] = a / c[0]
    for t in range(1, n):
        a = (alpha[t - 1] @ A) * B[t]
        c[t] = a.sum() or 1e-300
        alpha[t] = a / c[t]
    return alpha, c, float(np.sum(np.log(c) + shift))


def _backward(logB: np.ndarray, A: np.ndarray, c: np.ndarray) -> np.ndarray:
    n, K = logB.shape
    B = np.exp(logB - logB.max(axis=1, keepdims=True))
    beta = np.ones((n, K))
    for t in range(n - 2, -1, -1):
        beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
    return beta


class GaussHMM:
    """K-state diagonal Gaussian HMM fitted by Baum-Welch with multiple restarts.

    Restarts matter: EM finds a *local* optimum, and on regime data a single
    equal-variance initialisation reliably lands in a poor one. This is the same lesson
    the 3-state estimator learned the hard way.
    """

    def __init__(self, n_states: int, n_iter: int = 120, tol: float = 1e-6,
                 seed: int = 0, n_init: int = 5, min_occupancy: float = 0.02):
        self.K = int(n_states)
        self.n_iter, self.tol, self.seed, self.n_init = n_iter, tol, seed, n_init
        self.min_occupancy = min_occupancy
        self.degenerate_ = False
        self.mu_ = self.var_ = self.A_ = self.pi_ = None

    # ------------------------------------------------------------------ fit
    def fit(self, Y: np.ndarray) -> "GaussHMM":
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        if Y.shape[0] < Y.shape[1]:
            Y = Y.T
        Y = Y[np.isfinite(Y).all(axis=1)]
        if len(Y) < 50 * self.K:
            return self._set_defaults(Y)

        rng = np.random.default_rng(self.seed)
        # Warm start: consecutive walk-forward windows overlap heavily, so after the first
        # fit the previous solution is an excellent starting point and the full restart
        # search is wasted work. This is what makes the 9-state flat model tractable.
        warm = self.mu_ is not None
        n_init = 2 if warm else max(1, self.n_init)

        best = fallback = None
        for attempt in range(n_init):
            out = self._em_once(Y, rng, attempt, warm=(warm and attempt == 0))
            if out is None or not np.isfinite(out[-1]):
                continue
            if fallback is None or out[-1] > fallback[-1]:
                fallback = out
            occ = self._occupancy(out[2])
            if occ.min() >= self.min_occupancy and (best is None or out[-1] > best[-1]):
                best = out
        chosen, self.degenerate_ = (best, False) if best is not None else (fallback, True)
        if chosen is None:
            return self._set_defaults(Y)
        self.mu_, self.var_, self.A_, self.pi_, _ = chosen
        self._order_states()
        return self

    def _set_defaults(self, Y: np.ndarray) -> "GaussHMM":
        d = Y.shape[1] if Y.ndim == 2 and Y.size else 1
        mu = np.zeros((self.K, d)); var = np.ones((self.K, d))
        if Y.size:
            mu = np.linspace(Y.min(0), Y.max(0), self.K)
            var = np.tile(np.maximum(Y.var(0), VAR_FLOOR), (self.K, 1))
        self.mu_, self.var_ = mu, var
        self.A_ = np.full((self.K, self.K), 0.02 / max(self.K - 1, 1)) + np.eye(self.K) * 0.98
        self.A_ /= self.A_.sum(1, keepdims=True)
        self.pi_ = np.full(self.K, 1.0 / self.K)
        return self

    def _em_once(self, Y: np.ndarray, rng: np.random.Generator, attempt: int,
                 warm: bool = False):
        n, d = Y.shape
        if warm and self.mu_ is not None and self.mu_.shape == (self.K, d):
            mu, var = self.mu_.copy(), self.var_.copy()
            A, pi = self.A_.copy(), self.pi_.copy()
        else:
            q = np.linspace(0.1, 0.9, self.K)
            mu = np.quantile(Y, q, axis=0)
            if attempt:                               # jitter every restart but the first
                mu = mu + rng.normal(0, Y.std(0) * 0.5, size=mu.shape)
            var = np.tile(np.maximum(Y.var(0), VAR_FLOOR), (self.K, 1))
            var = var * rng.uniform(0.4, 1.6, size=var.shape) if attempt else var
            A = np.full((self.K, self.K), 0.05 / max(self.K - 1, 1)) + np.eye(self.K) * 0.95
            A /= A.sum(1, keepdims=True)
            pi = np.full(self.K, 1.0 / self.K)

        prev = -np.inf
        for _ in range(self.n_iter):
            logB = _log_emissions(Y, mu, var)
            alpha, c, ll = _forward(logB, A, pi)
            if not np.isfinite(ll):
                return None
            beta = _backward(logB, A, c)
            g = alpha * beta
            g /= np.maximum(g.sum(1, keepdims=True), 1e-300)

            B = np.exp(logB - logB.max(axis=1, keepdims=True))
            nxt = B[1:] * beta[1:] / np.maximum(c[1:, None], 1e-300)
            A = A * (alpha[:-1].T @ nxt)
            A /= np.maximum(A.sum(1, keepdims=True), 1e-300)

            w = np.maximum(g.sum(0), 1e-300)
            mu = (g.T @ Y) / w[:, None]
            dev = Y[:, None, :] - mu[None, :, :]
            var = np.einsum("nk,nkd->kd", g, dev * dev) / w[:, None]
            var = np.maximum(var, VAR_FLOOR)
            pi = g[0] / g[0].sum()

            if ll - prev < self.tol * max(1.0, abs(prev)):
                break
            prev = ll
        return mu, var, A, pi, ll

    @staticmethod
    def _occupancy(A: np.ndarray) -> np.ndarray:
        vals, vecs = np.linalg.eig(A.T)
        v = np.real(vecs[:, np.argmin(np.abs(vals - 1.0))])
        v = np.abs(v)
        return v / max(v.sum(), 1e-300)

    def _order_states(self) -> None:
        """Sort by the first observation dimension so labels are comparable across fits."""
        order = np.argsort(self.mu_[:, 0])
        self.mu_, self.var_ = self.mu_[order], self.var_[order]
        self.A_ = self.A_[np.ix_(order, order)]
        self.pi_ = self.pi_[order]

    # --------------------------------------------------------------- filter
    def filter(self, Y: np.ndarray) -> np.ndarray:
        """Forward-only posteriors. Row t uses observations up to and including t."""
        Y = np.atleast_2d(np.asarray(Y, dtype=float))
        if Y.shape[0] < Y.shape[1]:
            Y = Y.T
        if self.mu_ is None:
            self._set_defaults(Y)
        out = np.full((len(Y), self.K), np.nan)
        ok = np.isfinite(Y).all(axis=1)
        if ok.sum() == 0:
            return out
        alpha, _, _ = _forward(_log_emissions(Y[ok], self.mu_, self.var_), self.A_, self.pi_)
        out[ok] = alpha
        return out

    @property
    def n_params(self) -> int:
        """Free parameters: transitions (rows sum to 1) + means + variances."""
        K, d = self.K, (1 if self.mu_ is None else self.mu_.shape[1])
        return K * (K - 1) + K * d + K * d
