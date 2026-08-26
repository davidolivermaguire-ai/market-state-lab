"""Local linear trend — a state-space model of drifting level and slope.

The model treats log price as a noisy observation of a hidden level that drifts at a
hidden, slowly-changing rate:

    level_t = level_{t-1} + slope_{t-1} + w_level
    slope_t = slope_{t-1}               + w_slope
    y_t     = level_t                   + v

The **slope** is the trend state, and the Kalman filter gives it to us with an
uncertainty attached — which is the reason to prefer it over a moving-average rule.
The trend score is the slope in units of its own standard error, so a small drift
estimated precisely and a large drift estimated noisily are treated differently. A
crossover rule cannot make that distinction.

Two properties matter for this harness:

* **Filtered, never smoothed.** The forward pass at time t uses only observations up
  to t. The RTS smoother would produce a visibly better-looking trend and is pure
  look-ahead; it is deliberately not used. The look-ahead guard enforces this.
* **It learns.** The three variances are estimated by maximum likelihood on the
  training slice, so this is the first estimator with `requires_fit = True` and the
  first to exercise the walk-forward refit path.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from msl.estimators.base import StateEstimator, register, softmax_states

_BURN = 20          # observations skipped in the likelihood while the diffuse prior washes out
_EPS = 1e-12


def _kf(y: np.ndarray, q_level: float, q_slope: float, r: float):
    """Causal forward pass. Returns (slope, slope_sd, loglik)."""
    n = len(y)
    x = np.array([y[0], 0.0])
    P = np.array([[1e6, 0.0], [0.0, 1e2]])          # diffuse prior
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.array([[q_level, 0.0], [0.0, q_slope]])

    slope = np.empty(n)
    slope_sd = np.empty(n)
    loglik = 0.0

    for t in range(n):
        # predict
        x = F @ x
        P = F @ P @ F.T + Q
        # update with the scalar observation y_t = level_t + v
        v = y[t] - x[0]
        S = P[0, 0] + r
        if S <= _EPS or not np.isfinite(S):
            return slope, slope_sd, -np.inf
        K = P[:, 0] / S
        x = x + K * v
        P = P - np.outer(K, P[0, :])
        P = 0.5 * (P + P.T)                          # keep it symmetric

        slope[t] = x[1]
        slope_sd[t] = np.sqrt(max(P[1, 1], _EPS))
        if t >= _BURN:
            loglik += -0.5 * (np.log(2.0 * np.pi * S) + v * v / S)

    return slope, slope_sd, loglik


@register("kalman_trend")
class KalmanLocalLinearTrend(StateEstimator):
    """Filtered slope of a local-linear-trend model, scored against its own error."""

    requires_fit = True
    # Maximum likelihood drives q_slope toward zero (see below), which makes the slope
    # an integral of the whole history rather than a fading average. Bounding the
    # run-up therefore changes the estimate materially — measured at ~0.8 MAP agreement
    # against full replay — so this filter opts out of the windowed shortcut.
    full_replay = True

    def __init__(self, sharpness: float = 0.5, max_iter: int = 60,
                 q_slope: float | str | None = None,
                 trend_halflife: int = 42, drift_fraction: float = 0.1):
        """
        Parameters
        ----------
        q_slope : float | "mle" | None
            ``None`` (default) derives the slope-drift variance from a **stated prior**
            on how fast trend evolves — see below. A float pins it. ``"mle"`` estimates
            it by maximum likelihood, which is the documented failure case.
        trend_halflife, drift_fraction : int, float
            The prior. Trend drift should be able to wander by `drift_fraction` of a
            daily return's magnitude over `trend_halflife` days, giving
            ``q_slope = (drift_fraction * sigma_r)^2 / trend_halflife``. Defaults say a
            trend can shift by a tenth of daily volatility over roughly two months.

        Why the prior replaced MLE as the default (a finding, not a bug)
        ----------------------------------------------------------------
        Maximum likelihood drives `q_slope` to the floor. On real Nasdaq data it lands
        on 1e-12 — the exact lower bound — which freezes the slope entirely. Because
        equity log-price has persistent positive drift, a frozen slope stays positive
        forever, and the filter reported "up" on **100% of days** for four of seven
        assets. That is a constant, not a state estimator.

        The cause is an objective mismatch: MLE maximises the one-step predictive
        likelihood of *price*, which is dominated by fitting observation noise. It is
        not the same objective as *identifying the state*, and nothing forces the two
        to agree.

        The prior is a modelling choice made from the training window and a stated
        belief about trend persistence — never from the evaluation metric, which would
        be selecting on the answer. `sigma_r` is measured on training data only.
        """
        super().__init__(sharpness=sharpness, max_iter=max_iter, q_slope=q_slope,
                         trend_halflife=trend_halflife, drift_fraction=drift_fraction)
        self.sharpness = sharpness
        self.max_iter = max_iter
        self.trend_halflife = trend_halflife
        self.drift_fraction = drift_fraction
        self.use_mle_slope = (q_slope == "mle")
        self.user_q_slope = None if isinstance(q_slope, str) else q_slope
        # sensible defaults so filter() works before fit() (and in the contract tests):
        # observation noise dominates, the slope drifts very slowly.
        self.q_level_ = 1e-5
        self.q_slope_ = self.user_q_slope if self.user_q_slope is not None else 1e-8
        self.r_ = 1e-4

    # ---------------------------------------------------------------- fit
    def fit(self, features: pd.DataFrame) -> "KalmanLocalLinearTrend":
        y = np.log(features["close"].to_numpy(dtype=float))
        y = y[np.isfinite(y)]
        if len(y) < 100:
            return self                                   # too short to identify: keep defaults

        # Resolve the slope variance for THIS training window, recomputed at every fit
        # so the prior tracks the volatility of the window actually being trained on.
        pinned = self.user_q_slope
        if pinned is None and not self.use_mle_slope:
            sigma_r = float(np.std(np.diff(y)))          # training-window volatility only
            pinned = max((self.drift_fraction * sigma_r) ** 2 / max(self.trend_halflife, 1), 1e-14)

        if pinned is not None:
            # slope variance is set (prior-derived or user-pinned); estimate the rest
            def nll(theta: np.ndarray) -> float:
                q_l, r = np.exp(theta)
                _, _, ll = _kf(y, q_l, pinned, r)
                return -ll if np.isfinite(ll) else 1e12

            start = np.log([self.q_level_, self.r_])
            bounds = [(np.log(1e-10), np.log(1e-1)), (np.log(1e-10), np.log(1e-1))]
            try:
                res = minimize(nll, start, method="L-BFGS-B", bounds=bounds,
                               options={"maxiter": self.max_iter})
                if np.isfinite(res.fun):
                    self.q_level_, self.r_ = np.exp(res.x)
            except Exception:
                pass
            self.q_slope_ = pinned
            return self

        def nll(theta: np.ndarray) -> float:
            q_l, q_s, r = np.exp(theta)
            _, _, ll = _kf(y, q_l, q_s, r)
            return -ll if np.isfinite(ll) else 1e12

        start = np.log([self.q_level_, self.q_slope_, self.r_])
        bounds = [(np.log(1e-10), np.log(1e-1)),
                  (np.log(1e-12), np.log(1e-3)),
                  (np.log(1e-10), np.log(1e-1))]
        try:
            res = minimize(nll, start, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": self.max_iter})
            if res.success or np.isfinite(res.fun):
                self.q_level_, self.q_slope_, self.r_ = np.exp(res.x)
        except Exception:
            pass                                          # keep defaults rather than fail a sweep
        return self

    # ------------------------------------------------------------- filter
    def filter(self, features: pd.DataFrame) -> pd.DataFrame:
        close = features["close"].astype(float)
        y = np.log(close.to_numpy())
        slope, slope_sd, _ = _kf(y, self.q_level_, self.q_slope_, self.r_)

        # slope in units of its own standard error: a t-like statistic
        z = slope / np.maximum(slope_sd, _EPS)
        score = pd.Series(np.tanh(self.sharpness * z), index=features.index)
        score.iloc[:_BURN] = np.nan                       # diffuse prior not yet washed out
        return softmax_states(score.fillna(0.0)).where(score.notna())

    def params_(self) -> dict:
        return {"q_level": self.q_level_, "q_slope": self.q_slope_, "r": self.r_}
