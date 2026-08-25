"""Change-point detectors: CUSUM and Bayesian online change-point detection.

These answer a different question from the state estimators. An HMM asks *which
regime are we in*; CUSUM and BOCPD ask *has the distribution just changed*. Both are
online by construction — CUSUM is a recursive decision rule, BOCPD a recursive
posterior over run length — so both are causal without any special care, which is
exactly why they belong in a real-time system.

To fit the common schema they must still emit a state, and the honest way to do that
is to combine detection with direction: a detector says *a shift happened*, and the
drift since that shift says *which way*. That is how a change-point detector actually
feeds a state estimate in the proposal's architecture.

They are marked `kind = "changepoint"` because the metrics that matter for them are
timeliness — detection delay and false alarms per year — not classification accuracy.
CUSUM in particular is a decision rule, not a probability model: it emits a decision
with a confidence, so its Brier score is mediocre *by construction*. That contrast
with the probabilistic models is a finding, not a defect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from msl.estimators.base import (
    OUTPUT_COLUMNS,
    PROB_COLUMNS,
    STATES,
    StateEstimator,
    register,
    softmax_states,
)

K = len(STATES)
_EPS = 1e-12


def _standardised_returns(features: pd.DataFrame) -> pd.Series:
    """Daily return in units of trailing volatility — causal by construction."""
    r = features["ret"].astype(float)
    sd = (features["rv20"].astype(float) / np.sqrt(252.0)).replace(0.0, np.nan)
    return (r / sd).replace([np.inf, -np.inf], np.nan)


@register("cusum")
class CusumDetector(StateEstimator):
    """Two-sided CUSUM on standardised returns.

    Accumulates evidence of a mean shift in each direction and declares a regime when
    the running sum crosses the threshold `h`, then resets. The declared state is held
    until the next crossing, which is what makes CUSUM naturally persistent.

    `k` (the slack, in standard deviations) and `h` (the threshold) set the classic
    average-run-length trade-off: lower `h` detects faster and cries wolf more often.
    Defaults are standard textbook values, deliberately *not* tuned on the recovery
    metric — tuning them there would be selecting on the evaluation.
    """

    requires_fit = False
    kind = "changepoint"

    def __init__(self, k: float = 0.5, h: float = 5.0):
        super().__init__(k=k, h=h)
        self.k, self.h = k, h

    def filter(self, features: pd.DataFrame) -> pd.DataFrame:
        z = _standardised_returns(features)
        out = pd.DataFrame(np.nan, index=features.index, columns=OUTPUT_COLUMNS, dtype=object)

        s_pos = s_neg = 0.0
        declared = 1                      # start in "range" — no evidence yet
        rows: list[tuple] = []
        for ts, zi in z.items():
            if not np.isfinite(zi):
                rows.append((ts, None, 0.0))
                continue
            s_pos = max(0.0, s_pos + zi - self.k)
            s_neg = max(0.0, s_neg - zi - self.k)
            if s_pos > self.h:
                declared, s_pos, s_neg = 2, 0.0, 0.0        # shift up
            elif s_neg > self.h:
                declared, s_pos, s_neg = 0, 0.0, 0.0        # shift down
            evidence = (s_pos if declared == 2 else s_neg if declared == 0 else max(s_pos, s_neg))
            rows.append((ts, declared, min(1.0, evidence / self.h)))

        idx = [r[0] for r in rows if r[1] is not None]
        if not idx:
            return out
        dec = np.array([r[1] for r in rows if r[1] is not None], dtype=int)
        ev = np.array([r[2] for r in rows if r[1] is not None], dtype=float)

        # A decision rule, not a probability model: confidence in the declared state
        # rises with accumulated evidence, and the rest is spread evenly. Honest, and
        # deliberately not overconfident.
        p = np.full((len(dec), K), 0.0)
        conf = 0.50 + 0.35 * ev
        p[np.arange(len(dec)), dec] = conf
        rest = (1.0 - conf) / (K - 1)
        for j in range(K):
            mask = dec != j
            p[mask, j] = rest[mask]

        out.loc[idx, PROB_COLUMNS] = p
        out.loc[idx, "map_state"] = [STATES[i] for i in dec]
        out.loc[idx, "score"] = p[:, STATES.index("up")] - p[:, STATES.index("down")]
        return out


@register("bocpd")
class BayesianOnlineChangePoint(StateEstimator):
    """Bayesian online change-point detection (Adams & MacKay, 2007).

    Maintains a posterior over the **run length** — how long since the last change —
    updated recursively with a Normal-Inverse-Gamma conjugate model for the unknown
    mean and variance of returns. Unlike CUSUM it produces a genuine probability, and
    unlike an HMM it does not need to commit to a fixed number of regimes.

    The trend state is read off the most probable run length: the posterior mean of
    that segment, in units of its own standard error, gives a t-like score.

    Run length is capped at `r_max` purely for cost — the exact recursion is O(T²).
    """

    requires_fit = False
    kind = "changepoint"

    def __init__(self, expected_run: float = 120.0, r_max: int = 300, sharpness: float = 0.7):
        super().__init__(expected_run=expected_run, r_max=r_max, sharpness=sharpness)
        self.expected_run, self.r_max, self.sharpness = expected_run, r_max, sharpness

    def filter(self, features: pd.DataFrame) -> pd.DataFrame:
        r = features["ret"].astype(float)
        valid = r[np.isfinite(r)]
        out = pd.DataFrame(np.nan, index=features.index, columns=OUTPUT_COLUMNS, dtype=object)
        if len(valid) < 30:
            return out

        y = valid.to_numpy()
        hazard = 1.0 / float(self.expected_run)
        rmax = int(self.r_max)

        # Normal-Inverse-Gamma prior, scaled to the data so it is not accidentally sharp
        v0 = max(float(np.nanvar(y[: min(len(y), 250)])), 1e-10)
        mu0, kappa0, alpha0, beta0 = 0.0, 1.0, 1.0, v0

        mu = np.array([mu0]); kappa = np.array([kappa0])
        alpha = np.array([alpha0]); beta = np.array([beta0])
        R = np.array([1.0])

        scores = np.empty(len(y))
        for t, yt in enumerate(y):
            # predictive: Student-t per run-length hypothesis
            scale = np.sqrt(np.maximum(beta * (kappa + 1.0) / (alpha * kappa), _EPS))
            pred = student_t.pdf(yt, df=2.0 * alpha, loc=mu, scale=scale)
            pred = np.maximum(pred, _EPS)

            growth = R * pred * (1.0 - hazard)
            change = float(np.sum(R * pred * hazard))
            R_new = np.concatenate(([change], growth))
            R_new /= max(R_new.sum(), _EPS)

            # conjugate updates, with a fresh segment prepended
            mu_new = np.concatenate(([mu0], (kappa * mu + yt) / (kappa + 1.0)))
            kappa_new = np.concatenate(([kappa0], kappa + 1.0))
            alpha_new = np.concatenate(([alpha0], alpha + 0.5))
            beta_new = np.concatenate(
                ([beta0], beta + (kappa * (yt - mu) ** 2) / (2.0 * (kappa + 1.0)))
            )

            if len(R_new) > rmax:            # truncate the tail; renormalise
                R_new = R_new[:rmax]; R_new /= max(R_new.sum(), _EPS)
                mu_new, kappa_new = mu_new[:rmax], kappa_new[:rmax]
                alpha_new, beta_new = alpha_new[:rmax], beta_new[:rmax]

            R, mu, kappa, alpha, beta = R_new, mu_new, kappa_new, alpha_new, beta_new

            # state from the most probable segment: its mean over its own standard error
            j = int(np.argmax(R))
            se = np.sqrt(max(beta[j] / (alpha[j] * kappa[j]), _EPS))
            scores[t] = np.tanh(self.sharpness * mu[j] / se)

        probs = softmax_states(pd.Series(scores, index=valid.index))
        out.loc[probs.index, OUTPUT_COLUMNS] = probs[OUTPUT_COLUMNS].to_numpy()
        return out
