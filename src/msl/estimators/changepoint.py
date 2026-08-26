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
from scipy.special import gammaln

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


def _student_t_pdf(x: float, df: np.ndarray, loc: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Student-t density, vectorised over run-length hypotheses.

    Identical maths to `scipy.stats.t.pdf`, but computed directly: the recursion calls
    this once per observation (thousands of times per series), and scipy's per-call
    distribution machinery dominates the runtime at that frequency.
    """
    z = (x - loc) / scale
    log_p = (
        gammaln(0.5 * (df + 1.0))
        - gammaln(0.5 * df)
        - 0.5 * np.log(np.pi * df)
        - np.log(scale)
        - 0.5 * (df + 1.0) * np.log1p(z * z / df)
    )
    return np.exp(log_p)


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

    Design, not tuning
    ------------------
    The textbook constants (`k = 0.5`, `h = 5`) come from manufacturing process
    control, where the shift worth detecting is on the order of one standard
    deviation. Daily equity drift is nothing like that: measured on the Nasdaq, the
    mean standardised return is **+0.08**, so the trend signal is about **16% of that
    slack**. Evidence never accumulates, and the rule fires roughly four times in
    twelve years — reading "down" through a bull market because one rare volatility
    event tripped it and nothing since has moved it.

    So the parameters are *designed from the training window*, exactly as SPC
    prescribes:

    * `k` = half the shift worth detecting, where the shift scale is the dispersion of
      the slow-moving mean of standardised returns (how much drift actually varies).
    * `h` = calibrated to a target in-control average run length, by counting triggers
      on the training data.

    Both come from data the estimator is allowed to see, never from the evaluation
    metric — choosing them by recovery score would be selecting on the answer.
    """

    requires_fit = True
    kind = "changepoint"

    def __init__(self, k: float | None = None, h: float | None = None,
                 shift_window: int = 60, target_arl: float = 252.0):
        super().__init__(k=k, h=h, shift_window=shift_window, target_arl=target_arl)
        self.shift_window, self.target_arl = shift_window, target_arl
        self.fixed_k, self.fixed_h = k, h
        # textbook values as the pre-fit fallback (and the documented failure case)
        self.k_ = 0.5 if k is None else k
        self.h_ = 5.0 if h is None else h

    # ---------------------------------------------------------------- fit
    def fit(self, features: pd.DataFrame) -> "CusumDetector":
        z = _standardised_returns(features).dropna().to_numpy()
        if len(z) < 250:
            return self
        if self.fixed_k is None:
            slow = pd.Series(z).rolling(self.shift_window).mean().dropna()
            delta = 2.0 * float(slow.std())          # shift worth detecting, in sd units
            self.k_ = float(max(delta / 2.0, 1e-3))
        if self.fixed_h is None:
            self.h_ = self._calibrate_h(z, self.k_)
        return self

    def _run(self, z: np.ndarray, k: float, h: float):
        """The CUSUM recursion. Returns (declared state, evidence) per observation."""
        s_pos = s_neg = 0.0
        declared = 1
        dec = np.empty(len(z), dtype=int)
        ev = np.empty(len(z))
        for i, zi in enumerate(z):
            s_pos = max(0.0, s_pos + zi - k)
            s_neg = max(0.0, s_neg - zi - k)
            if s_pos > h:
                declared, s_pos, s_neg = 2, 0.0, 0.0        # shift up
            elif s_neg > h:
                declared, s_pos, s_neg = 0, 0.0, 0.0        # shift down
            dec[i] = declared
            ev[i] = min(1.0, (s_pos if declared == 2 else s_neg if declared == 0
                              else max(s_pos, s_neg)) / max(h, 1e-9))
        return dec, ev

    def _calibrate_h(self, z: np.ndarray, k: float) -> float:
        """Smallest threshold whose trigger rate on training data meets the target ARL."""
        target = max(1.0, len(z) / self.target_arl)
        lo, hi = 0.25, 25.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            dec, _ = self._run(z, k, mid)
            n = int((dec[1:] != dec[:-1]).sum())
            if n > target:
                lo = mid                     # too jumpy -> raise the threshold
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # ------------------------------------------------------------- filter
    def filter(self, features: pd.DataFrame) -> pd.DataFrame:
        z = _standardised_returns(features)
        out = pd.DataFrame(np.nan, index=features.index, columns=OUTPUT_COLUMNS, dtype=object)

        ok = np.isfinite(z.to_numpy())
        if ok.sum() < 5:
            return out
        dec_ok, ev_ok = self._run(z.to_numpy()[ok], self.k_, self.h_)

        rows: list[tuple] = []
        j = 0
        for ts, good in zip(z.index, ok):
            if good:
                rows.append((ts, int(dec_ok[j]), float(ev_ok[j])))
                j += 1
            else:
                rows.append((ts, None, 0.0))

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
            pred = _student_t_pdf(yt, 2.0 * alpha, mu, scale)
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
