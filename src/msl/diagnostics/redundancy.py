"""Are the specialists independent dimensions, or independent noise?

The proposal's architecture assumes a council of specialists covering *different*
dimensions of market state. If they all measure one thing, disagreement between them is
noise and the aggregation step has nothing to aggregate.

The obvious check is the effective number of independent signals, from the dispersion of
the correlation matrix's eigenvalues:

.. math::  N_{\\text{eff}} = \\frac{(\\sum_i \\lambda_i)^2}{\\sum_i \\lambda_i^2}

On the seven non-baseline estimators this returns 3.8-4.7 of 7 — apparently healthy. **It
is not sufficient, and reading it as sufficient is the trap this module exists to close.**

Effective-n cannot distinguish two situations that imply opposite build orders:

1. *Different dimensions.* Seven estimators genuinely measuring seven things decorrelate.
2. *One dimension, measured noisily.* Seven noisy estimates of the **same** quantity also
   decorrelate, because measurement error is independent even when the target is shared.
   Here a **higher** effective-n means **worse** estimators, and the correct response is
   to fix them rather than to add more.

Both produce a middling effective-n and low pairwise correlation. Counting dimensions
cannot tell them apart, because the difference is not in the geometry of the signals — it
is in whether the shared part or the specialist-specific part is the part that predicts.

So this module splits each signal into what the specialists **agree** on and where each
one **deviates**, and scores the two separately against the same targets and the same
purged walk-forward machinery as RQ1:

* consensus predicts, deviations do not  ->  one dimension plus noise; improve estimators
* deviations also predict                ->  real dimensions; add specialists
* neither predicts                       ->  the family is empty, which is the actual
                                             finding behind RQ1's null

**Causality.** The decomposition is the fragile part. A full-sample PCA, or a full-sample
standardisation, hands the consensus information about the future and inflates precisely
the quantity under test. Everything here uses trailing windows only, and
``tests/test_redundancy.py`` asserts it by recomputing on a truncated series.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from msl.metrics.decision import (BASE_FEATURES, TARGETS, _walk_forward_proba,
                                  build_targets, diebold_mariano)

Z_WINDOW = 252          # trailing window for standardisation
Z_MIN = 60              # minimum observations before a z-score is defined
MIN_SCORED = 300        # below this a walk-forward comparison is not worth running


# --------------------------------------------------------------------- dimension counting
def effective_n(signals: pd.DataFrame) -> float:
    """Effective number of independent signals, via eigenvalue dispersion.

    Returns 1.0 for perfectly correlated columns and M for orthogonal ones. Reported
    here as a *description*, never as evidence that the signals are informative — see
    the module docstring for why the two come apart.
    """
    x = signals.dropna()
    x = x.loc[:, x.std() > 1e-12]
    if x.shape[1] < 2 or len(x) < 3:
        return float(x.shape[1])
    C = np.corrcoef(x.to_numpy(dtype=float), rowvar=False)
    if not np.isfinite(C).all():
        return float("nan")
    ev = np.linalg.eigvalsh(C)
    ev = ev[ev > 1e-12]
    return float((ev.sum() ** 2) / (ev ** 2).sum())


def _causal_z(signals: pd.DataFrame, window: int = Z_WINDOW,
              min_periods: int = Z_MIN) -> pd.DataFrame:
    """Standardise each column on a trailing window only.

    A full-sample z-score leaks the future into the scale of every signal. That would
    not look like look-ahead in a plot, and it would raise the apparent predictive
    content of the consensus, which is the number this module reports.
    """
    mu = signals.rolling(window, min_periods=min_periods).mean()
    sd = signals.rolling(window, min_periods=min_periods).std(ddof=1)
    return (signals - mu) / sd.where(sd > 1e-12)


@dataclass
class Decomposition:
    """A signal panel split into agreement and deviation."""

    consensus: pd.Series            # what the specialists agree on
    idiosyncratic: pd.DataFrame     # where each one departs from that
    loadings: pd.Series             # corr of each specialist with the consensus
    effective_n: float
    notes: list[str] = field(default_factory=list)


def decompose(signals: pd.DataFrame, window: int = Z_WINDOW,
              min_periods: int = Z_MIN) -> Decomposition:
    """Split a specialist panel into a consensus and per-specialist deviations.

    The consensus is the cross-sectional mean of causally standardised signals — an
    equal-weight factor rather than a fitted first principal component. That is a
    deliberate choice: rolling PCA has to estimate M loadings from a short window, and
    the estimation error would be attributed to the consensus, which is the quantity
    under test. An equal-weight mean has no estimated parameters and therefore no
    estimation error to launder.

    Equal weighting assumes the specialists load on the common factor with the same
    sign. That assumption is checked, not asserted: `loadings` reports each one, and a
    negative loading is flagged rather than silently averaged away.
    """
    z = _causal_z(signals, window, min_periods)
    consensus = z.mean(axis=1, skipna=False).rename("consensus")
    idio = z.sub(consensus, axis=0)
    idio.columns = [f"idio_{c}" for c in z.columns]

    both = pd.concat([z, consensus], axis=1).dropna()
    loadings = (both.corr()["consensus"].drop("consensus")
                if len(both) > 10 else pd.Series(dtype=float))

    notes: list[str] = []
    neg = [s for s, v in loadings.items() if v < 0]
    if neg:
        notes.append(
            f"negative loading on the consensus: {', '.join(neg)} — an equal-weight mean "
            f"partially cancels these, so the consensus understates the shared component.")
    notes.append(
        "consensus and deviations are built from trailing-window z-scores only; a "
        "full-sample standardisation would leak the future into the consensus.")
    return Decomposition(consensus, idio, loadings, effective_n(signals), notes)


# ------------------------------------------------------------------------- the gate itself
def _score(df: pd.DataFrame, extra: list[str], horizon: int,
           n_splits: int) -> list[dict]:
    """Brier for [base + extra] against [base] alone, per target, purged walk-forward."""
    from sklearn.metrics import brier_score_loss

    X_base = df[BASE_FEATURES].to_numpy(dtype=float)
    X_aug = df[BASE_FEATURES + extra].to_numpy(dtype=float)
    rows = []
    for target in TARGETS:
        y = df[target].to_numpy(dtype=int)
        if len(np.unique(y)) < 2:
            continue
        p_b = _walk_forward_proba(X_base, y, n_splits, horizon)
        p_a = _walk_forward_proba(X_aug, y, n_splits, horizon)
        ok = ~(np.isnan(p_b) | np.isnan(p_a))
        if ok.sum() < 100:
            continue
        yb = y[ok]
        pb = np.clip(p_b[ok], 1e-6, 1 - 1e-6)
        pa = np.clip(p_a[ok], 1e-6, 1 - 1e-6)
        t, p = diebold_mariano((pa - yb) ** 2, (pb - yb) ** 2, lag=horizon)
        rows.append({
            "target": target, "n": int(ok.sum()),
            "brier_base": brier_score_loss(yb, pb),
            "brier_aug": brier_score_loss(yb, pa),
            "brier_delta": brier_score_loss(yb, pa) - brier_score_loss(yb, pb),
            "t_stat": t, "p_value": p,
        })
    return rows


def redundancy_gate(signals: pd.DataFrame, features: pd.DataFrame, horizon: int = 5,
                    n_splits: int = 5, window: int = Z_WINDOW) -> tuple[pd.DataFrame, Decomposition]:
    """Does the agreement predict, or do the deviations?

    Every variant is scored against the *same* volatility-only baseline used in RQ1, so
    the numbers are directly comparable to the published decision-value work. A negative
    `brier_delta` means the variant improved on volatility alone.

    Variants, chosen so the headline comparison is like-for-like:

    * ``consensus`` — one feature: what the specialists agree on.
    * ``idio_<name>`` — one feature each: where that specialist departs from consensus.
      One feature against one feature, so neither side wins on feature count.
    * ``idio_all`` — all deviations together. Reported, but it carries M features against
      the consensus's one, so it is not a fair head-to-head and is not the headline.
    """
    dec = decompose(signals, window)
    targets = build_targets(features, horizon)
    frame = pd.concat(
        [features[BASE_FEATURES], dec.consensus, dec.idiosyncratic, targets],
        axis=1).dropna()
    if len(frame) < MIN_SCORED:
        return pd.DataFrame(), dec

    idio_cols = list(dec.idiosyncratic.columns)
    variants: dict[str, list[str]] = {"consensus": ["consensus"]}
    for c in idio_cols:
        variants[c] = [c]
    variants["idio_all"] = idio_cols
    variants["consensus_plus_idio"] = ["consensus"] + idio_cols

    rows = []
    for name, cols in variants.items():
        for r in _score(frame, cols, horizon, n_splits):
            rows.append({"variant": name, "n_features": len(cols), **r})
    return pd.DataFrame(rows), dec


def verdict(gate: pd.DataFrame, alpha: float = 0.05) -> dict:
    """Turn the grid into the three-way answer the gate was built to give.

    The deflated bar is |t| > sqrt(2 ln N) over the whole grid, because every variant
    tried is another chance for noise to look like signal — the same correction applied
    to the published experiments.
    """
    if gate.empty:
        return {"verdict": "insufficient data"}
    n = len(gate)
    bar = float(np.sqrt(2.0 * np.log(max(n, 2))))

    def _wins(mask: pd.Series) -> int:
        sub = gate[mask]
        return int(((sub["t_stat"] < -bar) & (sub["brier_delta"] < 0)).sum())

    is_con = gate["variant"] == "consensus"
    is_idio = gate["variant"].str.startswith("idio_") & (gate["n_features"] == 1)
    con_wins, idio_wins = _wins(is_con), _wins(is_idio)

    if con_wins and not idio_wins:
        v = ("one dimension plus noise — the specialists' agreement carries the "
             "information and their disagreement does not. Improve the estimators; "
             "adding more of the same kind will not help.")
    elif idio_wins:
        v = ("genuinely multiple dimensions — specialist-specific variation predicts "
             "beyond the consensus. The council premise holds and more specialists are "
             "worth adding.")
    else:
        v = ("neither the agreement nor the disagreement beats volatility alone — the "
             "family is empty, and that is the finding behind RQ1's null.")
    return {
        "comparisons": n, "deflated_bar": bar,
        "consensus_wins": con_wins, "idiosyncratic_wins": idio_wins,
        "best_consensus_t": float(gate.loc[is_con, "t_stat"].min()) if is_con.any() else np.nan,
        "best_idio_t": float(gate.loc[is_idio, "t_stat"].min()) if is_idio.any() else np.nan,
        "verdict": v,
    }
