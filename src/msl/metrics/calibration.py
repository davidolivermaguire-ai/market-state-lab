"""H2 re-test: does regime-conditioning sharpen the probability of a dangerous day?

The original experiment compared a filtered HMM regime against an *expanding event rate
within a trailing-volatility tercile* and reported Brier 0.155 → 0.153 with no significance
test. Two things are wrong with that as evidence:

1. **The benchmark is cruder than the treatment.** A three-bucket lookup is a much weaker
   use of volatility than the regime model's filtered state. Beating it demonstrates that
   an HMM beats terciles, not that *state* beats *volatility*.
2. **0.002 is inside the noise.** For scale, the largest calibration gain anywhere in the
   RQ1 panel was ten times bigger and still failed to replicate across assets.

So this module runs the same claim against **both** benchmarks: the original tercile rule,
and a same-information logistic on the volatility features the regime model also sees.
Everything is purged walk-forward and scored with Diebold-Mariano, as in RQ1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from msl.metrics.decision import (BASE_FEATURES, diebold_mariano, _walk_forward_proba)

BIG_DOWN = -0.01          # the original experiment's primary target: r[t+1] < -1%


def big_move_target(features: pd.DataFrame, threshold: float = BIG_DOWN) -> pd.Series:
    """Tomorrow is a big adverse day. Forward-looking — a target, never an input."""
    fwd = features["ret"].astype(float).shift(-1)
    y = (fwd < threshold).astype(float)
    y[fwd.isna()] = np.nan
    return y.rename("big_down")


def _tercile_rate(features: pd.DataFrame, y: pd.Series, min_train: int = 750) -> np.ndarray:
    """The original benchmark: expanding event rate inside today's trailing-vol tercile.

    Causal by construction — the tercile cuts and the rates both use only past rows.
    """
    rv = features["rv20"].astype(float).to_numpy()
    yy = y.to_numpy(dtype=float)
    out = np.full(len(rv), np.nan)
    for t in range(min_train, len(rv)):
        past_rv, past_y = rv[:t], yy[:t]
        ok = np.isfinite(past_rv) & np.isfinite(past_y)
        if ok.sum() < 100 or not np.isfinite(rv[t]):
            continue
        q1, q2 = np.quantile(past_rv[ok], [1 / 3, 2 / 3])
        bucket = 0 if rv[t] <= q1 else (1 if rv[t] <= q2 else 2)
        pb = np.where(past_rv[ok] <= q1, 0, np.where(past_rv[ok] <= q2, 1, 2))
        sel = past_y[ok][pb == bucket]
        out[t] = sel.mean() if len(sel) >= 20 else past_y[ok].mean()
    return out


def h2_retest(features: pd.DataFrame, states: pd.DataFrame, n_splits: int = 5,
              threshold: float = BIG_DOWN) -> pd.DataFrame:
    """Regime-conditional tail probability against two benchmarks.

    Returns one row per contrast. `delta` negative means the regime forecast was better;
    `t_stat` is Diebold-Mariano with Newey-West errors at lag 1 (the target is one-day, so
    the loss differentials are only weakly autocorrelated).
    """
    from msl.estimators.base import PROB_COLUMNS

    y_full = big_move_target(features, threshold)
    terc = pd.Series(_tercile_rate(features, y_full), index=features.index, name="tercile")

    df = pd.concat([features[BASE_FEATURES], states[PROB_COLUMNS], terc,
                    y_full.rename("y")], axis=1).dropna()
    if len(df) < 300 or df["y"].nunique() < 2:
        return pd.DataFrame()

    y = df["y"].to_numpy(int)
    X_vol = df[BASE_FEATURES].to_numpy(float)
    X_reg = df[list(BASE_FEATURES) + list(PROB_COLUMNS)].to_numpy(float)

    p_vol = _walk_forward_proba(X_vol, y, n_splits, gap=1)
    p_reg = _walk_forward_proba(X_reg, y, n_splits, gap=1)
    p_ter = df["tercile"].to_numpy(float)

    ok = np.isfinite(p_vol) & np.isfinite(p_reg) & np.isfinite(p_ter)
    if ok.sum() < 200:
        return pd.DataFrame()
    yb = y[ok]
    L = {k: (np.clip(v[ok], 1e-6, 1 - 1e-6) - yb) ** 2
         for k, v in (("tercile", p_ter), ("vol_logistic", p_vol), ("regime", p_reg))}

    rows = []
    for bench in ("tercile", "vol_logistic"):
        t, _ = diebold_mariano(L["regime"], L[bench], lag=1)
        rows.append({
            "benchmark": bench, "n": int(ok.sum()), "base_rate": float(yb.mean()),
            "brier_benchmark": float(L[bench].mean()), "brier_regime": float(L["regime"].mean()),
            "delta": float(L["regime"].mean() - L[bench].mean()), "t_stat": t,
        })
    return pd.DataFrame(rows)
