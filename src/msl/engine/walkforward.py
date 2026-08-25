"""The walk-forward engine — one protocol, every method, every asset.

Estimators that learn parameters are refit on an expanding window at a fixed
cadence, with an embargo between the end of training and the scored rows, so no
training row shares information with the row it is scored on. Stateless rules skip
refitting but run through the identical path, which is what keeps the comparison
fair.

Everything lands in one tidy long frame:

    date | symbol | method | p_down | p_range | p_up | map_state | score

so adding a method or an asset never changes a single line of downstream code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from msl.estimators.base import OUTPUT_COLUMNS, StateEstimator, get_estimator, validate_output
from msl.features.core import build_features


def run_estimator(
    features: pd.DataFrame,
    estimator: StateEstimator,
    min_train: int = 750,
    refit_every: int = 63,
    embargo: int = 5,
) -> pd.DataFrame:
    """Walk-forward state estimates for one estimator on one asset."""
    if not estimator.requires_fit:
        out = estimator.filter(features)
        return validate_output(out, features.index, estimator.name)

    n = len(features)
    if n <= min_train + embargo:
        raise ValueError(f"need > {min_train + embargo} rows, got {n}")

    out = pd.DataFrame(index=features.index, columns=OUTPUT_COLUMNS, dtype=object)
    for start in range(min_train, n, refit_every):
        stop = min(start + refit_every, n)
        train_end = max(0, start - embargo)          # embargo: drop the rows adjacent to the block
        estimator.fit(features.iloc[:train_end])
        # filter over the whole prefix (the estimator is causal), then keep only new rows
        block = estimator.filter(features.iloc[:stop]).iloc[start:stop]
        out.iloc[start:stop] = block.values

    out[[c for c in OUTPUT_COLUMNS if c != "map_state"]] = out[
        [c for c in OUTPUT_COLUMNS if c != "map_state"]
    ].astype(float)
    return out


def run_sweep(
    prices: dict[str, pd.DataFrame],
    methods: list[str] | list[tuple[str, dict]],
    min_train: int = 750,
    refit_every: int = 63,
    embargo: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run every method across every asset; return one tidy long frame."""
    rows = []
    for symbol, px in prices.items():
        feats = build_features(px)
        for spec in methods:
            name, params = (spec, {}) if isinstance(spec, str) else spec
            est = get_estimator(name, **params)
            try:
                out = run_estimator(feats, est, min_train, refit_every, embargo)
            except Exception as exc:
                print(f"  [warn] {name} failed on {symbol}: {exc}")
                continue
            out = out.copy()
            out.insert(0, "method", name)
            out.insert(0, "symbol", symbol)
            rows.append(out)
            if verbose:
                cov = out["map_state"].notna().mean()
                print(f"  {symbol:<10} {name:<14} rows={len(out):<6} scored={cov:5.1%}")
    if not rows:
        raise RuntimeError("sweep produced no results")
    return pd.concat(rows).reset_index()


def state_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Per (symbol, method): state mix, flip rate and mean duration.

    Flip rate and duration are the *stability* tier of the evidence hierarchy — a
    'trend state' that changes every few days is unusable regardless of accuracy,
    and needs no labels to measure.
    """
    out = []
    for (sym, meth), g in results.groupby(["symbol", "method"], sort=True):
        s = g.dropna(subset=["map_state"]).sort_values("date")["map_state"]
        if s.empty:
            continue
        flips = (s != s.shift()).sum() - 1
        flip_rate = flips / max(len(s) - 1, 1)
        out.append({
            "symbol": sym,
            "method": meth,
            "n": len(s),
            "pct_up": (s == "up").mean(),
            "pct_range": (s == "range").mean(),
            "pct_down": (s == "down").mean(),
            "flip_rate": flip_rate,
            "mean_duration_days": (1.0 / flip_rate) if flip_rate > 0 else np.nan,
        })
    return pd.DataFrame(out)
