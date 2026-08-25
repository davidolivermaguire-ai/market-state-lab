"""Synthetic markets with a known hidden regime.

On real data there is no ground-truth "trend state", so a method cannot be scored
directly. Synthetic data is where truth exists: simulate from a Markov-switching
drift/volatility process, and a method that cannot recover a regime of exactly the
kind it was designed for is disqualified before it ever touches a real series.

This is the weakest but cleanest tier of the evidence hierarchy — necessary, not
sufficient.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# (annualised drift, annualised vol) for down / range / up
REGIME_PARAMS = {
    0: (-0.25, 0.28),   # down
    1: (0.02, 0.12),    # range
    2: (0.18, 0.14),    # up
}


def make_regime_series(
    n: int = 2500,
    seed: int = 0,
    persistence: float = 0.985,
    start: str = "2010-01-04",
) -> tuple[pd.DataFrame, pd.Series]:
    """Simulate OHLCV with a hidden 3-state regime.

    Returns (prices, true_state) where `true_state` is the hidden regime index,
    aligned to the price index — the label the recovery metrics score against.
    """
    rng = np.random.default_rng(seed)
    k = len(REGIME_PARAMS)

    # persistent transition matrix: regimes last, as real ones do
    P = np.full((k, k), (1.0 - persistence) / (k - 1))
    np.fill_diagonal(P, persistence)

    states = np.empty(n, dtype=int)
    states[0] = 1
    for t in range(1, n):
        states[t] = rng.choice(k, p=P[states[t - 1]])

    mu = np.array([REGIME_PARAMS[s][0] for s in states]) / 252.0
    sd = np.array([REGIME_PARAMS[s][1] for s in states]) / np.sqrt(252.0)
    r = rng.normal(mu, sd)

    close = 100.0 * np.exp(np.cumsum(r))
    # plausible OHLC around the close, and volume that rises with volatility
    noise = np.abs(rng.normal(0, sd * 0.6))
    high = close * (1.0 + noise)
    low = close * (1.0 - np.abs(rng.normal(0, sd * 0.6)))
    open_ = np.r_[close[0], close[:-1]] * (1.0 + rng.normal(0, sd * 0.3))
    volume = rng.lognormal(mean=16.0 + 4.0 * (sd - sd.mean()) / sd.std(), sigma=0.25)

    idx = pd.bdate_range(start=start, periods=n, name="date")
    px = pd.DataFrame(
        {"Open": open_, "High": np.maximum(high, np.maximum(open_, close)),
         "Low": np.minimum(low, np.minimum(open_, close)), "Close": close, "Volume": volume},
        index=idx,
    )
    return px, pd.Series(states, index=idx, name="true_state")
