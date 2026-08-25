"""The shared information set.

Every estimator sees exactly these features, which is the fairness constraint that
makes RQ1 answerable: if one method saw a richer feature set, a win would tell you
nothing about the *method*.

Every column is strictly causal — computed from data at or before each row's date.
There are no centred windows, no `shift(-n)`, and nothing that peeks forward. The
look-ahead test in tests/ enforces this mechanically rather than on trust.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "ret", "rv5", "rv20", "rv60", "mom20", "mom60",
    "ma_spread", "range", "gap", "volz", "amihud",
]

ANN = np.sqrt(252.0)


def build_features(px: pd.DataFrame) -> pd.DataFrame:
    """Daily OHLCV -> the shared causal feature frame (plus `close` for reference)."""
    c, h, l, o = px["Close"], px["High"], px["Low"], px["Open"]
    v = px["Volume"].astype(float)
    r = np.log(c).diff()

    f = pd.DataFrame(index=px.index)
    f["ret"] = r
    # realised volatility over three horizons: the transparent state signal
    f["rv5"] = r.rolling(5).std() * ANN
    f["rv20"] = r.rolling(20).std() * ANN
    f["rv60"] = r.rolling(60).std() * ANN
    # trend / momentum
    f["mom20"] = np.log(c / c.shift(20))
    f["mom60"] = np.log(c / c.shift(60))
    f["ma_spread"] = c.rolling(50).mean() / c.rolling(200).mean() - 1.0
    # microstructure-ish stress proxies from daily bars
    f["range"] = (h - l) / c
    f["gap"] = (o - c.shift(1)).abs() / c.shift(1)
    lv = np.log(v.replace(0.0, np.nan))
    f["volz"] = (lv - lv.rolling(20).mean()) / lv.rolling(20).std()
    f["amihud"] = r.abs() / (v / v.rolling(20).mean())

    f["close"] = c
    return f


def zscore_causal(s: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    """Rolling z-score using only past data (no centring, no full-sample moments)."""
    mu = s.rolling(window, min_periods=min_periods).mean()
    sd = s.rolling(window, min_periods=min_periods).std()
    return (s - mu) / sd.replace(0.0, np.nan)
