"""Transparent baselines — the bar every sophisticated method must clear.

These are not filler. RQ1 asks which methods identify states *most reliably*, and a
Kalman filter or an HMM that cannot beat a 50/200 moving-average crossover has not
earned its complexity. Keeping the baselines in the same harness, on the same
features, with the same metrics, is what turns that from an opinion into a result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from msl.estimators.base import StateEstimator, register, softmax_states
from msl.features.core import zscore_causal


@register("ma_cross")
class MovingAverageCross(StateEstimator):
    """Classic trend rule: fast MA above slow MA is an up-trend.

    The raw spread is z-scored on a trailing window so the score is comparable
    across assets — a 1% spread means something different for NAS100 than for a
    single name.
    """

    requires_fit = False

    def __init__(self, fast: int = 50, slow: int = 200, z_window: int = 252):
        super().__init__(fast=fast, slow=slow, z_window=z_window)
        self.fast, self.slow, self.z_window = fast, slow, z_window

    def filter(self, features: pd.DataFrame) -> pd.DataFrame:
        c = features["close"]
        spread = c.rolling(self.fast).mean() / c.rolling(self.slow).mean() - 1.0
        score = np.tanh(zscore_causal(spread, self.z_window))
        return softmax_states(score)


@register("return_sign")
class TrailingReturnSign(StateEstimator):
    """Sign and size of the trailing return, scaled by trailing volatility."""

    requires_fit = False

    def __init__(self, window: int = 60):
        super().__init__(window=window)
        self.window = window

    def filter(self, features: pd.DataFrame) -> pd.DataFrame:
        r = features["ret"]
        mu = r.rolling(self.window).mean()
        sd = r.rolling(self.window).std().replace(0.0, np.nan)
        score = np.tanh((mu / sd) * np.sqrt(self.window))
        return softmax_states(score)


@register("ewma_slope")
class EwmaSlope(StateEstimator):
    """Slope of an exponentially weighted mean of log price, in volatility units."""

    requires_fit = False

    def __init__(self, span: int = 40, z_window: int = 252):
        super().__init__(span=span, z_window=z_window)
        self.span, self.z_window = span, z_window

    def filter(self, features: pd.DataFrame) -> pd.DataFrame:
        logc = np.log(features["close"])
        ewm = logc.ewm(span=self.span, adjust=False).mean()
        slope = ewm.diff()
        score = np.tanh(zscore_causal(slope, self.z_window))
        return softmax_states(score)


@register("always_range")
class AlwaysRange(StateEstimator):
    """The null model: never claims a trend. The floor any real method must beat."""

    requires_fit = False

    def filter(self, features: pd.DataFrame) -> pd.DataFrame:
        return softmax_states(pd.Series(0.0, index=features.index))
