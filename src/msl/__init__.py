"""market-state-lab — a benchmark harness for out-of-sample market-state identification.

RQ1: *which* statistical, econometric and ML methods most reliably identify trend,
volatility, liquidity and structural-change states out of sample?

Because RQ1 is comparative, this package is a protocol rather than a model: one data
layer, one shared feature set, a pluggable estimator interface, one walk-forward
engine, and one metrics suite. Methods are interchangeable parts competing on
identical terms, and transparent baselines are the bar every method must clear.
"""

__version__ = "0.1.0"

from msl.estimators.base import STATES, StateEstimator, get_estimator, list_estimators, register

__all__ = ["STATES", "StateEstimator", "register", "get_estimator", "list_estimators", "__version__"]
