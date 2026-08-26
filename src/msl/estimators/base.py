"""The estimator contract and registry.

Two contracts carry the whole design:

1. **Causality.** `filter()` is a *filtered* estimate: the value at time t may use
   only rows at or before t. Never a smoothed pass over a completed history — that
   is look-ahead, and it is the single easiest way to produce a beautiful, useless
   result. tests/test_no_lookahead.py enforces it for every registered estimator.

2. **One output schema.** Every estimator emits the same frame: a probability
   vector over STATES, a MAP label, and a continuous `score` in [-1, 1]. Discrete
   methods (HMM) and continuous ones (Kalman slope) therefore stay comparable, and
   the continuous score keeps RQ2 (overlapping dimensions vs one exclusive label)
   open rather than pre-judged.

Adding a method is one file plus a @register decorator; nothing downstream changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

# Fixed order, used everywhere. p_down + p_range + p_up == 1.
STATES = ["down", "range", "up"]
PROB_COLUMNS = [f"p_{s}" for s in STATES]
OUTPUT_COLUMNS = PROB_COLUMNS + ["map_state", "score"]

_REGISTRY: dict[str, type["StateEstimator"]] = {}


def register(name: str):
    """Class decorator: make an estimator available by name to configs and the CLI."""
    def _wrap(cls):
        if name in _REGISTRY:
            raise KeyError(f"estimator '{name}' is already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls
    return _wrap


def get_estimator(name: str, **params) -> "StateEstimator":
    if name not in _REGISTRY:
        raise KeyError(f"unknown estimator '{name}'. Known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**params)


def list_estimators() -> list[str]:
    return sorted(_REGISTRY)


class StateEstimator(ABC):
    """Base class for every market-state estimator.

    Attributes
    ----------
    name : str
        Set by @register.
    requires_fit : bool
        True if the estimator learns parameters (HMM, MS-AR). Stateless rules
        (moving-average crossover) leave it False and the engine skips refitting.
    kind : str
        "state" for estimators answering *which state*; "changepoint" for detectors
        answering *has it changed* — scored on timeliness, not classification.
    full_replay : bool
        True if the estimator's state has effectively unbounded memory, so the
        walk-forward engine must replay the whole prefix rather than a bounded run-up.
        Most filters forget the distant past geometrically and are safe to window; the
        local-linear-trend filter is not, because a near-zero slope variance makes its
        slope an integral of the entire history.
    """

    name: str = "unnamed"
    requires_fit: bool = False
    kind: str = "state"
    full_replay: bool = False

    def __init__(self, **params):
        self.params = params

    def fit(self, features: pd.DataFrame) -> "StateEstimator":
        """Learn parameters from a training slice. Stateless estimators need not override."""
        return self

    @abstractmethod
    def filter(self, features: pd.DataFrame) -> pd.DataFrame:
        """Causal state estimates for every row of `features`.

        Must return a frame indexed like `features` with OUTPUT_COLUMNS.
        """

    def __repr__(self) -> str:  # pragma: no cover
        return f"{type(self).__name__}({self.params})"


def softmax_states(score: pd.Series, sharpness: float = 2.0, range_margin: float = 0.6) -> pd.DataFrame:
    """Map a continuous trend score in [-1, 1] to a valid 3-state distribution.

    `range_margin` gives the neutral state a head start, so a score near zero reads
    as "range" rather than as an even three-way split — which is what a flat market
    actually is.
    """
    s = pd.Series(score, index=score.index).clip(-1.0, 1.0).fillna(0.0)
    logits = np.column_stack([-sharpness * s.values, np.full(len(s), range_margin), sharpness * s.values])
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    p = e / e.sum(axis=1, keepdims=True)

    out = pd.DataFrame(p, index=s.index, columns=PROB_COLUMNS)
    out["map_state"] = [STATES[i] for i in p.argmax(axis=1)]
    out["score"] = s
    return out[OUTPUT_COLUMNS]


def validate_output(out: pd.DataFrame, index: pd.Index, who: str = "estimator") -> pd.DataFrame:
    """Contract check: right columns, right index, probabilities that sum to one."""
    missing = [c for c in OUTPUT_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"{who}: output missing columns {missing}")
    if not out.index.equals(index):
        raise ValueError(f"{who}: output index does not match the input index")

    # Check only fully-populated rows. Warm-up rows are legitimately all-NaN, and
    # pandas sums an all-NaN object row to 0.0 rather than NaN — so a naive
    # `.sum(axis=1).dropna()` would reject a perfectly valid estimator.
    probs = out[PROB_COLUMNS].apply(pd.to_numeric, errors="coerce")
    complete = probs.notna().all(axis=1)
    tot = probs[complete].sum(axis=1)
    if len(tot) and not np.allclose(tot.to_numpy(), 1.0, atol=1e-6):
        raise ValueError(f"{who}: state probabilities do not sum to 1")
    return out[OUTPUT_COLUMNS]
