"""The per-ticker current-state view — the trend specialist's operational output.

For each symbol this reports every method's filtered state estimate as of the last
available bar, plus a consensus across methods and the council's own disagreement.
It is the same object the AMI console's specialist-evidence panel consumes.

Read it as a **state estimate with uncertainty, not a trade confirmation.** The
evidence on this project is consistent: market state earns its keep in calibration
and risk control, not in return timing. A high `p_up` says "the trend estimators
agree the drift has been positive and persistent", not "buy" — and `disagreement`
rising is the more actionable signal, because that is where the value of state
information concentrates.

The consensus here is a deliberately naive **equal-weight** average. The proposal's
reliability-weighted combination replaces it once the walk-forward evidence exists
to justify weights — equal weighting is the transparent baseline that has to be
beaten first.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from msl.estimators.base import PROB_COLUMNS, STATES, get_estimator
from msl.features.core import build_features

K = len(STATES)


def _entropy(p: np.ndarray) -> float:
    """Normalised Shannon entropy of a state distribution, 0 (certain) to 1 (split)."""
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)) / np.log(K))


def _decompose(vectors: list[np.ndarray]) -> tuple[float, float]:
    """Split consensus entropy into *uncertainty* and genuine *disagreement*.

        H(mean p)  =  mean H(p_i)   +   JSD
        uncertainty   individual doubt   disagreement between methods

    This matters. The entropy of the averaged distribution alone cannot tell the two
    apart: three methods that each shrug produce the same flat average as three
    methods that confidently contradict each other. Only the second is disagreement,
    and the proposal's claim — that the value of state information concentrates where
    the council disagrees — is about the second. The Jensen-Shannon divergence is the
    part that isolates it.
    """
    P = np.vstack(vectors)
    mean_p = P.mean(axis=0)
    mean_p = mean_p / max(mean_p.sum(), 1e-12)
    total = _entropy(mean_p)                                   # H(mean p)
    individual = float(np.mean([_entropy(p) for p in P]))      # mean H(p_i)
    jsd = max(0.0, total - individual)                         # disagreement
    return total, jsd


def state_report(
    prices: dict[str, pd.DataFrame],
    methods: list[str],
    as_of: str | None = None,
) -> pd.DataFrame:
    """Latest filtered trend state per symbol, per method, plus consensus.

    Returns one row per symbol: each method's MAP state, the equal-weight consensus
    state and its probability, the fraction of methods agreeing with it, and the
    normalised disagreement entropy.
    """
    rows = []
    for symbol, px in prices.items():
        px = px[px.index <= pd.Timestamp(as_of)] if as_of else px
        feats = build_features(px)

        per_method: dict[str, str] = {}
        vectors = []
        for name in methods:
            out = get_estimator(name).filter(feats).dropna(subset=["map_state"])
            if out.empty:
                per_method[name] = "n/a"
                continue
            last = out.iloc[-1]
            per_method[name] = str(last["map_state"])
            vectors.append(last[PROB_COLUMNS].to_numpy(dtype=float))

        if not vectors:
            continue

        mean_p = np.mean(vectors, axis=0)
        mean_p = mean_p / mean_p.sum()
        consensus = STATES[int(mean_p.argmax())]
        agree = float(np.mean([s == consensus for s in per_method.values() if s != "n/a"]))
        uncertainty, disagreement = _decompose(vectors)

        rows.append({
            "symbol": symbol,
            "as_of": feats.index[-1].date(),
            **{f"m_{k}": v for k, v in per_method.items()},
            "consensus": consensus,
            "p_consensus": round(float(mean_p.max()), 3),
            "p_up": round(float(mean_p[STATES.index("up")]), 3),
            "p_down": round(float(mean_p[STATES.index("down")]), 3),
            "agreement": round(agree, 3),
            "uncertainty": round(uncertainty, 3),
            "disagreement": round(disagreement, 3),
        })

    if not rows:
        raise RuntimeError("state_report produced no rows")
    return pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)
