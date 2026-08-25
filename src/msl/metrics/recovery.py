"""Tier 1 of the evidence hierarchy: recovery on data where truth exists.

Real markets have no ground-truth "trend state", so this is the one place a state
estimator can be scored directly. Simulate from a Markov-switching drift/volatility
process, hand each method the same features, and ask whether it recovers the hidden
regime. A method that cannot recover a regime of exactly the kind it was designed
for is disqualified before it touches a real series.

Necessary, not sufficient — passing here says the method works when its assumptions
hold, not that those assumptions hold in the market.

Label permutation
-----------------
Unsupervised methods (HMM, GMM) return arbitrary state *labels*: their "state 2" need
not be our "up". Scoring them on raw labels would be meaningless, so every metric is
computed after an optimal label matching (Hungarian assignment on the confusion
matrix). Semantically-labelled methods such as the baselines are unaffected — for
them `accuracy` and `accuracy_matched` should agree, which is a useful sanity check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from msl.engine.synthetic import make_regime_series
from msl.estimators.base import PROB_COLUMNS, STATES, get_estimator, list_estimators
from msl.features.core import build_features

K = len(STATES)
STATE_INDEX = {s: i for i, s in enumerate(STATES)}  # down=0, range=1, up=2 (matches the simulator)


def _match_labels(pred_idx: np.ndarray, true_idx: np.ndarray) -> dict[int, int]:
    """Optimal predicted-label -> true-label mapping (maximises agreement)."""
    cm = np.zeros((K, K))
    for p, t in zip(pred_idx, true_idx):
        cm[int(p), int(t)] += 1.0
    rows, cols = linear_sum_assignment(-cm)
    return {int(r): int(c) for r, c in zip(rows, cols)}


def _balanced_accuracy(pred_idx: np.ndarray, true_idx: np.ndarray) -> float:
    """Mean per-class recall — the right accuracy when regimes are imbalanced."""
    accs = [float((pred_idx[true_idx == t] == t).mean()) for t in range(K) if (true_idx == t).any()]
    return float(np.mean(accs)) if accs else np.nan


def _detection(pred_idx: np.ndarray, true_idx: np.ndarray, horizon: int = 60, tol: int = 10):
    """Median delay to flag a true regime change, detection rate, and false alarms/yr.

    A detection requires an actual *transition into* the new state within
    [cp - tol, cp + horizon]. Merely already sitting in that state does not count —
    otherwise a degenerate model that never moves would be credited with detecting
    every change to the state it is stuck in. Negative delays mean anticipation.
    """
    true_cp = np.flatnonzero(np.diff(true_idx) != 0) + 1
    # indices where the prediction transitions INTO a state
    entries = np.flatnonzero(pred_idx[1:] != pred_idx[:-1]) + 1

    delays: list[float] = []
    for cp in true_cp:
        target = true_idx[cp]
        cand = entries[(entries >= cp - tol) & (entries <= cp + horizon) & (pred_idx[entries] == target)]
        delays.append(float(cand[0] - cp) if cand.size else np.nan)

    detected = np.array([d for d in delays if not np.isnan(d)])
    detection_rate = len(detected) / len(delays) if delays else np.nan
    median_delay = float(np.median(detected)) if detected.size else np.nan

    pred_cp = np.flatnonzero(np.diff(pred_idx) != 0) + 1
    if true_cp.size and pred_cp.size:
        near = np.min(np.abs(pred_cp[:, None] - true_cp[None, :]), axis=1)
        false_alarms = int((near > tol).sum())
    else:
        false_alarms = int(pred_cp.size)
    per_year = false_alarms / (len(true_idx) / 252.0)

    return median_delay, detection_rate, per_year


def recovery_metrics(pred: pd.DataFrame, true_state: pd.Series, horizon: int = 60) -> dict:
    """Score one estimator's output against the known hidden regime."""
    df = pred.join(true_state.rename("true"), how="inner").dropna(subset=["map_state", "true"])
    if df.empty:
        return {"n": 0}

    pred_idx = df["map_state"].map(STATE_INDEX).to_numpy(dtype=int)
    true_idx = df["true"].to_numpy(dtype=int)
    probs = df[PROB_COLUMNS].to_numpy(dtype=float)

    raw_acc = float((pred_idx == true_idx).mean())

    perm = _match_labels(pred_idx, true_idx)
    matched = np.array([perm.get(int(p), int(p)) for p in pred_idx])
    # permute the probability columns the same way so Brier is scored on matched labels
    probs_m = np.zeros_like(probs)
    for p_lbl, t_lbl in perm.items():
        probs_m[:, t_lbl] = probs[:, p_lbl]

    onehot = np.zeros_like(probs_m)
    onehot[np.arange(len(true_idx)), true_idx] = 1.0
    brier = float(np.mean(np.sum((probs_m - onehot) ** 2, axis=1)))

    delay, det_rate, fa = _detection(matched, true_idx, horizon=horizon)

    return {
        "n": int(len(df)),
        "accuracy": raw_acc,
        "accuracy_matched": float((matched == true_idx).mean()),
        "balanced_accuracy": _balanced_accuracy(matched, true_idx),
        "ari": float(adjusted_rand_score(true_idx, pred_idx)),
        "brier": brier,
        "median_delay_days": delay,
        "detection_rate": det_rate,
        "false_alarms_per_year": fa,
    }


def run_recovery(
    methods: list[str] | None = None,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7),
    n: int = 2500,
    persistence: float = 0.985,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the recovery suite over several simulated markets.

    Returns (per-seed rows, aggregate by method). Several seeds matter: a single
    simulated market is one draw, and ranking methods on one draw is the mistake the
    whole harness exists to avoid.
    """
    methods = methods or list_estimators()
    rows = []
    for seed in seeds:
        px, true_state = make_regime_series(n=n, seed=seed, persistence=persistence)
        feats = build_features(px)
        for name in methods:
            out = get_estimator(name).filter(feats)
            m = recovery_metrics(out, true_state)
            if m.get("n"):
                rows.append({"method": name, "seed": seed, **m})

    per_seed = pd.DataFrame(rows)
    agg = (
        per_seed.drop(columns=["seed"])
        .groupby("method", as_index=False)
        .mean(numeric_only=True)
        .sort_values("balanced_accuracy", ascending=False)
        .reset_index(drop=True)
    )
    return per_seed, agg
