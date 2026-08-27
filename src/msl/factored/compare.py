"""Factored (3x3) versus flat (9-state) market state, at matched cardinality.

Both descriptions carry exactly nine joint cells over the same two observations — a trend
proxy (daily return) and a volatility proxy (log realised volatility). The factored model
estimates two independent 3-state chains; the flat model estimates one 9-state chain. The
flat model can represent any dependence between the axes and the factored model cannot,
so the flat model is strictly more expressive and strictly more expensive. H1 says the
cheaper, misspecified one wins out of sample.

Everything is walk-forward and filtered: parameters are fitted on a training window, the
posteriors are produced by a forward-only pass, and the scoring reuses the same purged
walk-forward logistic regression and Diebold-Mariano machinery as the RQ1 experiment, so
the two results are directly comparable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from msl.factored.hmm_k import GaussHMM
from msl.metrics.decision import (BASE_FEATURES, TARGETS, build_targets,
                                  diebold_mariano, _walk_forward_proba)

TREND_OBS = "ret"
VOL_OBS = "rv20"


def _observations(features: pd.DataFrame) -> pd.DataFrame:
    """The two axis observations, causal by construction.

    Volatility is modelled in logs: realised volatility is right-skewed and strictly
    positive, and a Gaussian emission on the raw level would put most of a state's mass
    where the data cannot go.
    """
    obs = pd.DataFrame(index=features.index)
    obs["trend"] = features[TREND_OBS]
    obs["vol"] = np.log(features[VOL_OBS].clip(lower=1e-8))
    return obs


def _walk_forward_posteriors(obs: pd.DataFrame, mode: str, n_states: int,
                             min_train: int = 750, refit_every: int = 63,
                             max_train: int = 1250, seed: int = 0) -> pd.DataFrame:
    """Refit periodically, filter forward. Row t never uses data after t.

    `mode="factored"` fits one chain per axis; `mode="flat"` fits a single chain on both
    axes jointly. Column count differs (3+3 versus 9) but the joint cell count does not.
    """
    n = len(obs)
    cols = ([f"t{i}" for i in range(n_states)] + [f"v{i}" for i in range(n_states)]
            if mode == "factored" else [f"s{i}" for i in range(n_states)])
    out = pd.DataFrame(np.nan, index=obs.index, columns=cols)
    if n <= min_train:
        return out

    models: dict = {}
    for start in range(min_train, n, refit_every):
        lo = max(0, start - max_train)
        train = obs.iloc[lo:start]
        stop = min(start + refit_every, n)
        # filter from the start of the training window so the forward recursion has run
        # up before the scored block; only rows [start:stop) are kept.
        ctx = obs.iloc[lo:stop]

        if mode == "factored":
            for axis, key in (("trend", "t"), ("vol", "v")):
                m = models.get(key) or GaussHMM(n_states, seed=seed)
                models[key] = m.fit(train[[axis]].to_numpy())
                p = m.filter(ctx[[axis]].to_numpy())
                block = p[start - lo:stop - lo]
                out.iloc[start:stop, [cols.index(f"{key}{i}") for i in range(n_states)]] = block
        else:
            m = models.get("flat") or GaussHMM(n_states, seed=seed)
            models["flat"] = m.fit(train.to_numpy())
            p = m.filter(ctx.to_numpy())
            out.iloc[start:stop, :] = p[start - lo:stop - lo]
    return out


def _joint_map(post: pd.DataFrame, mode: str, n_states: int) -> pd.Series:
    """Collapse posteriors to one label per day, over the same 9 joint cells."""
    if mode == "factored":
        t = post[[f"t{i}" for i in range(n_states)]].to_numpy()
        v = post[[f"v{i}" for i in range(n_states)]].to_numpy()
        lab = np.argmax(t, axis=1) * n_states + np.argmax(v, axis=1)
        bad = ~np.isfinite(t).all(1) | ~np.isfinite(v).all(1)
    else:
        s = post.to_numpy()
        lab = np.argmax(s, axis=1)
        bad = ~np.isfinite(s).all(1)
    lab = lab.astype(float)
    lab[bad] = np.nan
    return pd.Series(lab, index=post.index, name="joint")


def stability(labels: pd.Series) -> dict:
    """Temporal stability of a label sequence — H1's first claim."""
    s = labels.dropna()
    if len(s) < 10:
        return {"mean_duration": np.nan, "flip_rate": np.nan, "n_cells_used": 0}
    flips = (s.to_numpy()[1:] != s.to_numpy()[:-1]).sum()
    return {
        "mean_duration": float(len(s) / max(flips, 1)),
        "flip_rate": float(flips / (len(s) - 1)),
        "n_cells_used": int(s.nunique()),
    }


def factored_vs_flat(features: pd.DataFrame, n_states: int = 3, horizon: int = 5,
                     n_splits: int = 5, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the whole comparison for one asset.

    Returns (decision_value, stability). Decision value scores three feature sets on the
    same rows: the volatility-only baseline, baseline + factored posteriors, and
    baseline + flat posteriors. `delta_vs_base` negative means the state helped;
    `t_factored_vs_flat` negative means the factored state beat the flat label directly.
    """
    obs = _observations(features)
    fac = _walk_forward_posteriors(obs, "factored", n_states, seed=seed)
    flat = _walk_forward_posteriors(obs, "flat", n_states * n_states, seed=seed)

    lab_fac = _joint_map(fac, "factored", n_states)
    lab_flat = _joint_map(flat, "flat", n_states * n_states)
    K, Kf = n_states, n_states ** 2
    n_par_fac = 2 * (K * (K - 1) + K * 1 + K * 1)      # two univariate chains
    n_par_flat = Kf * (Kf - 1) + Kf * 2 + Kf * 2       # one bivariate chain
    stab = pd.DataFrame([
        {"model": f"factored {K}x{K}", "n_params": n_par_fac, **stability(lab_fac)},
        {"model": f"flat {Kf}-state", "n_params": n_par_flat, **stability(lab_flat)},
    ])

    targets = build_targets(features, horizon)
    df = pd.concat([features[BASE_FEATURES], fac, flat, targets], axis=1).dropna()
    if len(df) < 300:
        return pd.DataFrame(), stab

    X_base = df[BASE_FEATURES].to_numpy(float)
    X_fac = df[list(BASE_FEATURES) + list(fac.columns)].to_numpy(float)
    X_flat = df[list(BASE_FEATURES) + list(flat.columns)].to_numpy(float)

    rows = []
    for target in TARGETS:
        y = df[target].to_numpy(int)
        if len(np.unique(y)) < 2:
            continue
        p = {k: _walk_forward_proba(X, y, n_splits, horizon)
             for k, X in (("base", X_base), ("fac", X_fac), ("flat", X_flat))}
        ok = ~np.any([np.isnan(v) for v in p.values()], axis=0)
        if ok.sum() < 100:
            continue
        yb = y[ok]
        L = {k: (np.clip(v[ok], 1e-6, 1 - 1e-6) - yb) ** 2 for k, v in p.items()}
        for name in ("fac", "flat"):
            t, _ = diebold_mariano(L[name], L["base"], lag=horizon)
            rows.append({"target": target, "model": "factored" if name == "fac" else "flat",
                         "n": int(ok.sum()), "brier": float(L[name].mean()),
                         "brier_base": float(L["base"].mean()),
                         "delta_vs_base": float(L[name].mean() - L["base"].mean()),
                         "t_vs_base": t})
        t_ff, _ = diebold_mariano(L["fac"], L["flat"], lag=horizon)
        rows[-2]["t_factored_vs_flat"] = t_ff
        rows[-1]["t_factored_vs_flat"] = t_ff
    return pd.DataFrame(rows), stab
