"""Tier 4 — decision value: is the state estimate actually *worth* anything?

The other tiers ask whether methods agree with each other, persist, and recover known
regimes. This one asks the question RQ1 ultimately turns on:

    Does conditioning on a method's state estimate improve out-of-sample forecast
    calibration and risk control, **beyond what a continuous volatility signal using
    the same information already provides**?

That final clause is the whole design. Comparing a state-conditional model against
buy-and-hold, or against nothing, proves only that volatility is informative — which
is not in dispute. The benchmark here is a model that sees the *same* volatility
features and simply does not see the state. Whatever the state adds over that is the
incremental value of state information, and nothing else.

Two questions, deliberately separated
-------------------------------------
**Calibration.** Walk-forward logistic regression on `[rv20, rv60]` (baseline) versus
`[rv20, rv60, p_down, p_range, p_up]` (state-augmented), scored by Brier and log-loss
out of sample. Three targets, because the project's own evidence says state should help
on some and not others:

* ``direction`` — next h-day return positive. A return-timing target: expected to be a
  **null**, and reported as such.
* ``loss`` — next h-day return below minus one trailing standard deviation. A downside
  risk target.
* ``volatility`` — next h-day realised volatility above its trailing median.

**Risk control.** Volatility-targeted sizing with and without a state overlay, compared
on Sharpe, drawdown and turnover, net of costs. The state rule is pre-committed and
deliberately defensive — cut exposure in proportion to `p_down` — because the claim
under test is risk control, not return timing.

Everything is causal: features and exposures use only data up to t, labels look only
forward, and the walk-forward splits are purged by the label horizon so no training row
shares its forward window with a scored row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from msl.estimators.base import PROB_COLUMNS

BASE_FEATURES = ["rv20", "rv60"]
TARGETS = ("direction", "loss", "volatility")
ANN = np.sqrt(252.0)


def build_targets(features: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Forward-looking labels. Used only as targets — never as inputs."""
    r = features["ret"].astype(float)
    fwd_ret = r.rolling(horizon).sum().shift(-horizon)          # r[t+1 .. t+h]
    fwd_vol = r.rolling(horizon).std().shift(-horizon) * ANN

    out = pd.DataFrame(index=features.index)
    out["direction"] = (fwd_ret > 0).astype(float)
    out["loss"] = (fwd_ret < -features["rv20"].astype(float) / ANN * np.sqrt(horizon)).astype(float)
    out["volatility"] = (fwd_vol > fwd_vol.rolling(252, min_periods=60).median()).astype(float)
    out.loc[fwd_ret.isna(), :] = np.nan
    return out


def instability(states: pd.DataFrame, window: int = 60) -> pd.Series:
    """Causal rolling flip rate — how often the estimator has been changing its mind.

    This exists to test a specific confound. The least *stable* estimator in the panel
    showed the largest volatility-calibration gain, which invites an alternative
    explanation: a state that oscillates rapidly is re-encoding volatility through its
    own instability rather than contributing state information. Feed this to the
    **baseline** and any remaining advantage must come from something else.
    """
    s = states["map_state"] if "map_state" in states else states[PROB_COLUMNS].idxmax(axis=1)
    changed = (s != s.shift()).astype(float)
    changed[s.isna()] = np.nan
    return changed.rolling(window, min_periods=window // 2).mean()


def _nw_var_of_mean(x: np.ndarray, lag: int) -> float:
    """Newey-West (HAC) variance of the sample mean.

    Overlapping h-day labels make the loss differentials serially correlated, so an
    iid standard error would overstate significance — exactly the mistake that turns
    noise into a finding.
    """
    n = len(x)
    xm = x - x.mean()
    v = float(xm @ xm) / n
    for l in range(1, max(lag, 0) + 1):
        if l >= n:
            break
        w = 1.0 - l / (lag + 1.0)
        v += 2.0 * w * float(xm[l:] @ xm[:-l]) / n
    return max(v, 1e-18) / n


def diebold_mariano(loss_state: np.ndarray, loss_base: np.ndarray, lag: int) -> tuple[float, float]:
    """DM test on a loss differential. Negative t = the state model has lower loss."""
    from scipy.stats import norm

    d = loss_state - loss_base
    t = float(d.mean() / np.sqrt(_nw_var_of_mean(d, lag)))
    return t, float(2.0 * (1.0 - norm.cdf(abs(t))))


def _walk_forward_proba(X: np.ndarray, y: np.ndarray, n_splits: int, gap: int) -> np.ndarray:
    """Out-of-sample probabilities from a purged, expanding-window logistic regression."""
    proba = np.full(len(y), np.nan)
    for tr, te in TimeSeriesSplit(n_splits=n_splits, gap=gap).split(X):
        if len(np.unique(y[tr])) < 2:
            continue
        scaler = StandardScaler().fit(X[tr])
        model = LogisticRegression(max_iter=1000, C=1.0)
        model.fit(scaler.transform(X[tr]), y[tr])
        proba[te] = model.predict_proba(scaler.transform(X[te]))[:, 1]
    return proba


def calibration_gain(
    states: pd.DataFrame,
    features: pd.DataFrame,
    horizon: int = 5,
    n_splits: int = 5,
    control_instability: bool = False,
) -> pd.DataFrame:
    """Brier/log-loss for baseline vs state-augmented forecasts, per target.

    A **negative** `brier_delta` means the state improved calibration. `t_stat` is a
    Diebold-Mariano statistic on the per-observation loss differential with
    Newey-West standard errors (negative = state better).

    `control_instability=True` adds the estimator's own rolling flip rate to **both**
    models, so the baseline already sees how much the state has been oscillating. Any
    remaining gain cannot be instability-as-volatility-proxy.
    """
    targets = build_targets(features, horizon)
    base_cols = list(BASE_FEATURES)
    parts = [features[BASE_FEATURES], states[PROB_COLUMNS], targets]
    if control_instability:
        parts.append(instability(states).rename("flip_rate"))
        base_cols = base_cols + ["flip_rate"]
    df = pd.concat(parts, axis=1).dropna()
    if len(df) < 300:
        return pd.DataFrame()

    X_base = df[base_cols].to_numpy(dtype=float)
    X_state = df[base_cols + PROB_COLUMNS].to_numpy(dtype=float)

    rows = []
    for target in TARGETS:
        y = df[target].to_numpy(dtype=int)
        if len(np.unique(y)) < 2:
            continue
        p_base = _walk_forward_proba(X_base, y, n_splits, horizon)
        p_state = _walk_forward_proba(X_state, y, n_splits, horizon)
        ok = ~(np.isnan(p_base) | np.isnan(p_state))
        if ok.sum() < 100:
            continue
        yb, pb, ps = y[ok], np.clip(p_base[ok], 1e-6, 1 - 1e-6), np.clip(p_state[ok], 1e-6, 1 - 1e-6)
        t_stat, p_value = diebold_mariano((ps - yb) ** 2, (pb - yb) ** 2, lag=horizon)
        rows.append({
            "target": target,
            "n": int(ok.sum()),
            "base_rate": float(yb.mean()),
            "brier_base": brier_score_loss(yb, pb),
            "brier_state": brier_score_loss(yb, ps),
            "brier_delta": brier_score_loss(yb, ps) - brier_score_loss(yb, pb),
            "t_stat": t_stat,
            "p_value": p_value,
            "logloss_base": log_loss(yb, pb),
            "logloss_state": log_loss(yb, ps),
        })
    return pd.DataFrame(rows)


def deflate(cal: pd.DataFrame, fdr: float = 0.10) -> pd.DataFrame:
    """Correct a grid of calibration comparisons for multiple testing.

    Two corrections, because they answer different questions:

    * **Benjamini-Hochberg FDR** — controls the expected proportion of false claims
      among those declared significant.
    * **Expected maximum |t|** — under the null, the largest of N independent
      t-statistics is around ``sqrt(2 ln N)``. Any result below that threshold is what
      searching N times produces by chance, which is the Bailey / López de Prado point
      applied to loss differentials instead of Sharpe ratios. With 189 comparisons the
      bar is |t| > 3.24, not 1.96.

    Adds `p_fdr`, `survives_fdr`, `t_threshold` and `survives_deflated`.
    """
    out = cal.copy()
    n = len(out)
    if n == 0:
        return out

    order = np.argsort(out["p_value"].to_numpy())
    ranked = out["p_value"].to_numpy()[order]
    adj = np.minimum.accumulate((ranked * n / np.arange(1, n + 1))[::-1])[::-1]
    p_fdr = np.empty(n)
    p_fdr[order] = np.clip(adj, 0.0, 1.0)

    threshold = float(np.sqrt(2.0 * np.log(max(n, 2))))
    out["p_fdr"] = p_fdr
    out["survives_fdr"] = (out["p_fdr"] < fdr) & (out["brier_delta"] < 0)
    out["t_threshold"] = threshold
    out["survives_deflated"] = (out["t_stat"] < -threshold)
    return out


def _perf(rets: np.ndarray, exposure: np.ndarray, cost_bps: float) -> dict:
    """Sharpe, volatility, max drawdown and turnover for an exposure path, net of costs."""
    turnover = np.abs(np.diff(np.r_[0.0, exposure]))
    net = exposure * rets - turnover * (cost_bps / 1e4)
    eq = np.cumprod(1.0 + net)
    peak = np.maximum.accumulate(eq)
    sd = float(np.std(net))
    dd = float((eq / peak - 1.0).min())
    vol = sd * ANN
    return {
        "sharpe": float(np.mean(net) / sd * ANN) if sd > 0 else np.nan,
        "vol": vol,
        "max_drawdown": dd,
        # Raw drawdown across strategies at different volatilities is a confounded
        # comparison: a rule that merely holds less will always show less drawdown.
        # Normalising by realised volatility is what makes the comparison honest —
        # as does Sharpe, which is already scale-invariant.
        "dd_per_vol": dd / vol if vol > 0 else np.nan,
        "turnover_pa": float(turnover.mean() * 252),
    }


def risk_control(
    states: pd.DataFrame,
    features: pd.DataFrame,
    target_vol: float = 0.10,
    max_leverage: float = 2.0,
    cost_bps: float = 2.0,
) -> pd.DataFrame:
    """Volatility targeting with and without a state overlay.

    The baseline is the standard volatility scaler: exposure = target / trailing vol.
    The state variant multiplies it by ``(1 - p_down)`` — a pre-committed, purely
    defensive rule. It can only ever *reduce* exposure, so any improvement is risk
    control rather than a disguised return-timing bet.
    """
    df = pd.concat([features[["ret", "rv20"]], states[PROB_COLUMNS]], axis=1).dropna()
    if len(df) < 300:
        return pd.DataFrame()

    rv = df["rv20"].to_numpy(dtype=float)
    rets = df["ret"].to_numpy(dtype=float)
    p_down = df["p_down"].to_numpy(dtype=float)

    w_base = np.clip(target_vol / np.maximum(rv, 1e-6), 0.0, max_leverage)
    w_state = w_base * (1.0 - p_down)

    # exposure decided at t is applied to the return at t+1 — no look-ahead
    fwd = np.r_[rets[1:], np.nan]
    ok = ~np.isnan(fwd)

    rows = [
        {"strategy": "buy_and_hold", **_perf(fwd[ok], np.ones(ok.sum()), 0.0)},
        {"strategy": "vol_target (baseline)", **_perf(fwd[ok], w_base[ok], cost_bps)},
        {"strategy": "vol_target x state", **_perf(fwd[ok], w_state[ok], cost_bps)},
    ]
    return pd.DataFrame(rows)


def decision_value(results: pd.DataFrame, features_by_symbol: dict[str, pd.DataFrame],
                   horizon: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run both questions for every (symbol, method) in a tidy sweep result."""
    cal_rows, risk_rows = [], []
    for (sym, meth), g in results.groupby(["symbol", "method"], sort=True):
        feats = features_by_symbol.get(sym)
        if feats is None:
            continue
        states = g.set_index("date")[PROB_COLUMNS].apply(pd.to_numeric, errors="coerce")
        states = states.reindex(feats.index)

        cal = calibration_gain(states, feats, horizon)
        if not cal.empty:
            cal.insert(0, "method", meth); cal.insert(0, "symbol", sym)
            cal_rows.append(cal)

        rc = risk_control(states, feats)
        if not rc.empty:
            rc = rc[rc["strategy"] == "vol_target x state"].copy()
            base = risk_control(states, feats)
            b = base[base["strategy"] == "vol_target (baseline)"].iloc[0]
            rc["sharpe_delta"] = rc["sharpe"].to_numpy() - b["sharpe"]
            rc["drawdown_delta"] = rc["max_drawdown"].to_numpy() - b["max_drawdown"]
            rc.insert(0, "method", meth); rc.insert(0, "symbol", sym)
            risk_rows.append(rc)

    cal = pd.concat(cal_rows, ignore_index=True) if cal_rows else pd.DataFrame()
    risk = pd.concat(risk_rows, ignore_index=True) if risk_rows else pd.DataFrame()
    return cal, risk
