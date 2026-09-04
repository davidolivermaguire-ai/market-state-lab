"""H8 re-test: do hard risk limits earn their keep, or do they just hold less?

H8 claims deterministic risk controls improve robustness and viability even where they cut
gross return. The original experiment compared a 2x leveraged long against the same 2x long
with limits attached, and found the limited versions had shallower drawdowns and higher
Calmar.

Every limit in that comparison **reduces exposure** — a cap takes 2x to 1.5x, a volatility
budget scales down, a kill-switch halves. So two things are confounded:

1. **Drawdown falls because you hold less.** The same artefact caught in the RQ1 overlay:
   a rule that cuts exposure always shows a shallower drawdown.
2. **Calmar is not scale-invariant.** Under compounding, leverage k gives roughly
   k*mu - k^2*sigma^2/2 in log terms, so *less* leverage means less volatility drag and a
   better CAGR-to-drawdown ratio. "Capped beats uncapped on Calmar" is close to arithmetic.

The fix is a **matched-risk benchmark**: a constant leverage scaled so its realised
volatility equals the limited strategy's. If a limit adds anything beyond holding less, it
must beat a dumb constant exposure carrying exactly the same risk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANN = np.sqrt(252.0)


def _stats(r: np.ndarray) -> dict:
    """Performance of a daily return stream. CAGR is geometric, so drag is included."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 100:
        return {k: np.nan for k in
                ("cagr", "vol", "sharpe", "max_drawdown", "calmar", "dd_per_vol")}
    eq = np.cumprod(1.0 + r)
    years = len(r) / 252.0
    cagr = eq[-1] ** (1.0 / years) - 1.0 if eq[-1] > 0 else -1.0
    vol = r.std(ddof=1) * ANN
    dd = float((eq / np.maximum.accumulate(eq) - 1.0).min())
    return {
        "cagr": float(cagr), "vol": float(vol),
        "sharpe": float(r.mean() / r.std(ddof=1) * ANN) if r.std(ddof=1) > 0 else np.nan,
        "max_drawdown": dd,
        "calmar": float(cagr / abs(dd)) if dd < 0 else np.nan,
        "dd_per_vol": float(dd / vol) if vol > 0 else np.nan,
    }


def _apply_limits(ret: np.ndarray, rv: np.ndarray, base_lev: float, kind: str,
                  cap: float = 1.5, vol_budget: float = 0.175,
                  dd_trigger: float = -0.20, cost_bps: float = 2.0) -> np.ndarray:
    """Exposure path for one limit, applied to tomorrow's return. Causal throughout."""
    n = len(ret)
    w = np.full(n, base_lev)
    if kind == "leverage_cap":
        w = np.minimum(w, cap)
    elif kind == "vol_budget":
        w = np.minimum(w, vol_budget / np.maximum(rv, 1e-6))
    elif kind == "kill_switch":
        # reactive: halve after a trailing drawdown breach. Path-dependent, so stepped.
        eq, peak, w_out = 1.0, 1.0, np.empty(n)
        for t in range(n):
            w_out[t] = base_lev * (0.5 if (eq / peak - 1.0) <= dd_trigger else 1.0)
            if t + 1 < n and np.isfinite(ret[t + 1]):
                eq *= 1.0 + w_out[t] * ret[t + 1]
                peak = max(peak, eq)
        w = w_out
    fwd = np.r_[ret[1:], np.nan]
    turn = np.abs(np.diff(np.r_[0.0, w]))
    net = w * fwd - turn * cost_bps / 1e4
    return net[np.isfinite(fwd)]


def _matched_constant(ret: np.ndarray, target_vol: float, cost_bps: float = 2.0) -> np.ndarray:
    """Constant leverage carrying the same realised volatility as the limited strategy.

    This is the benchmark the original lacked. Constant exposure has no risk management at
    all, so anything a limit is worth must show up as beating this.
    """
    fwd = np.r_[ret[1:], np.nan]
    ok = np.isfinite(fwd)
    base_vol = fwd[ok].std(ddof=1) * ANN
    m = target_vol / base_vol if base_vol > 0 else 0.0
    out = m * fwd[ok]
    # same cost model as the limited strategies: pay to get on at day one, nothing after.
    # Without this the benchmark is free and a one-day fee reads as underperformance.
    out[0] -= m * cost_bps / 1e4
    return out


def _block_bootstrap_stat_diff(a: np.ndarray, b: np.ndarray, stat: str = "sharpe",
                               block: int = 21, n_boot: int = 2000,
                               seed: int = 0) -> tuple[float, float]:
    """p-value for stat(a) - stat(b) under a stationary block bootstrap.

    Daily returns are serially dependent in volatility, so an i.i.d. bootstrap would
    understate the standard error and manufacture significance — the same reason the
    forecasting tests use Newey-West.

    `stat="calmar"` matters here: Sharpe is a mean-variance measure and cannot see a limit
    that reshapes the *tail* without changing mean-variance efficiency, which is exactly
    what a drawdown-control rule is supposed to do.
    """
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    def _f(x):
        if stat == "sharpe":
            sd = x.std(ddof=1)
            return x.mean() / sd * ANN if sd > 0 else np.nan
        st = _stats(x)
        return st["calmar"]

    obs = _f(a) - _f(b)
    # Degenerate case: a leverage cap IS constant leverage, so the two series can be
    # numerically identical. The bootstrap then has zero variance and float noise reads as
    # overwhelming significance. There is nothing to test — report it as such.
    if np.allclose(a, b, atol=1e-12) or abs(obs) < 1e-9:
        return float(obs), 1.0

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, max(n - block, 1), size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        idx = idx[idx < n]
        da, db = a[idx], b[idx]
        diffs[i] = _f(da) - _f(db)
    # centre on the bootstrap mean: test whether the observed gap is bigger than resampling noise
    # A limit that reduces to constant leverage produces two near-identical series, so the
    # resampling distribution collapses and float noise would read as overwhelming
    # significance. No spread means nothing to test.
    if diffs.std() < 1e-8:
        return float(obs), 1.0
    p = float(np.mean(np.abs(diffs - diffs.mean()) >= abs(obs)))
    return float(obs), p


def h8_retest(features: pd.DataFrame, base_lev: float = 2.0, cost_bps: float = 2.0,
              seed: int = 0) -> pd.DataFrame:
    """Each hard limit against both the original benchmark and a matched-risk one."""
    ret = features["ret"].astype(float).to_numpy()
    rv = features["rv20"].astype(float).to_numpy()
    ok = np.isfinite(ret) & np.isfinite(rv)
    ret, rv = ret[ok], rv[ok]

    unconstrained = _apply_limits(ret, rv, base_lev, "none", cost_bps=cost_bps)
    rows = []
    for kind in ("leverage_cap", "vol_budget", "kill_switch"):
        lim = _apply_limits(ret, rv, base_lev, kind, cost_bps=cost_bps)
        s_lim = _stats(lim)
        matched = _matched_constant(ret, s_lim["vol"], cost_bps)
        s_unc, s_mat = _stats(unconstrained), _stats(matched)
        d_unc, p_unc = _block_bootstrap_stat_diff(lim, unconstrained, "sharpe", seed=seed)
        d_mat, p_mat = _block_bootstrap_stat_diff(lim, matched, "sharpe", seed=seed)
        c_mat, pc_mat = _block_bootstrap_stat_diff(lim, matched, "calmar", seed=seed)

        rows.append({
            "limit": kind,
            "vol_limited": s_lim["vol"], "vol_matched": s_mat["vol"],
            "sharpe_limited": s_lim["sharpe"], "sharpe_unconstrained": s_unc["sharpe"],
            "sharpe_matched": s_mat["sharpe"],
            "dd_limited": s_lim["max_drawdown"], "dd_unconstrained": s_unc["max_drawdown"],
            "dd_matched": s_mat["max_drawdown"],
            "calmar_limited": s_lim["calmar"], "calmar_unconstrained": s_unc["calmar"],
            "calmar_matched": s_mat["calmar"],
            "sharpe_gap_vs_unconstrained": d_unc, "p_vs_unconstrained": p_unc,
            "sharpe_gap_vs_matched": d_mat, "p_vs_matched": p_mat,
            "calmar_gap_vs_matched": c_mat, "p_calmar_vs_matched": pc_mat,
        })
    return pd.DataFrame(rows)
