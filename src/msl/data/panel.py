"""The cross-asset panel: many symbols on one aligned clock.

`load_universe` returns independent frames, which is fine for per-asset estimators and
useless for anything that needs assets *together* — correlation, dispersion, absorption,
DCC. Those are the Step 2 specialists that cover dimensions trend estimators cannot, so
the panel is what unblocks them.

Two design decisions carry all the risk.

**Never forward-fill.** Assets trade on different calendars, and the obvious fix — reindex
to a union and pad — creates a *fake zero return* on every padded day. Zeros bias realised
volatility down, bias pairwise correlation down, and flatter every dependence model
downstream, all while the frame looks complete and healthy. This module refuses to do it;
`how="union"` leaves NaN so a caller must decide knowingly.

**Alignment is itself a choice with a cost.** Intersecting calendars discards days, and if
missingness correlates with market events — a halt, a local holiday during a selloff — the
discarding is not random. The panel therefore *reports* what alignment cost rather than
performing it silently.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from msl.data.calendar import Interval, get_interval
from msl.data.loaders import load_universe
from msl.data.symbols import expand

HOW = ("intersect", "union")


@dataclass
class Panel:
    """Aligned multi-asset data plus the record of what alignment cost."""

    prices: dict[str, pd.DataFrame]     # per symbol, on the shared index
    returns: pd.DataFrame               # T x N log returns — what dependence models want
    interval: Interval
    how: str
    coverage: pd.DataFrame              # per symbol: own bars, kept, dropped
    notes: list[str] = field(default_factory=list)

    @property
    def symbols(self) -> list[str]:
        return list(self.returns.columns)

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.returns.index

    def __len__(self) -> int:
        return len(self.returns)

    def __repr__(self) -> str:
        span = f"{self.index.min().date()}..{self.index.max().date()}" if len(self) else "empty"
        return (f"Panel({len(self.symbols)} symbols, {len(self)} {self.interval.label} bars, "
                f"{span}, how={self.how!r})")


def load_panel(name_or_symbols: str | list[str], start: str | None = None,
               end: str | None = None, interval: str | Interval | None = None,
               how: str = "intersect", allow_download: bool = True,
               prefer_fresh: bool = False, min_coverage: float = 0.90) -> Panel:
    """Load several symbols onto one clock.

    `how="intersect"` (default) keeps only bars every symbol has — the honest choice for
    dependence estimates, because every observation is genuinely simultaneous.
    `how="union"` keeps every date any symbol has and leaves NaN where a symbol did not
    trade. Nothing is ever forward-filled.

    `min_coverage` flags a symbol that would drag the intersection down: if including it
    costs more than this fraction of the other symbols' bars, that is worth knowing before
    a panel silently shrinks.
    """
    if how not in HOW:
        raise ValueError(f"how must be one of {HOW}, got {how!r}")
    iv = get_interval(interval)
    names = expand(name_or_symbols)
    prices = load_universe(names, start, end, allow_download=allow_download,
                           prefer_fresh=prefer_fresh, interval=iv)
    if not prices:
        raise RuntimeError("no symbols loaded")

    own = {s: df.index for s, df in prices.items()}
    if how == "intersect":
        idx = None
        for i in own.values():
            idx = i if idx is None else idx.intersection(i)
    else:
        idx = None
        for i in own.values():
            idx = i if idx is None else idx.union(i)
    idx = pd.DatetimeIndex(sorted(idx))

    notes: list[str] = []
    aligned = {s: df.reindex(idx) for s, df in prices.items()}   # reindex only: no padding
    ret = pd.DataFrame(
        {s: np.log(df["Close"].astype(float)).diff() for s, df in aligned.items()},
        index=idx)

    rows = []
    for s in prices:
        kept = len(own[s].intersection(idx))
        rows.append({"symbol": s, "own_bars": len(own[s]), "in_panel": kept,
                     "dropped": len(own[s]) - kept,
                     "missing_in_panel": int(len(idx) - kept)})
    coverage = pd.DataFrame(rows)

    if how == "intersect":
        widest = coverage["own_bars"].max()
        if len(idx) < widest:
            lost = widest - len(idx)
            notes.append(
                f"alignment cost {lost} bars ({lost / widest:.1%} of the longest symbol) — "
                f"intersecting calendars discards any date not shared by every symbol.")
        thin = coverage[coverage["own_bars"] < min_coverage * widest]
        for _, r in thin.iterrows():
            notes.append(
                f"{r['symbol']} has {r['own_bars']} of {widest} bars "
                f"({r['own_bars'] / widest:.1%}) and is pulling the intersection down.")
    else:
        gaps = ret.isna().sum()
        for s, n in gaps.items():
            if n > len(idx) * 0.02:
                notes.append(f"{s} is NaN on {n} of {len(idx)} panel dates "
                             f"({n / len(idx):.1%}) — union keeps them rather than padding.")

    notes.append("no forward-fill: a padded price would produce a fake zero return and "
                 "bias correlation and volatility downward.")
    return Panel(aligned, ret, iv, how, coverage, notes)


# ------------------------------------------------------------------ cross-asset features
def average_correlation(returns: pd.DataFrame, window: int = 60,
                        min_periods: int | None = None) -> pd.Series:
    """Mean pairwise correlation on a trailing window — the classic co-movement gauge.

    Rises toward 1 when everything sells off together, which is exactly the condition a
    per-asset trend estimator cannot see. Causal: window ends at t.
    """
    mp = min_periods or max(window // 2, 20)
    n = returns.shape[1]
    if n < 2:
        return pd.Series(np.nan, index=returns.index, name="avg_corr")
    out = (returns.rolling(window, min_periods=mp)
           .corr(pairwise=True)
           .groupby(level=0)
           .apply(lambda m: (m.to_numpy().sum() - n) / (n * (n - 1))))
    return pd.Series(out.to_numpy(), index=returns.index[-len(out):], name="avg_corr") \
        .reindex(returns.index)


def dispersion(returns: pd.DataFrame) -> pd.Series:
    """Cross-sectional spread of returns on each date. High = idiosyncratic, low = macro."""
    return returns.std(axis=1, ddof=1).rename("dispersion")


def absorption_ratio(returns: pd.DataFrame, window: int = 252, n_components: int = 1,
                     min_periods: int | None = None) -> pd.Series:
    """Fraction of panel variance explained by the leading eigenvectors (Kritzman et al.).

    A high absorption ratio means the assets have collapsed onto a few common factors —
    a systemic-fragility signature that no single-asset feature can express. Computed on
    a trailing window only, so it is causal by construction.
    """
    mp = min_periods or window
    vals = np.full(len(returns), np.nan)
    R = returns.to_numpy(dtype=float)
    k = max(1, min(int(n_components), returns.shape[1]))
    for t in range(len(returns)):
        lo = max(0, t - window + 1)
        w = R[lo:t + 1]
        w = w[~np.isnan(w).any(axis=1)]
        if len(w) < mp or w.shape[1] < 2:
            continue
        C = np.corrcoef(w, rowvar=False)
        if not np.isfinite(C).all():
            continue
        ev = np.sort(np.linalg.eigvalsh(C))[::-1]
        tot = ev.sum()
        if tot > 0:
            vals[t] = float(ev[:k].sum() / tot)
    return pd.Series(vals, index=returns.index, name="absorption")


def cross_asset_features(panel: Panel, corr_window: int = 60,
                         absorption_window: int = 252) -> pd.DataFrame:
    """The features that only exist because there is more than one asset.

    These are the inputs a cross-asset specialist needs and a per-symbol feature frame
    structurally cannot provide — which is why every estimator built so far measures the
    same dimension.
    """
    r = panel.returns
    f = pd.DataFrame(index=r.index)
    f["avg_corr"] = average_correlation(r, corr_window)
    f["dispersion"] = dispersion(r)
    f["absorption"] = absorption_ratio(r, absorption_window)
    f["breadth"] = (r > 0).sum(axis=1) / r.notna().sum(axis=1).replace(0, np.nan)
    f.attrs["interval"] = panel.interval.key
    f.attrs["periods_per_year"] = panel.interval.periods_per_year
    return f
