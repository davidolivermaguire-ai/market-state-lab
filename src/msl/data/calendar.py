"""Bar intervals and their calendars.

Annualisation is not a constant. `sqrt(252)` is correct for daily bars and wrong for
every other frequency, and it was hard-coded in eight places before intervals existed.
Adding weekly data without threading the calendar through would have made every
volatility, Sharpe and CAGR silently wrong by a factor of sqrt(5) — the same class of
invisible error the rest of this project exists to catch.

So an interval is a small object carrying everything downstream code needs to be
correct, and frames are stamped with it (``df.attrs["interval"]``). Code that has not
been made interval-aware can call :func:`require_daily` and fail loudly rather than
quietly producing numbers that are off by a constant.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Interval:
    """A bar frequency and the calendar facts that depend on it."""

    key: str                 # what a config or CLI writes: "1d", "1wk"
    label: str               # human-readable
    periods_per_year: float  # THE number every annualisation must use
    resample_rule: str | None  # pandas rule to build it from daily; None = native
    min_bars_hint: int       # rough sanity floor for a usable series

    @property
    def ann(self) -> float:
        """Volatility annualisation factor for this interval."""
        return float(self.periods_per_year) ** 0.5


DAILY = Interval("1d", "daily", 252.0, None, 500)
WEEKLY = Interval("1wk", "weekly", 52.0, "W-FRI", 150)

INTERVALS: dict[str, Interval] = {i.key: i for i in (DAILY, WEEKLY)}
# tolerated spellings, so a config saying "daily" or "weekly" still works
_ALIASES = {"d": "1d", "day": "1d", "daily": "1d",
            "w": "1wk", "wk": "1wk", "week": "1wk", "weekly": "1wk"}


def get_interval(key: str | Interval | None) -> Interval:
    """Look up an interval by key or alias. ``None`` means daily."""
    if key is None:
        return DAILY
    if isinstance(key, Interval):
        return key
    k = str(key).strip().lower()
    k = _ALIASES.get(k, k)
    if k not in INTERVALS:
        raise ValueError(
            f"unknown interval {key!r}. Known: {', '.join(INTERVALS)} "
            f"(aliases: {', '.join(sorted(_ALIASES))})"
        )
    return INTERVALS[k]


OHLCV_AGG = {"Open": "first", "High": "max", "Low": "min",
             "Close": "last", "Volume": "sum"}


def resample(df: pd.DataFrame, interval: Interval) -> pd.DataFrame:
    """Aggregate daily OHLCV to a coarser interval, right-labelled.

    Right-labelling matters for causality: a weekly bar stamped Friday must contain
    only that week's days, so a model reading it on Friday is not seeing the future.
    Partial trailing weeks are kept — dropping them would silently move the as-of date.
    """
    if interval.resample_rule is None:
        return df
    out = df.resample(interval.resample_rule, label="right", closed="right").agg(OHLCV_AGG)
    out = out.dropna(subset=["Close"])
    # A trailing partial bar is labelled at the period end, which can sit days *after*
    # the last real observation. That direction is safe for causality — the stamp is
    # later than its content, never earlier — but it overstates the as-of date, so
    # record the true end for anything that reports freshness.
    if len(out):
        out.attrs["data_end"] = df.index.max()
        out.attrs["last_bar_partial"] = bool(out.index.max() > df.index.max())
    return out


def stamp(df: pd.DataFrame, interval: Interval) -> pd.DataFrame:
    """Record the interval on the frame so downstream code can annualise correctly."""
    df.attrs["interval"] = interval.key
    df.attrs["periods_per_year"] = interval.periods_per_year
    df.attrs.setdefault("data_end", df.index.max() if len(df) else None)
    df.attrs.setdefault("last_bar_partial", False)
    return df


def interval_of(df: pd.DataFrame) -> Interval:
    """Read the interval off a frame, defaulting to daily for un-stamped frames."""
    return get_interval(df.attrs.get("interval"))


def periods_per_year(df: pd.DataFrame) -> float:
    return float(df.attrs.get("periods_per_year", DAILY.periods_per_year))


def require_daily(df: pd.DataFrame, who: str) -> None:
    """Guard for code that still hard-codes a daily calendar.

    Several metrics modules assume 252 internally. Until they are made interval-aware,
    handing them a weekly frame would produce numbers that look plausible and are wrong.
    Fail loudly instead.
    """
    iv = interval_of(df)
    if iv is not DAILY:
        raise NotImplementedError(
            f"{who} still assumes a daily calendar, but was given {iv.label} data "
            f"({iv.periods_per_year} periods/year). Annualisation would be wrong by "
            f"sqrt({DAILY.periods_per_year / iv.periods_per_year:.0f}). Make {who} "
            f"interval-aware before using it on {iv.label} bars."
        )
