"""Pre-flight visibility for the data layer.

Three published-number problems in this project came from not knowing what had actually
been loaded:

* a committed CSV frozen at 2026-07-06 sitting beside a cache holding 2026-08-24, with
  nothing saying which a run had used;
* a panel that ran 2010–2026 when the write-up said 2015–2026, caught only by noticing
  the row count;
* symbols carrying different as-of dates, so states from different days were compared.

Each was discovered *after* the numbers were published. This module answers, before a
run: what resolves to what, where it came from, how stale it is, what the span and bar
count are, where the gaps are, and whether the panel is aligned.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from msl.data.calendar import Interval, get_interval
from msl.data.loaders import LAST_SOURCE, load_prices
from msl.data.symbols import describe, expand, is_registered, resolve

STALE_DAYS = 7          # a symbol this far behind the panel leader is flagged
ALIGN_TOLERANCE = 5     # as-of spread beyond this makes a panel non-comparable


def _gaps(idx: pd.DatetimeIndex, interval: Interval) -> tuple[int, str]:
    """Count unusually long holes. Weekends and holidays are normal; a month is not."""
    if len(idx) < 3:
        return 0, ""
    d = pd.Series(idx).diff().dt.days.dropna()
    typical = max(float(d.median()), 1.0)
    big = d[d > typical * 5]
    if big.empty:
        return 0, ""
    worst = int(big.max())
    at = idx[int(big.idxmax())].date()
    return len(big), f"{len(big)} gap(s), worst {worst}d before {at}"


def audit(symbols: str | list[str], start: str | None = None, end: str | None = None,
          interval: str | Interval | None = None, allow_download: bool = False,
          prefer_fresh: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """Load each symbol and report what arrived. Returns (report, warnings)."""
    iv = get_interval(interval)
    names = expand(symbols)
    rows, warnings = [], []

    unreg = [s for s in names if not is_registered(s)]
    if unreg:
        warnings.append(
            f"not in the symbol registry, relying on vendor pass-through: {', '.join(unreg)}. "
            f"Add them to msl/data/symbols.py so what they are is recorded.")

    for s in names:
        meta = describe(s)
        try:
            df = load_prices(s, start, end, allow_download=allow_download,
                             prefer_fresh=prefer_fresh, interval=iv)
        except Exception as exc:
            rows.append({"symbol": s, "ticker": resolve(s), "status": f"FAILED: {exc}"})
            warnings.append(f"{s} failed to load: {exc}")
            continue
        n_gaps, gap_note = _gaps(df.index, iv)
        # For coarse bars the final label can post-date the last real observation.
        # Freshness must be judged on the data, not on the label.
        data_end = df.attrs.get("data_end") or df.index.max()
        rows.append({
            "symbol": s, "ticker": resolve(s),
            "class": meta.asset_class if meta else "unregistered",
            "source": LAST_SOURCE.get(s.upper(), "?"),
            "bars": len(df),
            "first": df.index.min().date(), "last": pd.Timestamp(data_end).date(),
            "partial": "yes" if df.attrs.get("last_bar_partial") else "",
            "years": round(len(df) / iv.periods_per_year, 1),
            "gaps": n_gaps, "gap_detail": gap_note,
            "status": "ok",
        })
        if meta and meta.note:
            warnings.append(f"{s}: {meta.note}")

    rep = pd.DataFrame(rows)
    ok = rep[rep.status == "ok"] if "status" in rep else rep
    if len(ok) > 1:
        last = pd.to_datetime(ok["last"])
        spread = (last.max() - last.min()).days
        if spread > ALIGN_TOLERANCE:
            behind = ok.loc[last < last.max() - pd.Timedelta(days=ALIGN_TOLERANCE), "symbol"]
            warnings.append(
                f"as-of dates span {spread} days — this panel is NOT like-for-like. "
                f"Behind: {', '.join(behind)}. Re-run with --refresh for a common date.")
        srcs = set(ok["source"])
        if len(srcs) > 1:
            warnings.append(
                f"mixed provenance {sorted(srcs)} — committed CSVs are frozen at their "
                f"publication date, so mixing them with downloads compares different days.")
        short = ok[ok["bars"] < iv.min_bars_hint]
        if not short.empty:
            warnings.append(
                f"fewer than {iv.min_bars_hint} {iv.label} bars: "
                f"{', '.join(short['symbol'])} — too short to fit and score reliably.")
    return rep, warnings


def search_space(n_symbols: int, n_intervals: int, n_methods: int, n_targets: int) -> dict:
    """The multiple-testing cost of a run, before it is run.

    Every dimension you add multiplies the number of chances to find something. This
    project has repeatedly shown that a bar set for the wrong N manufactures results,
    so the count belongs in the pre-flight report, not in a footnote afterwards.
    """
    n = max(int(n_symbols) * int(n_intervals) * int(n_methods) * int(n_targets), 1)
    return {"comparisons": n, "deflated_bar": float(np.sqrt(2.0 * np.log(n))) if n > 1 else 0.0}
