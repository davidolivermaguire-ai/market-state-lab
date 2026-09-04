"""The symbol registry: friendly names -> vendor tickers, with what each one *is*.

Configs and published results refer to NAS100 / US500 / US30, not to whatever ticker a
particular vendor happens to use, so swapping data provider means editing this file
rather than every experiment.

Previously this was a bare dict of eight entries, and universes referenced names that
were not in it — AAPL and JPM worked only because unknown names pass straight through
to the vendor. That is convenient and invisible, which is a bad combination: nothing
could answer "what can I ask for, and what will I get?". Every symbol used anywhere is
now declared with its asset class and a one-line description, and `unregistered()`
reports anything still relying on pass-through.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Symbol:
    name: str          # the friendly name experiments use
    ticker: str        # what the vendor wants
    asset_class: str   # index | etf | equity | volatility
    description: str
    note: str = ""     # anything a reader should know before using it


_S = [
    # ---- price indices. No dividends: fine for state identification, not for
    # total-return claims. Stated here so it cannot be forgotten downstream.
    Symbol("NAS100", "^NDX", "index", "Nasdaq-100 price index", "price index, no dividends"),
    Symbol("US500", "^GSPC", "index", "S&P 500 price index", "price index, no dividends"),
    Symbol("US30", "^DJI", "index", "Dow Jones Industrial Average", "price-weighted, no dividends"),
    Symbol("RUSSELL2000", "^RUT", "index", "Russell 2000 small-cap index", "price index"),
    Symbol("VIX", "^VIX", "volatility", "CBOE volatility index",
           "an index level, not a return series — do not treat as an asset"),

    # ---- ETFs: tradable, dividend-adjusted when auto_adjust is on
    Symbol("QQQ", "QQQ", "etf", "Invesco QQQ Trust (Nasdaq-100)"),
    Symbol("SPY", "SPY", "etf", "SPDR S&P 500 ETF"),
    Symbol("DIA", "DIA", "etf", "SPDR Dow Jones Industrial Average ETF"),

    # ---- single names used by the pre-declared universes. Fixed lists, never edited
    # to match today's index membership — that is what keeps the panel survivorship-free.
    Symbol("AAPL", "AAPL", "equity", "Apple Inc."),
    Symbol("MSFT", "MSFT", "equity", "Microsoft Corp."),
    Symbol("JPM", "JPM", "equity", "JPMorgan Chase & Co."),
    Symbol("XOM", "XOM", "equity", "Exxon Mobil Corp."),
    Symbol("JNJ", "JNJ", "equity", "Johnson & Johnson"),
    Symbol("GE", "GE", "equity", "General Electric", "restructured 2021-24; a broken series in places"),
    Symbol("WMT", "WMT", "equity", "Walmart Inc."),
    Symbol("PG", "PG", "equity", "Procter & Gamble Co."),
]

REGISTRY: dict[str, Symbol] = {s.name: s for s in _S}
SYMBOL_MAP: dict[str, str] = {s.name: s.ticker for s in _S}   # kept for compatibility

UNIVERSES: dict[str, list[str]] = {
    "indices": ["NAS100", "US500", "US30"],
    "indices_wide": ["NAS100", "US500", "US30", "RUSSELL2000"],
    "megacap_2015": ["AAPL", "MSFT", "JNJ", "XOM", "GE", "WMT", "JPM", "PG"],
    "mixed": ["NAS100", "US500", "US30", "AAPL", "MSFT", "XOM", "JPM"],
}


def resolve(symbol: str) -> str:
    """Friendly name -> vendor ticker. Unknown names pass through unchanged."""
    key = symbol.upper()
    s = REGISTRY.get(key)
    return s.ticker if s else key


def is_registered(symbol: str) -> bool:
    return symbol.upper() in REGISTRY


def describe(symbol: str) -> Symbol | None:
    return REGISTRY.get(symbol.upper())


def expand(name_or_symbols: str | list[str]) -> list[str]:
    """A universe name or an explicit list -> a list of symbol names."""
    if isinstance(name_or_symbols, str):
        if name_or_symbols not in UNIVERSES:
            raise KeyError(
                f"unknown universe {name_or_symbols!r}. Known: {', '.join(UNIVERSES)}. "
                f"To use an explicit list, pass a list rather than a string."
            )
        return list(UNIVERSES[name_or_symbols])
    return [str(s).upper() for s in name_or_symbols]


def unregistered(name_or_symbols: str | list[str]) -> list[str]:
    """Symbols that would rely on vendor pass-through rather than the registry."""
    return [s for s in expand(name_or_symbols) if not is_registered(s)]


def catalogue() -> pd.DataFrame:
    """Everything the registry knows, as a frame — the answer to 'what can I ask for?'."""
    rows = [{"symbol": s.name, "ticker": s.ticker, "asset_class": s.asset_class,
             "description": s.description, "note": s.note,
             "universes": ", ".join(u for u, m in UNIVERSES.items() if s.name in m)}
            for s in REGISTRY.values()]
    return pd.DataFrame(rows).sort_values(["asset_class", "symbol"]).reset_index(drop=True)
