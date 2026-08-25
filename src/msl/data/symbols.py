"""Friendly symbol names -> vendor tickers, and named universes.

The point of the map is that experiment configs and published results refer to
NAS100 / US500 / US30, not to whatever ticker a particular vendor happens to use.
Swapping data provider later means editing this file, not the experiments.
"""
from __future__ import annotations

SYMBOL_MAP: dict[str, str] = {
    # index proxies (price indices)
    "NAS100": "^NDX",
    "US500": "^GSPC",
    "US30": "^DJI",
    "RUSSELL2000": "^RUT",
    "VIX": "^VIX",
    # liquid ETFs — total-return-ish proxies, useful when a tradable series is wanted
    "QQQ": "QQQ",
    "SPY": "SPY",
    "DIA": "DIA",
}

# Fixed, pre-declared single-name lists. Declared once and never edited to match
# today's index membership — that is what keeps the panel free of survivorship bias.
UNIVERSES: dict[str, list[str]] = {
    "indices": ["NAS100", "US500", "US30"],
    "indices_wide": ["NAS100", "US500", "US30", "RUSSELL2000"],
    "megacap_2015": ["AAPL", "MSFT", "JNJ", "XOM", "GE", "WMT", "JPM", "PG"],
    "mixed": ["NAS100", "US500", "US30", "AAPL", "MSFT", "XOM", "JPM"],
}


def resolve(symbol: str) -> str:
    """Friendly name -> vendor ticker. Unknown names pass through (any ticker works)."""
    return SYMBOL_MAP.get(symbol.upper(), symbol.upper())
