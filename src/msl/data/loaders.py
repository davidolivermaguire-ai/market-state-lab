"""Price loading: committed CSV first, then cache, then network.

Resolution order is deliberate and reproducibility-first:

  1. ``data/raw/<SYMBOL>.csv``  - committed alongside published results, so anyone
     can reproduce them with no network and no vendor account.
  2. ``data/cache/<SYMBOL>.parquet`` - local cache of a previous download.
  3. yfinance download (optional extra), which is then written to the cache.

Every frame comes back with the same columns and a sorted DatetimeIndex, so the
rest of the package never knows or cares where the data came from.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from msl.data.symbols import UNIVERSES, resolve

COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _repo_root() -> Path:
    env = os.environ.get("MSL_DATA_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3]


def _raw_dir() -> Path:
    return _repo_root() / "data" / "raw"


def _cache_dir() -> Path:
    return _repo_root() / "data" / "cache"


def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], utc=False, errors="coerce")
        df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"price frame missing columns: {missing}")
    df = df[COLUMNS].apply(pd.to_numeric, errors="coerce")
    return df.dropna(subset=["Close"]).sort_index()


def load_prices(
    symbol: str,
    start: str | None = None,
    end: str | None = None,
    allow_download: bool = True,
    prefer_fresh: bool = False,
) -> pd.DataFrame:
    """Daily OHLCV for a friendly symbol name, as a sorted, tidy frame.

    `prefer_fresh=True` skips the committed CSV and downloads. Use it when symbols must
    share a common as-of date: a committed CSV is frozen at the date it was published,
    so mixing it with freshly-downloaded symbols silently compares states at different
    points in time.
    """
    key = symbol.upper()

    csv = _raw_dir() / f"{key}.csv"
    if prefer_fresh and allow_download:
        csv = Path("__skip__")
    if csv.exists():
        df = _tidy(pd.read_csv(csv))
    else:
        pq = _cache_dir() / f"{key}.parquet"
        cached = None
        if pq.exists():
            try:
                cached = _tidy(pd.read_parquet(pq))
            except Exception:
                # An unreadable cache (e.g. no parquet engine installed here) is a
                # reason to re-fetch, not a reason to lose the symbol.
                cached = None
        if cached is not None:
            df = cached
        elif allow_download:
            df = _tidy(_download(key, start, end))
            _cache_dir().mkdir(parents=True, exist_ok=True)
            try:
                df.to_parquet(pq)
            except Exception:  # pyarrow missing - cache is a nicety, not a requirement
                df.to_csv(_cache_dir() / f"{key}.csv")
        else:
            raise FileNotFoundError(
                f"no committed CSV or cache for {key}, and downloads are disabled. "
                f"Put a CSV at {csv} or call with allow_download=True."
            )

    if start is not None:
        df = df[df.index >= pd.Timestamp(start)]
    if end is not None:
        df = df[df.index <= pd.Timestamp(end)]
    if df.empty:
        raise ValueError(f"no rows for {key} in the requested window")
    return df


def _download(symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install the data extra for downloads: pip install -e '.[data]'") from exc

    raw = yf.download(
        resolve(symbol), start=start, end=end, auto_adjust=False, progress=False, threads=False
    )
    if raw is None or len(raw) == 0:
        raise ValueError(f"download returned no rows for {symbol}")
    if isinstance(raw.columns, pd.MultiIndex):  # yfinance returns MultiIndex for 1+ tickers
        raw.columns = raw.columns.get_level_values(0)
    return raw


def load_universe(
    name_or_symbols: str | list[str],
    start: str | None = None,
    end: str | None = None,
    allow_download: bool = True,
    prefer_fresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load a named universe (or an explicit list) as {symbol: prices}.

    Symbols that fail to load are skipped with a warning rather than killing a sweep —
    a missing ticker should not cost you the other nine.
    """
    symbols = UNIVERSES[name_or_symbols] if isinstance(name_or_symbols, str) else list(name_or_symbols)
    out: dict[str, pd.DataFrame] = {}
    for s in symbols:
        try:
            out[s] = load_prices(s, start, end, allow_download=allow_download,
                                 prefer_fresh=prefer_fresh)
        except Exception as exc:
            print(f"  [warn] skipping {s}: {exc}")
    if not out:
        raise RuntimeError("no symbols loaded")
    return out
