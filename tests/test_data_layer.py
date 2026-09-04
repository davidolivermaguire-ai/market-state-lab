"""Step 1: intervals, the symbol registry, and the pre-flight audit."""
import numpy as np
import pandas as pd
import pytest

from msl.data.calendar import (DAILY, WEEKLY, get_interval, interval_of,
                               require_daily, resample, stamp)
from msl.data import symbols as sym


def _daily(n=600, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n, name="date")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99,
                         "Close": c, "Volume": rng.integers(1e6, 5e6, n)}, index=idx)


def test_weekly_bar_never_contains_future_days():
    """The causality guarantee for coarse bars: a bar's content ends at or before its label."""
    d = _daily()
    w = resample(d, WEEKLY)
    for label in w.index:
        window = d[(d.index > label - pd.Timedelta(days=7)) & (d.index <= label)]
        assert len(window) > 0
        # the weekly close must be the last daily close at or before the label
        assert np.isclose(w.loc[label, "Close"], window["Close"].iloc[-1])
        assert np.isclose(w.loc[label, "High"], window["High"].max())
        assert np.isclose(w.loc[label, "Low"], window["Low"].min())


def test_partial_trailing_bar_is_flagged_not_hidden():
    """A trailing label can post-date the data. Safe for causality, but must be visible."""
    d = _daily()
    w = resample(d, WEEKLY)
    assert w.attrs["data_end"] == d.index.max()
    if w.index.max() > d.index.max():
        assert w.attrs["last_bar_partial"] is True


def test_annualisation_follows_the_interval():
    """The bug this exists to prevent: sqrt(252) applied to weekly bars."""
    from msl.features.core import build_features
    d = _daily()
    fd = build_features(stamp(d.copy(), DAILY))
    fw = build_features(stamp(resample(d, WEEKLY), WEEKLY))
    assert fd.attrs["periods_per_year"] == 252.0
    assert fw.attrs["periods_per_year"] == 52.0
    assert np.isclose(DAILY.ann / WEEKLY.ann, np.sqrt(252 / 52))
    # a weekly rv built with the daily factor would be ~2.2x too big
    assert fw["rv20"].dropna().median() < fd["rv20"].dropna().median() * DAILY.ann / WEEKLY.ann


def test_daily_frames_are_unchanged_by_the_interval_work():
    """Regression: the default path must behave exactly as before."""
    d = _daily()
    assert resample(d, DAILY) is d
    assert interval_of(pd.DataFrame()) is DAILY          # un-stamped defaults to daily
    assert get_interval("daily") is DAILY and get_interval("1wk") is WEEKLY


def test_require_daily_fails_loudly_on_weekly():
    """Metrics that still hard-code 252 must refuse weekly data, not mis-annualise it."""
    w = stamp(resample(_daily(), WEEKLY), WEEKLY)
    require_daily(stamp(_daily(), DAILY), "test")        # daily passes
    with pytest.raises(NotImplementedError, match="daily calendar"):
        require_daily(w, "risk metrics")


def test_unknown_interval_is_rejected():
    with pytest.raises(ValueError, match="unknown interval"):
        get_interval("4h")


def test_registry_covers_every_universe_member():
    """The gap that made AAPL work only by accident of vendor pass-through."""
    for name in sym.UNIVERSES:
        assert sym.unregistered(name) == [], f"{name} has unregistered symbols"


def test_registry_resolves_and_describes():
    assert sym.resolve("NAS100") == "^NDX"
    assert sym.resolve("UNKNOWNTICKER") == "UNKNOWNTICKER"   # pass-through still works
    assert not sym.is_registered("UNKNOWNTICKER")
    assert sym.describe("VIX").asset_class == "volatility"
    assert len(sym.catalogue()) == len(sym.REGISTRY)


def test_search_space_counts_before_the_run():
    from msl.data.audit import search_space
    assert search_space(7, 1, 8, 3)["comparisons"] == 168
    assert np.isclose(search_space(7, 1, 8, 3)["deflated_bar"], np.sqrt(2 * np.log(168)))
    # adding an interval raises the bar — the whole point of counting up front
    assert search_space(7, 2, 8, 3)["deflated_bar"] > search_space(7, 1, 8, 3)["deflated_bar"]
