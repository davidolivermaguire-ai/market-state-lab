"""Step 1c: the cross-asset panel — alignment honesty and causal cross-asset features."""
import numpy as np
import pandas as pd
import pytest

from msl.data.calendar import DAILY, stamp
from msl.data.panel import (Panel, absorption_ratio, average_correlation,
                            cross_asset_features, dispersion)


def _px(n=400, seed=0, drop=()):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n, name="date")
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99,
                       "Close": c, "Volume": 1e6}, index=idx)
    if len(drop):
        df = df.drop(df.index[list(drop)])
    return stamp(df, DAILY)


def _panel(how="intersect", **kw):
    """Build a Panel directly so tests do not depend on the data cache."""
    from msl.data import panel as P
    prices = {"A": _px(seed=1), "B": _px(seed=2), "C": _px(seed=3, drop=(10, 11, 12))}
    idx = None
    for df in prices.values():
        idx = df.index if idx is None else (idx.intersection(df.index) if how == "intersect"
                                            else idx.union(df.index))
    aligned = {s: d.reindex(idx) for s, d in prices.items()}
    ret = pd.DataFrame({s: np.log(d["Close"]).diff() for s, d in aligned.items()}, index=idx)
    cov = pd.DataFrame([{"symbol": s, "own_bars": len(d), "in_panel": len(d.index.intersection(idx)),
                         "dropped": len(d) - len(d.index.intersection(idx)),
                         "missing_in_panel": len(idx) - len(d.index.intersection(idx))}
                        for s, d in prices.items()])
    return Panel(aligned, ret, DAILY, how, cov, [])


def test_intersection_drops_only_unshared_dates():
    p = _panel("intersect")
    assert len(p) == 397                      # 400 less the three C is missing
    assert p.returns.notna().all().all() or p.returns.iloc[1:].notna().all().all()
    assert set(p.symbols) == {"A", "B", "C"}


def test_union_leaves_nan_and_never_pads():
    """The trap this module exists to avoid: a padded price is a fake zero return."""
    p = _panel("union")
    assert len(p) == 400
    c = p.prices["C"]["Close"]
    assert c.isna().sum() == 3, "missing days must stay missing, not be forward-filled"
    # and the fake-zero signature must be absent
    r = p.returns["C"].dropna()
    assert (r == 0).sum() == 0


def test_load_panel_rejects_unknown_alignment():
    from msl.data.panel import load_panel
    with pytest.raises(ValueError, match="how must be one of"):
        load_panel(["NAS100"], how="ffill")


def test_cross_asset_features_are_causal():
    """Every cross-asset feature at t must be unchanged when future rows are appended."""
    p_full = _panel("intersect")
    cut = len(p_full) - 60
    f_full = cross_asset_features(p_full, corr_window=30, absorption_window=100)

    p_trunc = Panel({s: d.iloc[:cut] for s, d in p_full.prices.items()},
                    p_full.returns.iloc[:cut], DAILY, "intersect", p_full.coverage, [])
    f_trunc = cross_asset_features(p_trunc, corr_window=30, absorption_window=100)

    a, b = f_full.iloc[:cut], f_trunc
    both = a.notna() & b.notna()
    assert both.to_numpy().sum() > 100, "not enough overlap to test"
    assert np.allclose(a.to_numpy()[both.to_numpy()], b.to_numpy()[both.to_numpy()],
                       atol=1e-10), "a cross-asset feature used future data"


def test_absorption_is_a_fraction_and_rises_with_common_factor():
    """One shared factor should absorb more variance than independent noise."""
    rng = np.random.default_rng(7)
    n = 400
    idx = pd.bdate_range("2020-01-01", periods=n)
    indep = pd.DataFrame(rng.normal(0, 0.01, (n, 4)), index=idx)
    f = rng.normal(0, 0.01, n)
    common = pd.DataFrame({i: f + rng.normal(0, 0.002, n) for i in range(4)}, index=idx)

    a_i = absorption_ratio(indep, window=200).dropna()
    a_c = absorption_ratio(common, window=200).dropna()
    assert ((a_i >= 0) & (a_i <= 1)).all() and ((a_c >= 0) & (a_c <= 1)).all()
    assert a_c.mean() > a_i.mean() + 0.3, "common factor must absorb more variance"


def test_average_correlation_tracks_comovement():
    rng = np.random.default_rng(11)
    n = 300
    idx = pd.bdate_range("2020-01-01", periods=n)
    indep = pd.DataFrame(rng.normal(0, 0.01, (n, 3)), index=idx)
    f = rng.normal(0, 0.01, n)
    together = pd.DataFrame({i: f + rng.normal(0, 0.001, n) for i in range(3)}, index=idx)
    assert average_correlation(together, 60).dropna().mean() > \
           average_correlation(indep, 60).dropna().mean() + 0.5


def test_dispersion_is_cross_sectional_not_temporal():
    p = _panel()
    d = dispersion(p.returns)
    assert len(d) == len(p.returns)
    row = p.returns.iloc[100]
    assert np.isclose(d.iloc[100], row.std(ddof=1))


# ------------------------------------------------------- alignment loss is not one thing
def _breakdown(own):
    from msl.data.panel import _alignment_breakdown
    idx = None
    for i in own.values():
        idx = i if idx is None else idx.intersection(i)
    return _alignment_breakdown(own, pd.DatetimeIndex(sorted(idx)))


def test_stale_symbol_is_reported_as_staleness_not_calendar_disagreement():
    """The real NAS100 case: a frozen CSV truncates the panel and must say so.

    Calling this a calendar cost invites accepting it. It is one refresh away from zero.
    """
    full = pd.bdate_range("2020-01-01", periods=300)
    own = {"FRESH": full, "OTHER": full, "STALE": full[:265]}
    info, notes = _breakdown(own)
    assert info["trailing"] == 35
    assert info["leading"] == 0 and info["interior"] == 0, "loss is all at the end"
    assert info["ends_earliest"] == ["STALE"]
    joined = " ".join(notes)
    assert "staleness" in joined and "STALE" in joined
    assert "refresh recovers" in joined


def test_short_history_is_reported_as_leading_loss():
    full = pd.bdate_range("2020-01-01", periods=300)
    own = {"LONG": full, "NEW": full[40:]}
    info, notes = _breakdown(own)
    assert info["leading"] == 40 and info["trailing"] == 0 and info["interior"] == 0
    assert info["starts_latest"] == ["NEW"]
    assert "will not help" in " ".join(notes), "a refresh cannot fix a short history"


def test_interior_gaps_are_the_only_unavoidable_loss_and_name_the_culprit():
    full = pd.bdate_range("2020-01-01", periods=300)
    holiday = full.delete([100, 101, 102])
    own = {"A": full, "B": full, "C": holiday}
    info, notes = _breakdown(own)
    assert info["interior"] == 3 and info["leading"] == 0 and info["trailing"] == 0
    assert info["interior_blame"] == {"C": 3}
    joined = " ".join(notes)
    assert "INSIDE" in joined and "C (3)" in joined
    assert "bias" in joined, "non-random missingness must be flagged, not just counted"


def test_identical_calendars_cost_nothing_and_say_so():
    full = pd.bdate_range("2020-01-01", periods=200)
    info, notes = _breakdown({"A": full, "B": full})
    assert (info["leading"], info["trailing"], info["interior"]) == (0, 0, 0)
    assert "free" in " ".join(notes)
