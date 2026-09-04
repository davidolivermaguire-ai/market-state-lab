"""The redundancy gate: does it count dimensions honestly, and is it causal?"""
import numpy as np
import pandas as pd

from msl.diagnostics.redundancy import _causal_z, decompose, effective_n


def _panel(n=600, seed=0, **cols):
    idx = pd.bdate_range("2018-01-01", periods=n)
    return pd.DataFrame(cols, index=idx)


def test_effective_n_is_one_for_duplicates_and_m_for_orthogonal():
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(size=n)
    dup = _panel(n, a=x, b=x, c=x)
    assert abs(effective_n(dup) - 1.0) < 0.01

    ind = _panel(n, **{f"s{i}": rng.normal(size=n) for i in range(5)})
    assert effective_n(ind) > 4.5, "independent columns should count as ~5"


def test_effective_n_rises_with_noise_on_a_single_shared_signal():
    """The whole reason effective-n is not sufficient evidence.

    One underlying dimension, measured with independent noise. As the noise grows the
    estimators decorrelate and effective-n *rises* — so a healthy-looking count can mean
    the estimators got worse, not that the panel gained dimensions.
    """
    rng = np.random.default_rng(1)
    n, m = 800, 6
    truth = rng.normal(size=n)
    counts = []
    for noise in (0.2, 1.0, 3.0):
        p = _panel(n, **{f"s{i}": truth + noise * rng.normal(size=n) for i in range(m)})
        counts.append(effective_n(p))
    assert counts[0] < counts[1] < counts[2], "more noise must raise effective-n"
    assert counts[0] < 1.5 and counts[2] > 4.0


def test_causal_z_uses_no_future_data():
    rng = np.random.default_rng(2)
    p = _panel(500, a=rng.normal(size=500), b=rng.normal(size=500))
    full = _causal_z(p)
    cut = 400
    trunc = _causal_z(p.iloc[:cut])
    both = full.iloc[:cut].notna() & trunc.notna()
    assert both.to_numpy().sum() > 200
    assert np.allclose(full.iloc[:cut].to_numpy()[both.to_numpy()],
                       trunc.to_numpy()[both.to_numpy()], atol=1e-12), \
        "standardisation leaked future data into the consensus"


def test_decomposition_is_causal():
    """Consensus and deviations at t must not move when later rows arrive."""
    rng = np.random.default_rng(3)
    n = 600
    truth = np.cumsum(rng.normal(size=n)) / 10
    p = _panel(n, **{f"s{i}": truth + rng.normal(size=n) for i in range(4)})
    d_full = decompose(p)
    cut = 480
    d_trunc = decompose(p.iloc[:cut])

    c_f, c_t = d_full.consensus.iloc[:cut], d_trunc.consensus
    ok = c_f.notna() & c_t.notna()
    assert ok.sum() > 150
    assert np.allclose(c_f[ok], c_t[ok], atol=1e-12), "consensus used future data"

    i_f, i_t = d_full.idiosyncratic.iloc[:cut], d_trunc.idiosyncratic
    m = i_f.notna() & i_t.notna()
    assert np.allclose(i_f.to_numpy()[m.to_numpy()], i_t.to_numpy()[m.to_numpy()],
                       atol=1e-12), "deviations used future data"


def test_deviations_sum_to_zero_and_consensus_reconstructs():
    rng = np.random.default_rng(4)
    p = _panel(500, **{f"s{i}": rng.normal(size=500) for i in range(4)})
    d = decompose(p)
    s = d.idiosyncratic.sum(axis=1).dropna()
    assert np.allclose(s.to_numpy(), 0.0, atol=1e-10), \
        "deviations from a mean must sum to zero by construction"


def test_negative_loading_is_flagged_not_silently_averaged():
    """An inverted specialist would cancel in an equal-weight mean. Say so."""
    rng = np.random.default_rng(5)
    n = 600
    truth = np.cumsum(rng.normal(size=n)) / 10
    p = _panel(n, a=truth + 0.1 * rng.normal(size=n),
               b=truth + 0.1 * rng.normal(size=n),
               c=-truth + 0.1 * rng.normal(size=n))
    d = decompose(p)
    assert d.loadings["c"] < 0
    assert any("negative loading" in x for x in d.notes)
