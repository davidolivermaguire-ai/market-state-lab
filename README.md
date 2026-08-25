# market-state-lab

A benchmark harness for **RQ1** of the [Adaptive Market Intelligence proposal](https://davidmaguire.ai/phd/):

> *Which statistical, econometric and ML methods most reliably identify trend, volatility,
> liquidity and structural-change states out of sample?*

RQ1 is a **comparative** question, so this is a *protocol*, not a model. One data layer, one
shared feature set, a pluggable estimator interface, one walk-forward engine, one metrics
suite — methods are interchangeable parts competing on identical terms. Trend is built first;
the same harness serves the other three dimensions.

## The two ideas that shape everything

**1. Transparent baselines are the bar.** A Kalman filter or an HMM that cannot beat a 50/200
moving-average crossover has not earned its complexity. The baselines live in the same harness,
on the same features, scored by the same metrics — which turns that from an opinion into a result.

**2. "Reliably" needs defining, because there is no ground truth.** You cannot score a state
estimator against labels that do not exist, and manufacturing labels from forward returns builds
circularity and look-ahead into the metric. So reliability is an **evidence hierarchy**,
cleanest first:

| Tier | Question | Where truth comes from |
|---|---|---|
| **Recovery** | Can it recover a regime it was designed for? | Synthetic data with a known hidden state |
| **Stability** | Does the state persist long enough to be usable? | No labels needed — flip rate, duration |
| **Timeliness** | How fast does it flag a change, at what false-alarm cost? | Small pre-registered event list |
| **Decision value** | Does conditioning on it improve calibration and drawdown control? | Out-of-sample, vs a volatility-only model on the same information |

Decision value is the real test. The others are necessary, not sufficient.

**Multi-asset is the replication dimension, not a convenience.** A method that wins on NAS100
alone has produced one draw. Running indices *and* single names turns the comparison into a
panel, and **agreement of the ranking across assets** is the actual evidence for "most reliably".

## Install and run

```bash
pip install -e ".[data,dev]"      # 'data' adds yfinance + parquet, 'dev' adds pytest

msl list                          # registered estimators
msl run -c configs/trend_indices.yaml
msl run -c configs/trend_indices.yaml --offline   # committed CSVs only, no network
pytest -q                         # includes the look-ahead guard
```

```python
from msl.data import load_universe
from msl.engine import run_sweep
from msl.engine.walkforward import state_summary

prices = load_universe(["NAS100", "US500", "US30", "AAPL"], start="2010-01-01")
results = run_sweep(prices, ["ma_cross", "return_sign", "ewma_slope"])
print(state_summary(results))
```

## The two contracts

**Causality.** `filter()` returns a *filtered* estimate: the value at time *t* uses only rows at
or before *t*. Never a smoothed pass over a completed history — that is look-ahead, and it is the
easiest way to produce a beautiful, useless result. `tests/test_no_lookahead.py` asserts that
every registered estimator's output at *t* is unchanged when future rows are appended, and does
the same for the feature layer. A new method cannot be added without passing it.

**One output schema.** Every estimator emits the same frame — a probability vector over
`{down, range, up}`, a MAP label, and a continuous `score` in [-1, 1]. Discrete methods (HMM) and
continuous ones (Kalman slope) stay comparable, and the continuous score keeps
[RQ2](https://davidmaguire.ai/phd/) (overlapping dimensions vs one exclusive label) open rather
than pre-judged.

Adding a method is one file plus a decorator; nothing downstream changes:

```python
@register("my_method")
class MyMethod(StateEstimator):
    requires_fit = True
    def fit(self, features): ...; return self
    def filter(self, features): return softmax_states(my_causal_score(features))
```

## Layout

```
src/msl/
  data/        symbol map (NAS100 -> ^NDX), committed CSV -> cache -> download
  features/    the shared causal information set
  estimators/  base.py (contract + registry), baselines.py, …
  engine/      walkforward.py (purge/embargo/refit), synthetic.py (known-regime sim)
configs/       experiment definitions
data/raw/      committed CSVs behind published results — reproducible with no network
results/       tidy output: date | symbol | method | p_down | p_range | p_up | map_state | score
tests/         look-ahead guard + schema contract tests
```

## Status

**Phase 1 complete** — data layer, feature layer, estimator contract + registry, four transparent
baselines, walk-forward engine, stability metrics, look-ahead guard (12 tests passing).

Next: synthetic recovery suite → Kalman / HMM / MS-AR → CUSUM / BOCPD → timeliness and
decision-value metrics → cross-asset sweep and report.

## Honest notes

- Index symbols are **price** indices (no dividends) — fine for state identification, not for
  total-return claims.
- Refit cadence is a hidden hyperparameter that can flatter slow methods; it is set in config and
  reported with results.
- Comparing ~8 methods × ~10 assets is a multiple-comparisons problem. Rankings will be reported
  with deflated significance / a Model Confidence Set, not as a naive leaderboard.
- Single-name universes are fixed, pre-declared lists — never edited to match today's index
  membership, which is what keeps the panel free of survivorship bias.
