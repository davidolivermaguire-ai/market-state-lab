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

msl list                              # registered estimators
msl run      -c configs/trend_indices.yaml        # walk-forward sweep -> tidy results
msl recovery                                      # tier-1 scoring on synthetic markets
msl state    -c configs/trend_indices.yaml        # current state per ticker
msl state    -c configs/trend_indices.yaml --offline   # committed CSVs only, no network
pytest -q                             # includes the look-ahead guard
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

## The per-ticker state view

`msl state` reports each method's latest filtered estimate per symbol, an equal-weight
consensus, and the council's disagreement:

```
symbol      as_of m_ma_cross m_return_sign m_ewma_slope consensus  p_up  agreement  disagreement
NAS100 2026-07-06         up            up        range        up 0.559      0.667         0.870
```

Read it as a **state estimate with uncertainty, not a trend confirmation.** The evidence on
this project is consistent: state earns its keep in calibration and risk control, not return
timing. A high `p_up` says the trend estimators agree drift has been positive and persistent —
not "buy". `disagreement` rising is the more actionable signal, since that is where the value of
state information concentrates.

## Status

**Phases 1–2 complete** (17 tests passing):

- data layer, shared feature layer, estimator contract + registry, four transparent baselines
- walk-forward engine, stability metrics, and the look-ahead guard
- **recovery suite** (tier 1) with permutation-invariant scoring, and the **per-ticker state report**

First recovery results, 8 simulated markets × 2,500 days — the speed/false-alarm frontier is
already explicit, and no method is close to solved:

| method | kind | balanced acc. | ARI | Brier | label gap | median delay | detection rate | false alarms/yr |
|---|---|---|---|---|---|---|---|---|
| **`hmm_gaussian`** | state | **0.616** | 0.422 | **0.510** | 0.185 | 4.9d | 0.77 | 9.0 |
| `ms_regression` | state | 0.610 | **0.431** | 0.521 | 0.324 | 3.0d | 0.79 | 21.9 |
| `bocpd` | changepoint | 0.477 | 0.098 | 0.645 | **0.040** | 8.5d | **0.80** | 15.8 |
| `ma_cross` | state | 0.430 | 0.058 | 0.735 | 0.059 | 23.4d | 0.33 | 2.8 |
| `return_sign` | state | 0.426 | 0.046 | 0.723 | 0.022 | 12.0d | 0.80 | 20.9 |
| `ewma_slope` | state | 0.421 | 0.031 | 0.734 | 0.025 | 3.3d | 0.95 | 33.7 |
| `cusum` | changepoint | 0.393 | 0.034 | 0.662 | 0.071 | 33.3d | 0.10 | **0.3** |
| `kalman_trend` (MLE) | state | 0.381 | 0.015 | 0.691 | 0.005 | 22.4d | 0.16 | 2.8 |
| `always_range` (null) | — | 0.333 | 0.000 | 0.653 | — | — | 0.00 | 0.0 |

**"Most reliably" has no single answer — it depends what you need the state for.** Rank by
classification and the switching models win. Rank by catching a change without crying wolf and
the ordering changes completely: `cusum` at the textbook threshold fires 0.3 false alarms a year
against `ewma_slope`'s 33.7. `bocpd` matches the best detection rate in the table (0.80) with
near-perfect label reliability (gap 0.040) while sitting mid-table on classification — a good
detector and a mediocre classifier, exactly as its design implies. Reporting one number would
have hidden all of this.

The CUSUM threshold traces the average-run-length trade-off directly, and — untuned, as with the
Kalman variance — the defaults sit at the conservative extreme:

| `h` | median delay | detection rate | false alarms/yr | balanced acc. |
|---|---|---|---|---|
| 2.0 | 11.9d | 0.571 | 5.18 | 0.382 |
| 3.0 | 15.6d | 0.254 | 1.51 | 0.413 |
| 4.0 | 27.2d | 0.090 | 0.45 | 0.424 |
| 5.0 (default) | 33.3d | 0.101 | 0.31 | 0.393 |
| 7.0 | 21.8d | 0.044 | 0.11 | 0.434 |

Detection and false alarms move monotonically with `h`, while balanced accuracy stays flat at
0.38–0.43 throughout. The threshold buys **timeliness, not classification** — which is precisely
why change-point detectors are scored on a different axis and carry `kind = "changepoint"`.

**The switching models decisively earn their complexity.** Both reach ~0.61 balanced accuracy
against a 0.33 floor, with roughly seven times the ARI of the best baseline, and both beat the
hedging null on calibration. When the truth switches, modelling switching beats modelling smooth
drift — which also confirms the Kalman result was **misspecification, not implementation**: the
local-linear-trend model was answering the wrong question, and its MLE faithfully optimised it.

**`label_gap` is the column that changed the design.** It is matched accuracy minus raw
accuracy — how much of a method's score depends on *relabelling* its states. Cross-checking the
hand-rolled HMM against statsmodels on the same model class exposed the problem: the two agreed
on the partition (ARI 0.44 vs 0.45) but their MAP labels agreed only 27% of the time, below
chance. Both were finding the right regimes and attaching the wrong semantics — and ARI, matched
accuracy and Brier all hid it completely.

The cause was ordering states by fitted mean. On some seeds EM lands on a higher-likelihood but
degenerate solution where a rarely-occupied state captures a short quiet episode at an extreme
mean (+0.5%/day at 0.3% vol), which scrambles the ordering. Rejecting fits whose stationary
occupancy falls below 8% cut the HMM's label gap from ~0.35 to 0.185 and its false alarms from
13.7 to 9.0/yr. `ms_regression` still carries a 0.324 gap and is the least label-reliable method
in the table despite the best ARI.

This matters practically, not just aesthetically: **anything consuming the labels literally — the
per-ticker state report, which prints "up" — is only trustworthy when the label gap is small.**
The transparent baselines have gaps of 0.02–0.06 because their semantics are hard-coded; the
sophisticated methods buy partition quality at the cost of label reliability. `label_gap` is now
reported on every run so that trade-off can never hide again.

One caveat holds throughout: this is a **home fixture**. The synthetic markets are generated from
a Markov-switching process, so the switching models' assumptions hold exactly. Recovery is
necessary, not sufficient.

Three further readings.

**Identification is hard.** Even on data generated from exactly the process these methods
target, balanced accuracy is 0.38–0.43 against a 0.33 chance floor. Regime drift is small
relative to daily noise.

**Discrimination and calibration are different axes.** `ma_cross` beats the null on
discrimination while scoring a *worse* Brier: it takes confident positions and pays when wrong,
where the null hedges. The harness reports both rather than the flattering one.

**The Kalman filter loses to a moving-average crossover — for a diagnosable reason.** Maximum
likelihood drives the slope-drift variance `q_slope` toward zero, freezing the slope so the
filter barely tracks regime changes (detection rate 0.16). MLE maximises the one-step predictive
likelihood of *price*, which is dominated by fitting observation noise — that is not the same
objective as *identifying the state*. Pinning `q_slope = 1e-8` instead lifts balanced accuracy to
**0.459**, above every baseline. MLE nonetheless stays the default, because choosing `q_slope` on
the recovery metric is tuning on the evaluation — the backtest-overfitting trap this project
exists to avoid. The parameter is exposed so the sensitivity can be *reported*; any principled
tuning has to happen in-fold against a pre-committed criterion.

> A methodology note worth keeping. On a single simulated market the pinned variant scored 0.604,
> and averaging over six markets it fell to 0.459. One market is one draw. The multi-seed design
> caught an over-claim that a single-seed run would have published.

Next: decision-value metrics → the cross-asset sweep on **real** data (everything above is a home
fixture) → deflated significance for the method × asset comparison → write-up.

## Honest notes

- Index symbols are **price** indices (no dividends) — fine for state identification, not for
  total-return claims.
- Refit cadence is a hidden hyperparameter that can flatter slow methods; it is set in config and
  reported with results.
- Comparing ~8 methods × ~10 assets is a multiple-comparisons problem. Rankings will be reported
  with deflated significance / a Model Confidence Set, not as a naive leaderboard.
- Single-name universes are fixed, pre-declared lists — never edited to match today's index
  membership, which is what keeps the panel free of survivorship bias.
