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
msl decision -c configs/trend_indices.yaml        # tier-4: decision value vs a vol-only benchmark
pytest -q                             # includes the look-ahead guard
```

Useful flags: `--offline` uses committed CSVs only and never touches the network; `--refresh`
skips the committed CSVs and downloads, so every symbol shares an as-of date; `--no-control`
drops the flip-rate control from the `decision` benchmark (less conservative).

Reproducing the published result exactly — one asset, all eight scored methods:

```bash
msl decision -c configs/trend_indices.yaml --offline
```

The committed `data/raw/NAS100.csv` is frozen at 2026-07-06 so this is deterministic. Adding
`--refresh` will pull newer data and move the numbers slightly; that is expected, not a bug.

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
symbol      as_of m_ma_cross m_return_sign m_ewma_slope consensus  p_up  agreement  uncertainty  disagreement
NAS100 2026-08-24         up            up        range        up 0.559      0.667        0.870         0.___
```

Read it as a **state estimate with uncertainty, not a trend confirmation.** The evidence on
this project is consistent: state earns its keep in calibration and risk control, not return
timing. A high `p_up` says the trend estimators agree drift has been positive and persistent —
not "buy".

**`uncertainty` and `disagreement` are different things, and conflating them was a real bug.**
The first live run showed US500 with `agreement 1.00` (all three methods said "range") *and*
`disagreement 0.97` — an apparent contradiction. The cause: the metric was the entropy of the
*averaged* distribution, which cannot distinguish three methods that each shrug from three
methods that confidently contradict each other. Both produce a flat average. They now decompose:

```
H(mean p)   =   mean H(p_i)   +   JSD
uncertainty     individual doubt   disagreement
```

On the two cases the old metric scored identically (~0.99): three weakly-agreeing methods now
give **disagreement 0.000**, three confidently-contradicting ones give **0.564**. This matters
because the proposal's claim is that the value of state information concentrates where the
council *disagrees* — so only the second term is the transition signal, and the old metric was
not measuring it.

**Watch the as-of dates.** Committed CSVs are frozen at their publication date, so mixing them
with freshly-downloaded symbols compares states from different days. `msl state` now warns when
the as-of spread exceeds five days; `--refresh` forces downloads so every symbol shares a date.

## Decision value: the tier that answers RQ1

The other tiers ask whether methods agree, persist, and recover known regimes. This one asks
whether the state is *worth anything* — and against the only benchmark that makes the question
meaningful: **a model seeing the same volatility features that simply does not see the state.**
Comparing against buy-and-hold, or against nothing, would only prove volatility is informative,
which nobody disputes.

`msl decision -c configs/trend_indices.yaml`

**Calibration.** Walk-forward logistic regression on `[rv20, rv60]` versus
`[rv20, rv60, p_down, p_range, p_up]`, purged by the label horizon. Negative = the state helped.
Real NAS100, 5-day horizon:

| method | direction | loss | volatility |
|---|---|---|---|
| `bocpd` | +0.0080 | +0.0032 | **−0.0051** |
| `hmm_gaussian` | +0.0023 | +0.0001 | **−0.0013** |
| `kalman_trend` | +0.0007 | −0.0002 | +0.0046 |
| `ma_cross` | +0.0042 | +0.0009 | +0.0016 |

**Every method makes direction forecasts worse.** Not one of the four improves on volatility
alone at predicting whether the next week is positive — a clean, unanimous null on return timing.

### Then the panel contradicted the single-asset story

Running all nine methods across NAS100 + US500 + US30 replicated some of that and broke the rest:

| method | direction | loss | volatility |
|---|---|---|---|
| `ewma_slope` | +0.0017 | −0.0003 | **−0.0242** |
| `return_sign` | +0.0020 | +0.0006 | −0.0055 |
| `bocpd` | +0.0013 | +0.0002 | −0.0047 |
| `ma_cross` | +0.0010 | +0.0008 | −0.0027 |
| `hmm_gaussian` | +0.0015 | +0.0028 | +0.0005 |
| `kalman_trend` | +0.0030 | +0.0002 | +0.0022 |
| `ms_regression` | +0.0018 | +0.0069 | +0.0063 |
| `cusum` | +0.0022 | +0.0004 | +0.0091 |

**Direction held: 8 of 8 methods worse.** That null is now replicated across three assets and is
the most robust result in the project.

**The volatility story did not hold.** From NAS100 alone this README claimed the benefit was
"specific to methods that model volatility and change". On the panel `bocpd` replicated
(−0.0047 vs −0.0051) but `hmm_gaussian` did not (+0.0005, was −0.0013), and the largest gain by
far comes from **`ewma_slope` — a trend rule** (−0.0242, ~10% relative). The original claim was
generalised from one asset and was wrong.

### The confound hypothesis was wrong

The obvious explanation was that `ewma_slope` is the *least stable* estimator (5-day mean
duration), so its probabilities oscillate rapidly — and oscillation rate is itself a volatility
signal. On that story the "gain" would be volatility re-encoded through the estimator's own
instability, not state information at all.

`calibration_gain(..., control_instability=True)` tests it directly by adding the estimator's own
causal rolling flip rate to the **baseline**, so the benchmark already sees how much the state has
been churning. If the hypothesis held, the gain would collapse. It did the opposite:

| method (volatility target) | plain | t | + flip-rate control | t |
|---|---|---|---|---|
| `ewma_slope` | −0.0157 | −3.69 | **−0.0171** | **−3.98** |
| `bocpd` | −0.0051 | −1.80 | −0.0074 | −2.60 |
| `hmm_gaussian` | −0.0013 | −0.30 | −0.0043 | −1.57 |
| `ma_cross` | +0.0016 | +0.84 | −0.0002 | −0.11 |

Every effect got *stronger*, not weaker. **Instability-as-volatility-proxy is rejected** — the
state probabilities carry information beyond both trailing realised volatility and the
estimator's own churn. The mechanism remains unexplained, which is the honest position.

### Multiple testing: what actually survives

`deflate()` applies two corrections. Benjamini-Hochberg FDR controls the share of false claims
among those declared significant. The **deflated bar** takes the Bailey / López de Prado point and
applies it to loss differentials: under the null the largest of N t-statistics is around
`sqrt(2 ln N)`, so with 24 comparisons the bar is |t| > 2.52 — not 1.96 — and with the full
189-cell grid it is 3.24. Statistics use Diebold-Mariano with Newey-West standard errors, because
overlapping 5-day labels make the differentials serially correlated and an iid error would
manufacture significance.

On NAS100, all eight scored methods × three targets = **24 comparisons**, instability-controlled,
bar |t| > sqrt(2 ln 24) = **2.52**:

| result | brier delta | t | p (FDR) | survives |
|---|---|---|---|---|
| `ewma_slope` / volatility | −0.0171 | **−3.98** | 0.0008 | **FDR + deflated, and would clear the 189-cell bar** |
| `bocpd` / volatility | −0.0074 | −2.60 | 0.038 | FDR + deflated at N=24; **fails** at N=189 |
| everything else negative | — | > −1.9 | — | no |

(`always_range` is excluded from this tier: a state that never changes carries no information to
test, and scoring it would only pad N and loosen the bar.)

**One result survives everything.** And the direction/loss columns are not merely nulls — several
are significant *harms*: `bocpd`/loss t = **+4.57**, `return_sign`/loss +3.84, `bocpd`/direction
+3.31. Adding state to a volatility-only model does not just fail to help there, it measurably
hurts.

A tension worth keeping: the one robust winner is `ewma_slope` — a trend rule that ranks *worst*
on the stability tier (5-day duration, 0.19 flip rate). It is a poor state estimator by every
persistence measure and the best one by this calibration measure. Any account of RQ1 has to
explain that rather than average it away.

Caveats: this deflation is NAS100-only (24 comparisons); the 189 figure is a projection, and the
full grid needs running. The DM test's empirical size is ~0.07 against a nominal 0.05 — slightly
liberal, measured in `tests/test_significance.py` rather than assumed.

This is the second time in this project a single-draw result failed to replicate (the first was a
Kalman variant scoring 0.604 on one synthetic seed and 0.459 over six). Both were caught by the
panel design rather than by inspection.

**Risk control.** Volatility targeting with and without a defensive state overlay
(`exposure × (1 − p_down)`, pre-committed, can only reduce exposure):

Averaged across the three indices, all nine methods:

| strategy | Sharpe | vol | max DD | **DD/vol** | turnover |
|---|---|---|---|---|---|
| buy and hold | 0.713 | 18.2% | −38.2% | −2.120 | 0.07 |
| vol target (baseline) | **0.749** | 11.3% | −16.9% | **−1.497** | 8.8 |
| vol target × state | 0.667 | 9.2% | −16.5% | −1.806 | 10.7 |

The raw drawdown column is a **trap, and reporting it alone would have been misleading**: the
overlay can only cut exposure, so of course volatility and drawdown fall. A rule that simply
holds less will always show less drawdown. Read `sharpe` (scale-invariant) and `dd_per_vol`
instead — and on those the overlay is **worse on both**: Sharpe 0.667 against 0.749, drawdown per
unit of volatility −1.806 against −1.497, with higher turnover on top. The apparent 16.9% → 16.5%
drawdown improvement is entirely an artefact of holding less.

This null replicated and strengthened on the panel: on NAS100 alone the overlay was merely
inconsistent, across three assets it is clearly worse.

### What this says for RQ1

Two results are solid, one is not.

**Solid — state does not time returns.** Unanimous across 8 methods and 3 assets. **Solid — state
adds no risk-adjusted sizing value** beyond a volatility scaler; the same de-risking is free by
lowering the volatility target. Both support the proposal's position, now against a fair
benchmark rather than asserted.

**Not solid — the volatility-calibration channel.** The direction of the effect is method-specific
and did not survive replication in the form first claimed. `bocpd` held up; the interpretation
did not.

Honest limitations.

1. **The gains are small.** 2–10% relative Brier. Statistically distinguishable is not
   economically interesting, and nothing here has been costed as a strategy.
2. **The deflation is single-asset.** 24 comparisons on NAS100. The full 8 × 7 × 3 = 168-cell
   panel (189 with the ninth method) raises the bar to |t| > 3.24, and on cross-asset runs
   `ewma_slope` held (−0.0138 over seven assets) while `bocpd` decayed with every widening of the
   sample (−0.0074 → −0.0047 → −0.0020). Running that grid under deflation is the outstanding
   test, and until it exists `bocpd` should be read as a hypothesis, not a finding.
3. **The scored samples are unequal.** Methods that fit parameters give up a training window and
   score 1,755 days against the baselines' 2,355. Each is compared to a benchmark on its own
   sample, so no individual comparison is biased — but ranking a fitted method against a baseline
   is not quite like-for-like. A common-window re-run is the clean version.
4. **The mechanism is unexplained.** The one survivor is the least stable estimator in the panel,
   and the obvious confound was tested and rejected. That is an open question, not a result.

## Two methods failed on real data — and why

The first full real-data sweep (9 methods × 7 assets, 2010–2026) exposed two degenerate
estimators. Both failures came from the same source: **parameter values that encode
signal-to-noise assumptions markets do not satisfy.**

**Kalman: maximum likelihood froze the filter.** MLE drove `q_slope` to **1e-12 — the exact
lower bound** — which freezes the slope. Since equity log-price has persistent positive drift, a
frozen slope stays positive forever: the filter reported "up" on **100% of days for four of the
seven assets**. A constant, not a state estimator. The cause is an objective mismatch — MLE
maximises the one-step predictive likelihood of *price*, dominated by fitting observation noise,
which is not the same objective as identifying state.

**CUSUM: the textbook slack is ten times the signal.** `k = 0.5` comes from manufacturing process
control, where a shift worth detecting is about one standard deviation. Measured on the Nasdaq,
the mean standardised return is **+0.079** — the trend signal is **16% of that slack**. Evidence
never accumulates: four triggers in 2,872 days, one state change every ~718 days, reading "down"
84% of the time through a bull market because a rare volatility event tripped it and nothing
since moved it.

### The fix: specify from training data, never from the metric

| | before | after | effect on NAS100 |
|---|---|---|---|
| `kalman_trend` | MLE → `q_slope` 1e-12 | prior → **2.8e-08** | 100% "up" → 42/50/8 mix, 11.6-day duration |
| `cusum` | `k` 0.5, `h` 5.0 | **`k` 0.121, `h` 9.30** | 718-day → 214-day duration; 84% "down" → 75% "up" |

Kalman's slope variance now comes from a **stated prior** — trend drift can wander by a tenth of
daily volatility over roughly two months, so `q_slope = (0.1·σ_r)² / 42` with `σ_r` measured on
the training window. CUSUM is designed the way SPC prescribes: `k` = half the shift worth
detecting (from the dispersion of the slow-moving mean of standardised returns) and `h`
calibrated to a target in-control average run length, both on training data.

Neither was chosen by looking at a recovery score — that would be selecting on the evaluation,
the exact trap this project exists to avoid. `q_slope = "mle"` still reproduces the failure.

Worth noting as corroboration rather than justification: the prior lands on 2.8e-08, close to the
1e-08 that independently maximised synthetic recovery. And the repaired Kalman now tracks
`hmm_gaussian` closely (42/50/8 vs 41/48/11), which is two structurally different methods
agreeing — a better sign than either number alone.

One honest limitation: CUSUM never returns to "range" once it commits, so it is effectively a
two-state detector with "range" only as its initial condition.

## Performance

The engine was quadratic in two independent places, which made a full sweep unusable. Both are
now bounded, so total cost is linear in series length:

- **`warmup`** bounds the history replayed before each scored block (was: the entire prefix,
  every refit).
- **`max_train`** bounds the history each refit sees, making the fit a *rolling* rather than
  expanding window — this is where the maximum-likelihood estimators' cost actually sat.

Both are approximations and are **measured, not assumed**: `tests/test_walkforward.py` compares
bounded against full replay and fails if the difference grows. One estimator opts out entirely.
The Kalman local-linear-trend filter sets `full_replay = True`, because MLE drives its slope
variance toward zero, which makes the slope an integral of the entire history rather than a
fading average — windowing it changed the answer (MAP agreement fell to ~0.8), so it doesn't.

Alongside that: BOCPD's Student-t is computed directly instead of through `scipy.stats`
(thousands of calls per series, where scipy's per-call overhead dominates), the HMM's transition
update is one matmul instead of a Python loop over timesteps, and both the HMM and
`ms_regression` warm-start each refit from the previous block's solution.

Measured on real NAS100 (2,892 rows), per asset:

| method | before | after |
|---|---|---|
| `hmm_gaussian` | 115.3s | **14.3s** |
| `ms_regression` | — | 70.1s |
| `kalman_trend` | — | 17.1s |
| `bocpd` | — | 0.2s |
| baselines / `cusum` | — | ~0.0s |

A full 7-asset sweep is ~12 minutes, and the test suite went from 110s to 18s. `ms_regression`
now dominates; raising `refit_every` in the config is the lever if that matters.

## Status

**All four scoring tiers are built** — 42 tests passing, nine estimators registered.

- data layer, shared feature layer, estimator contract + registry, four transparent baselines
- walk-forward engine, stability metrics, and the look-ahead guard
- **recovery suite** (tier 1) with permutation-invariant scoring, and the **per-ticker state report**
- Kalman local-linear-trend, Gaussian HMM, Markov-switching regression, CUSUM and BOCPD
- **decision value** (tier 4): calibration gain vs a volatility-only benchmark, Diebold-Mariano
  with Newey-West errors, flip-rate control, Benjamini-Hochberg FDR and a deflated bar

Outstanding: the full cross-asset grid under deflation, and a common-window re-run so every
method is scored on identical dates.

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

Next: the full method × asset grid under deflation, and a common-window re-run.

## Honest notes

- Index symbols are **price** indices (no dividends) — fine for state identification, not for
  total-return claims.
- Refit cadence is a hidden hyperparameter that can flatter slow methods; it is set in config and
  reported with results.
- Comparing 8 methods × 3 targets is already a multiple-comparisons problem, which is why
  `deflate()` applies both Benjamini-Hochberg FDR and a `sqrt(2 ln N)` bar. Rankings are never
  reported as a naive leaderboard.
- Single-name universes are fixed, pre-declared lists — never edited to match today's index
  membership, which is what keeps the panel free of survivorship bias.
- Everything in this repo is **filtered, never smoothed**: the estimate at time *t* uses only data
  up to *t*, asserted by `tests/test_no_lookahead.py` rather than by inspection.

## Results write-up

The tier-4 findings are written up in full, with every parameter stated, at
[davidmaguire.ai/quant-lab/state-decision-value](https://davidmaguire.ai/quant-lab/state-decision-value/).

## License

MIT — see [LICENSE](LICENSE). This is research code: it produces state estimates, not trading
signals, and nothing it outputs is investment advice.
