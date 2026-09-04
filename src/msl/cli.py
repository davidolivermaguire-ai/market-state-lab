"""Command line entry points.

    msl list                                  # registered estimators
    msl run      -c configs/trend_indices.yaml    # walk-forward sweep -> tidy results
    msl recovery                                  # tier-1 scoring on synthetic markets
    msl state    -c configs/trend_indices.yaml    # current state per ticker (a view, not a signal)

Add --offline to any data-backed command to use committed CSVs only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

from msl.data.loaders import _repo_root, load_prices, load_universe
from msl.engine.walkforward import run_sweep, state_summary
from msl.estimators.base import list_estimators
from msl.features.core import build_features

# Not specialists: these carry no state estimate to be redundant *with*. `always_range`
# is a constant and `return_sign` is the raw observation, so including either in a
# consensus would dilute it with something that is not an opinion.
BASELINE_METHODS = frozenset({"always_range", "return_sign"})


def _find_config(p: str) -> Path:
    """Locate a config whether or not the shell happens to be in the project folder.

    A new terminal rarely opens in the repo, and requiring one is a papercut that
    costs more than the two lines it takes to avoid.
    """
    path = Path(p)
    if path.exists():
        return path
    alt = _repo_root() / p
    if alt.exists():
        return alt
    raise FileNotFoundError(f"config not found: tried '{path}' and '{alt}'")


def _anchor(p: str | Path) -> Path:
    """Anchor a relative output path to the project, not the current directory."""
    path = Path(p)
    return path if path.is_absolute() else _repo_root() / path


def _write(df: pd.DataFrame, base: Path) -> Path:
    base = _anchor(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    try:
        path = base.with_suffix(".parquet")
        df.to_parquet(path, index=False)
    except Exception:            # pyarrow not installed - CSV is a fine fallback
        path = base.with_suffix(".csv")
        df.to_csv(path, index=False)
    return path


def cmd_run(args: argparse.Namespace) -> int:
    cfg = yaml.safe_load(_find_config(args.config).read_text(encoding="utf-8"))
    eng = cfg.get("engine", {}) or {}

    print(f"[{cfg.get('name', 'run')}] loading prices…")
    prices = load_universe(
        cfg["universe"], cfg.get("start"), cfg.get("end"),
        allow_download=not args.offline, prefer_fresh=args.refresh,
    )
    print(f"  loaded {len(prices)} symbol(s): {', '.join(prices)}")

    # A cross-asset comparison is only like-for-like if the symbols cover the same
    # period. A committed CSV frozen over a shorter window silently puts one asset on
    # a different sample — and different regimes — from the rest of the panel.
    spans = {s: (df.index[0].date(), df.index[-1].date(), len(df)) for s, df in prices.items()}
    n_rows = [v[2] for v in spans.values()]
    if max(n_rows) - min(n_rows) > 0.1 * max(n_rows):
        print("\n  [warn] symbols cover different periods — the panel is not like-for-like:")
        for s, (lo, hi, n) in sorted(spans.items(), key=lambda kv: kv[1][2]):
            print(f"           {s:<8} {lo} .. {hi}  ({n} rows)")
        print("         Re-run with --refresh to download all symbols over a common window.")

    print("\nrunning sweep…")
    results = run_sweep(
        prices,
        cfg["methods"],
        min_train=eng.get("min_train", 750),
        refit_every=eng.get("refit_every", 63),
        embargo=eng.get("embargo", 5),
        warmup=eng.get("warmup", 750),
        max_train=eng.get("max_train", 1250),
    )

    out_base = Path(cfg.get("output", f"results/{cfg.get('name', 'run')}"))
    res_path = _write(results, out_base)
    summary = state_summary(results)
    sum_path = _write(summary, out_base.with_name(out_base.name + "_summary"))

    print(f"\nresults  -> {res_path}  ({len(results):,} rows)")
    print(f"summary  -> {sum_path}\n")
    with pd.option_context("display.width", 120, "display.max_columns", 20):
        print(summary.round(3).to_string(index=False))
    return 0


def cmd_recovery(args: argparse.Namespace) -> int:
    """Tier 1: can each method recover a regime on data where truth is known?"""
    from msl.metrics.recovery import run_recovery

    seeds = tuple(range(args.seeds))
    print(f"recovery suite — {args.seeds} simulated markets x {args.n} days\n")
    per_seed, agg = run_recovery(methods=args.methods, seeds=seeds, n=args.n)

    cols = ["method", "balanced_accuracy", "ari", "brier", "label_gap",
            "median_delay_days", "detection_rate", "false_alarms_per_year"]
    with pd.option_context("display.width", 140, "display.max_columns", 20):
        print(agg[cols].round(3).to_string(index=False))

    out = Path(args.out)
    _write(per_seed, out.with_name(out.name + "_per_seed"))
    path = _write(agg, out)
    print(f"\nwritten -> {path}")
    print("\nRecovery is necessary, not sufficient: it says a method works when its "
          "assumptions hold,\nnot that they hold in the market.")
    return 0


def cmd_state(args: argparse.Namespace) -> int:
    """The per-ticker current-state view — a state estimate, not a trade signal."""
    from msl.report.state_report import state_report

    cfg = yaml.safe_load(_find_config(args.config).read_text(encoding="utf-8"))
    prices = load_universe(
        cfg["universe"], cfg.get("start"), cfg.get("end"),
        allow_download=not args.offline, prefer_fresh=args.refresh,
    )
    methods = [m for m in cfg["methods"] if m != "always_range"]
    rep = state_report(prices, methods, as_of=args.as_of)

    with pd.option_context("display.width", 170, "display.max_columns", 30):
        print("\n" + rep.to_string(index=False))

    # Symbols must share an as-of date to be comparable. A committed CSV is frozen at
    # the date it was published, so mixing it with fresh downloads silently compares
    # states from different days.
    spread = (pd.to_datetime(rep["as_of"]).max() - pd.to_datetime(rep["as_of"]).min()).days
    if spread > 5:
        print(f"\n  [warn] as-of dates span {spread} days — these states are NOT comparable.")
        print("         A symbol is loading from a frozen committed CSV. Re-run with --refresh.")

    print("\nRead as a filtered state estimate with uncertainty — not a trend confirmation.")
    print("`uncertainty` is how unsure the consensus is; `disagreement` is how much the")
    print("methods actually contradict each other. Only the second is the transition signal.")
    return 0


def cmd_decision(args: argparse.Namespace) -> int:
    """Tier 4: does the state add anything beyond a volatility signal?"""
    from msl.engine.walkforward import run_estimator
    from msl.estimators.base import get_estimator
    from msl.features.core import build_features
    from msl.metrics.decision import calibration_gain, deflate, risk_control, scored_index

    cfg = yaml.safe_load(_find_config(args.config).read_text(encoding="utf-8"))
    prices = load_universe(cfg["universe"], cfg.get("start"), cfg.get("end"),
                           allow_download=not args.offline, prefer_fresh=args.refresh)
    methods = [m for m in cfg["methods"] if m != "always_range"]
    control = not args.no_control

    cal_rows, risk_rows = [], []
    for symbol, px in prices.items():
        feats = build_features(px)

        # Pass 1: estimate. Methods that fit surrender a training window, so their
        # usable spans differ; scoring each on its own span makes every individual
        # comparison fair but the cross-method ranking a comparison of periods.
        est: dict[str, pd.DataFrame] = {}
        for name in methods:
            try:
                est[name] = run_estimator(feats, get_estimator(name))
            except Exception as exc:
                print(f"  [warn] {name} failed on {symbol}: {exc}")
        if not est:
            continue

        common = None
        if args.common_window:
            spans = {n: scored_index(s, feats, args.horizon, control) for n, s in est.items()}
            common = spans[min(spans, key=lambda n: len(spans[n]))]
            for idx in spans.values():
                common = common.intersection(idx)
            lo, hi = min(len(i) for i in spans.values()), max(len(i) for i in spans.values())
            print(f"  {symbol}: spans {lo}–{hi} days -> common window {len(common)}")

        # Pass 2: score, every method on identical dates when --common-window is set.
        for name, states in est.items():
            c = calibration_gain(states, feats, horizon=args.horizon,
                                 control_instability=control, common_index=common)
            if not c.empty:
                c.insert(0, "method", name); c.insert(0, "symbol", symbol); cal_rows.append(c)
            r = risk_control(states, feats, common_index=common)
            if not r.empty:
                r = r.assign(method=name, symbol=symbol); risk_rows.append(r)
            print(f"  {symbol:<8} {name}")

    if not cal_rows:
        raise RuntimeError("no decision-value results")
    cal = pd.concat(cal_rows, ignore_index=True)
    risk = pd.concat(risk_rows, ignore_index=True)

    ctrl = "flip-rate controlled" if not args.no_control else "NOT instability-controlled"
    print(f"\n=== CALIBRATION: mean brier_delta ({ctrl}; negative = the state helped) ===")
    with pd.option_context("display.width", 140):
        print(cal.pivot_table(index="method", columns="target", values="brier_delta").round(4).to_string())

    # A grid of this size guarantees some large t-statistics by chance. Report what
    # survives the correction, not the raw leaderboard.
    dfl = deflate(cal)
    bar = dfl["t_threshold"].iloc[0]
    print(f"\n=== SIGNIFICANCE: {len(dfl)} comparisons, deflated bar |t| > {bar:.2f} ===")
    win = dfl[dfl["survives_deflated"] | dfl["survives_fdr"]]
    if win.empty:
        print("  Nothing survives. Every apparent gain is within what searching this many")
        print("  times produces by chance.")
    else:
        cols = ["symbol", "method", "target", "brier_delta", "t_stat", "p_fdr",
                "survives_fdr", "survives_deflated"]
        with pd.option_context("display.width", 160):
            print(win.sort_values("t_stat")[cols].round(4).to_string(index=False))
    harm = dfl[(dfl["t_stat"] > bar)]
    print(f"\n  significant HARMS (state made calibration worse): {len(harm)} of {len(dfl)}")

    print("\n=== RISK CONTROL: risk-adjusted, averaged across assets ===")
    agg = risk.groupby("strategy")[["sharpe", "vol", "max_drawdown", "dd_per_vol", "turnover_pa"]]
    print(agg.mean().round(3).to_string())

    _write(cal, Path(cfg.get("output", "results/run")).with_name("decision_calibration"))
    path = _write(risk, Path(cfg.get("output", "results/run")).with_name("decision_risk"))
    print(f"\nwritten -> {path.parent}")
    print("\nRaw drawdown across strategies at different volatilities is confounded: a rule")
    print("that simply holds less always shows less drawdown. Read `sharpe` and `dd_per_vol`.")
    return 0


def cmd_data(args: argparse.Namespace) -> int:
    """Step 1 visibility: what will I load, from where, and is it comparable?"""
    from msl.data.audit import audit, search_space
    from msl.data.calendar import INTERVALS, get_interval
    from msl.data.symbols import UNIVERSES, catalogue

    if args.catalogue:
        print("\n=== symbol registry ===")
        with pd.option_context("display.width", 170, "display.max_colwidth", 46):
            print(catalogue().to_string(index=False))
        print("\n=== universes ===")
        for u, m in UNIVERSES.items():
            print(f"  {u:<14} {len(m)} symbols: {', '.join(m)}")
        print("\n=== intervals ===")
        for k, iv in INTERVALS.items():
            print(f"  {k:<5} {iv.label:<8} {iv.periods_per_year:>6.0f} periods/yr  "
                  f"annualisation x{iv.ann:.2f}")
        return 0

    if not args.symbols:
        print("give --symbols (a universe name or a list), or --catalogue")
        return 2

    syms = args.symbols[0] if len(args.symbols) == 1 and args.symbols[0] in UNIVERSES \
        else args.symbols
    iv = get_interval(args.interval)
    print(f"\n=== data audit — {iv.label} bars ({iv.periods_per_year:.0f}/yr, "
          f"annualisation x{iv.ann:.2f}) ===")
    rep, warnings = audit(syms, args.start, args.end, iv,
                          allow_download=not args.offline, prefer_fresh=args.refresh)
    with pd.option_context("display.width", 190, "display.max_columns", 20):
        print(rep.to_string(index=False))

    n_sym = int((rep["status"] == "ok").sum()) if "status" in rep else len(rep)
    sp = search_space(n_sym, 1, args.methods, args.targets)
    print(f"\n  search space if you run {args.methods} methods x {args.targets} targets "
          f"on these {n_sym} symbols: {sp['comparisons']} comparisons, "
          f"deflated bar |t| > {sp['deflated_bar']:.2f}")
    print("  (add a second interval and that bar rises — count it before you run, not after)")

    if args.panel:
        from msl.data.panel import cross_asset_features, load_panel
        print(f"\n=== cross-asset panel (how={args.how}) ===")
        p = load_panel(syms, args.start, args.end, iv, how=args.how,
                       allow_download=not args.offline, prefer_fresh=args.refresh)
        print(f"  {p}")
        print(p.coverage.to_string(index=False))
        caf = cross_asset_features(p)
        ready = caf.dropna()
        print(f"\n  cross-asset features: {', '.join(caf.columns)}")
        print(f"  usable from {ready.index.min().date() if len(ready) else 'n/a'} "
              f"({len(ready)} of {len(caf)} bars once windows fill)")
        if len(ready):
            print(ready.tail(3).round(4).to_string())
        warnings = list(warnings) + p.notes
    if warnings:
        print("\n  WARNINGS")
        for w in warnings:
            print(f"    - {w}")
    else:
        print("\n  no warnings: registered, aligned, single provenance.")
    return 0


def cmd_redundancy(args: argparse.Namespace) -> int:
    """Are the specialists independent dimensions, or independent noise?

    Effective-n alone cannot answer this: noisy measurements of one dimension decorrelate
    exactly like genuinely different dimensions. So the panel is split into what the
    specialists agree on and where each departs from that, and both are scored.
    """
    import pandas as pd

    from msl.diagnostics.redundancy import redundancy_gate, verdict

    res = pd.read_parquet(args.results)
    if args.start:
        res = res[res["date"] >= args.start]
    methods = [m for m in sorted(res["method"].unique()) if m not in BASELINE_METHODS]
    symbols = args.symbols or sorted(res["symbol"].unique())

    print("=== specialist redundancy gate ===")
    print(f"  {len(methods)} specialists: {', '.join(methods)}")
    frames = []
    for sym in symbols:
        sig = (res[(res.symbol == sym) & (res.method.isin(methods))]
               .pivot_table(index="date", columns="method", values="score"))
        try:
            px = load_prices(sym, args.start, allow_download=not args.offline)
        except Exception as exc:
            print(f"  {sym}: skipped ({exc})")
            continue
        feats = build_features(px)
        gate, dec = redundancy_gate(sig.reindex(feats.index), feats, horizon=args.horizon)
        if gate.empty:
            print(f"  {sym}: too few scored rows")
            continue
        gate.insert(0, "symbol", sym)
        frames.append(gate)
        print(f"  {sym:<8} effective_n = {dec.effective_n:.2f} of {sig.shape[1]}"
              f"   (a count, not evidence — see msl.diagnostics.redundancy)")
        for n in dec.notes:
            if "negative loading" in n:
                print(f"    [warn] {n}")
    if not frames:
        return 1

    g = pd.concat(frames, ignore_index=True)
    one = g[g.n_features == 1]
    print("\n  t-statistic vs a volatility-only baseline (negative = the feature helped)")
    print(one.pivot_table(index="variant", columns=["symbol", "target"],
                          values="t_stat").round(2).to_string())
    v = verdict(g)
    print(f"\n  {v['comparisons']} comparisons -> deflated bar |t| > {v['deflated_bar']:.2f}")
    print(f"  consensus clears it {v['consensus_wins']}x (best t = {v['best_consensus_t']:.2f})")
    print(f"  deviations clear it {v['idiosyncratic_wins']}x (best t = {v['best_idio_t']:.2f})")
    print(f"\n  VERDICT: {v['verdict']}")
    if args.out:
        g.to_parquet(args.out)
        print(f"\n  wrote {args.out}")
    return 0


def cmd_list(_: argparse.Namespace) -> int:
    print("registered estimators:")
    for n in list_estimators():
        print(f"  {n}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="msl", description="market-state-lab")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run an experiment config")
    r.add_argument("-c", "--config", required=True)
    r.add_argument("--offline", action="store_true", help="never download; committed CSVs only")
    r.add_argument("--refresh", action="store_true",
                   help="skip committed CSVs and download, for a common date range")
    r.set_defaults(func=cmd_run)

    v = sub.add_parser("recovery", help="tier-1 recovery scoring on synthetic markets")
    v.add_argument("--seeds", type=int, default=8, help="number of simulated markets")
    v.add_argument("--n", type=int, default=2500, help="days per simulated market")
    v.add_argument("--methods", nargs="*", default=None, help="defaults to all registered")
    v.add_argument("--out", default="results/recovery")
    v.set_defaults(func=cmd_recovery)

    s = sub.add_parser("state", help="current state per ticker (a view, not a signal)")
    s.add_argument("-c", "--config", required=True)
    s.add_argument("--as-of", default=None, help="report as of this date (YYYY-MM-DD)")
    s.add_argument("--offline", action="store_true", help="never download; committed CSVs only")
    s.add_argument("--refresh", action="store_true",
                   help="skip committed CSVs and download, so all symbols share an as-of date")
    s.set_defaults(func=cmd_state)

    d = sub.add_parser("decision", help="tier-4 decision value vs a volatility-only benchmark")
    d.add_argument("-c", "--config", required=True)
    d.add_argument("--horizon", type=int, default=5, help="forecast horizon in days")
    d.add_argument("--no-control", action="store_true",
                   help="do NOT add the estimator's flip rate to the baseline (less conservative)")
    d.add_argument("--common-window", action="store_true",
                   help="score every method on the intersection of usable dates, so the "
                        "cross-method ranking is like-for-like rather than period-dependent")
    d.add_argument("--offline", action="store_true", help="never download; committed CSVs only")
    d.add_argument("--refresh", action="store_true", help="skip committed CSVs and download")
    d.set_defaults(func=cmd_decision)

    da = sub.add_parser("data", help="inspect symbols, intervals and data quality before a run")
    da.add_argument("--symbols", nargs="*", default=None,
                    help="a universe name (indices, mixed, ...) or an explicit symbol list")
    da.add_argument("--interval", default="1d", help="bar interval: 1d or 1wk (default 1d)")
    da.add_argument("--start", default=None)
    da.add_argument("--end", default=None)
    da.add_argument("--catalogue", action="store_true",
                    help="list every known symbol, universe and interval, then exit")
    da.add_argument("--methods", type=int, default=8,
                    help="methods you intend to run, for the search-space estimate")
    da.add_argument("--targets", type=int, default=3, help="targets, for the search-space estimate")
    da.add_argument("--panel", action="store_true",
                    help="also build the aligned cross-asset panel and show what alignment cost")
    da.add_argument("--how", default="intersect", choices=["intersect", "union"],
                    help="panel alignment: intersect (shared dates) or union (NaN, never padded)")
    da.add_argument("--offline", action="store_true", help="never download")
    da.add_argument("--refresh", action="store_true", help="skip committed CSVs and download")
    da.set_defaults(func=cmd_data)

    rd = sub.add_parser("redundancy",
                        help="are the specialists independent dimensions, or independent noise?")
    rd.add_argument("--results", default="results/trend_mixed.parquet",
                    help="tidy sweep output holding per-method scores")
    rd.add_argument("--symbols", nargs="*", help="default: every symbol in the results")
    rd.add_argument("--start", default="2015-01-01")
    rd.add_argument("--horizon", type=int, default=5)
    rd.add_argument("--offline", action="store_true", help="never download")
    rd.add_argument("--out", help="write the full grid to this parquet path")
    rd.set_defaults(func=cmd_redundancy)

    sub.add_parser("list", help="list registered estimators").set_defaults(func=cmd_list)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
