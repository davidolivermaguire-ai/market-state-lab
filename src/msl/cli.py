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

from msl.data.loaders import _repo_root, load_universe
from msl.engine.walkforward import run_sweep, state_summary
from msl.estimators.base import list_estimators


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

    sub.add_parser("list", help="list registered estimators").set_defaults(func=cmd_list)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
