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

from msl.data.loaders import load_universe
from msl.engine.walkforward import run_sweep, state_summary
from msl.estimators.base import list_estimators


def _write(df: pd.DataFrame, base: Path) -> Path:
    base.parent.mkdir(parents=True, exist_ok=True)
    try:
        path = base.with_suffix(".parquet")
        df.to_parquet(path, index=False)
    except Exception:            # pyarrow not installed - CSV is a fine fallback
        path = base.with_suffix(".csv")
        df.to_csv(path, index=False)
    return path


def cmd_run(args: argparse.Namespace) -> int:
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    eng = cfg.get("engine", {}) or {}

    print(f"[{cfg.get('name', 'run')}] loading prices…")
    prices = load_universe(
        cfg["universe"], cfg.get("start"), cfg.get("end"), allow_download=not args.offline
    )
    print(f"  loaded {len(prices)} symbol(s): {', '.join(prices)}\n")

    print("running sweep…")
    results = run_sweep(
        prices,
        cfg["methods"],
        min_train=eng.get("min_train", 750),
        refit_every=eng.get("refit_every", 63),
        embargo=eng.get("embargo", 5),
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

    cols = ["method", "balanced_accuracy", "ari", "brier",
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

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    prices = load_universe(
        cfg["universe"], cfg.get("start"), cfg.get("end"), allow_download=not args.offline
    )
    methods = [m for m in cfg["methods"] if m != "always_range"]
    rep = state_report(prices, methods, as_of=args.as_of)

    with pd.option_context("display.width", 160, "display.max_columns", 30):
        print("\n" + rep.to_string(index=False))
    print("\nRead as a filtered state estimate with uncertainty — not a trend confirmation.")
    print("`disagreement` rising is the signal to attend to: that is where the value of")
    print("state information concentrates.")
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
    s.set_defaults(func=cmd_state)

    sub.add_parser("list", help="list registered estimators").set_defaults(func=cmd_list)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
