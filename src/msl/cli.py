"""Command line: run a config, write tidy results, print the stability summary.

    msl list                             # registered estimators
    msl run -c configs/trend_indices.yaml
    msl run -c configs/trend_indices.yaml --offline   # committed CSVs only
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

    sub.add_parser("list", help="list registered estimators").set_defaults(func=cmd_list)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
