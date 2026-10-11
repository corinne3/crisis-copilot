"""Benchmark: crises × strategies (× Monte Carlo seeds), in parallel with joblib.

Command line:
    python -m copilot.benchmark                               # 100 crises, R1 + R2 + WSPT
    python -m copilot.benchmark --crises 1-20 --strategies R1,R2
    python -m copilot.benchmark --monte-carlo 5               # 5 draws per crisis and strategy
    python -m copilot.benchmark --strategies R2,my_module:MyPolicy

Outputs (outputs/):
    benchmark_kpi.csv / .parquet   one row per (crisis, strategy, seed)
    benchmark_series.parquet       customers-off curves for the app
    benchmark_summary.csv          mean per strategy + gain vs references
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from copilot.config import load_config, project_path
from copilot.kpi import compute_kpi
from copilot.policies.base import Policy
from copilot.policies.wspt import resolve_strategy

log = logging.getLogger("copilot.benchmark")

SUMMARY_COLS = ["customer_hours", "pct_restored_4h", "wait_mean_sensitive_h", "wait_p90_sensitive_h",
                "wait_max_sensitive_h", "gini_towns", "saturation_pct", "restored_95pct_h", "end_h",
                "shift_overruns"]


def parse_crises(spec: str) -> list[int]:
    """'1-100', '27', '1,5,27-30'."""
    out: list[int] = []
    for part in str(spec).split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


def run_one(crisis, name: str, cls: type[Policy], params: dict, seed: int | None,
            scenario: dict | None = None, options: dict | None = None) -> tuple[dict, pd.DataFrame]:
    from copilot.simulator import simulate_crisis

    p = dict(params)
    p.setdefault("explain", False)             # explanations not needed for the benchmark (×10 faster)
    if seed is not None:
        p["seed"] = seed
        p["monte_carlo"] = {**p.get("monte_carlo", {}), "enabled": True}
    res = simulate_crisis(crisis, cls(), p, scenario=scenario, **dict(options or {}))
    k = compute_kpi(res)
    k["strategy"], k["seed"], k["compute_s"] = name, seed, res.metadata["compute_s"]
    series = res.series[["t", "customers_off", "crews_available", "incidents_waiting"]].copy()
    series["crisis"], series["strategy"], series["seed"] = k["crisis"], name, -1 if seed is None else seed
    return k, series


def run(crises: list, strategies: dict[str, type[Policy]], params: dict | None = None,
        seeds: list[int | None] | None = None, scenario: dict | None = None, options: dict | None = None,
        n_jobs: int = -1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (kpi, series). seeds=None → one nominal simulation (no Monte Carlo)."""
    from copilot.simulator import simulation_params

    params = params or simulation_params()
    seeds = seeds or [None]
    valid = {}                                    # pre-flight: a crashing strategy is dropped, not the benchmark
    for name, cls in strategies.items():
        try:
            run_one(crises[0], name, cls, params, None, scenario, options)
            valid[name] = cls
        except Exception as e:                    # noqa: BLE001
            log.error("strategy %s dropped (error on crisis %s): %r", name, crises[0], e)
    tasks = [(c, n, cls, params, s, scenario, options) for c in crises for n, cls in valid.items() for s in seeds]
    clock = time.perf_counter()
    if n_jobs == 1 or len(tasks) < 8:
        results = [run_one(*t) for t in tasks]
    else:
        try:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=n_jobs)(delayed(run_one)(*t) for t in tasks)
        except ImportError:
            results = [run_one(*t) for t in tasks]
    log.info("%s simulations in %.1f s", len(tasks), time.perf_counter() - clock)
    return (pd.DataFrame([r[0] for r in results]),
            pd.concat([r[1] for r in results], ignore_index=True) if results else pd.DataFrame())


def summarise(kpi: pd.DataFrame) -> pd.DataFrame:
    """Mean per strategy + % customer-hours gain vs each reference + number of crises won."""
    summary = kpi.groupby("strategy")[[c for c in SUMMARY_COLS if c in kpi]].mean()
    per_crisis = kpi.groupby(["crisis", "strategy"])["customer_hours"].mean().unstack()
    for ref in [c for c in per_crisis.columns if c in ("R1", "R2")]:
        summary[f"gain_vs_{ref}_pct"] = [float(100 * (1 - per_crisis[s] / per_crisis[ref]).mean())
                                         for s in summary.index]
        summary[f"crises_won_vs_{ref}"] = [int((per_crisis[s] < per_crisis[ref]).sum()) for s in summary.index]
    return summary.round(2)


def save(kpi: pd.DataFrame, series: pd.DataFrame, folder: Path, suffix: str = "") -> pd.DataFrame:
    folder.mkdir(parents=True, exist_ok=True)
    kpi.to_csv(folder / f"benchmark_kpi{suffix}.csv", index=False)
    summary = summarise(kpi)
    summary.to_csv(folder / f"benchmark_summary{suffix}.csv")
    try:
        kpi.to_parquet(folder / f"benchmark_kpi{suffix}.parquet", index=False)
        series.to_parquet(folder / f"benchmark_series{suffix}.parquet", index=False)
    except ImportError:
        series.to_csv(folder / f"benchmark_series{suffix}.csv", index=False)
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Benchmark of dispatch strategies over the crises")
    ap.add_argument("--crises", default="1-100")
    ap.add_argument("--strategies", default="R1,R2,WSPT")
    ap.add_argument("--monte-carlo", type=int, default=0, help="number of random draws (0 = nominal)")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--suffix", default="")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("copilot.simulator").setLevel(logging.ERROR)

    strategies = {n.split(":")[-1]: resolve_strategy(n) for n in args.strategies.split(",")}
    seeds = list(range(1, args.monte_carlo + 1)) if args.monte_carlo else None
    kpi, series = run(parse_crises(args.crises), strategies, seeds=seeds, n_jobs=args.jobs)
    folder = project_path(load_config()["paths"]["outputs"])
    summary = save(kpi, series, folder, args.suffix)
    print("\n=== Summary (mean per strategy) ===")
    print(summary.T.to_string())
    print(f"\nFiles written to {folder}")


if __name__ == "__main__":
    main()
