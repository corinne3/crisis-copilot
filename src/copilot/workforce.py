"""Workforce realism (from V2): meal breaks, fatigue and crew rotation — comparison and sensitivity.

    python -m copilot.workforce                              # crises 1-30, fatigue slopes 0 to 10 %/h, WSPT
    python -m copilot.workforce --crises 1-100 --slopes 0,0.05,0.1 --jobs 2

Outputs: outputs/break_sensitivity.csv and outputs/rotation_comparison.csv + printed summaries.

Key assumption (to be validated by the utility): beyond `threshold_h` hours of work without a break, every extra
hour lengthens repairs by `slope` (e.g. 0.05 = +5 %/h). Without real data, the result is shown for several
slopes: this is the sensitivity curve.

Step-by-step comparison (`steps`): each row adds ONE change to the previous one, so the effect of each change
can be read on its own: ① base → ② + fatigue → ③ + fixed break → ④ smart break → ⑤ + rotation.
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from copilot.benchmark import parse_crises, run
from copilot.config import load_config, project_path
from copilot.policies.wspt import resolve_strategy
from copilot.simulator import simulation_params

BREAK_MODES = {
    "no_break": {"enabled": False},
    "fixed_break": {"enabled": True, "mode": "fixed"},
    "smart_break": {"enabled": True, "mode": "smart"},
}

ROTATIONS = {
    "none": {"enabled": False},
    "2 groups, +4h": {"enabled": True, "groups": 2, "offset_min": 240},
    "2 groups, +6h": {"enabled": True, "groups": 2, "offset_min": 360},
    "3 groups, +4h": {"enabled": True, "groups": 3, "offset_min": 240},
}

STEPS = {
    "① base (no break, no fatigue)": ("no_break", 0.0, "none"),
    "② + fatigue 5 %/h": ("no_break", 0.05, "none"),
    "③ + fixed break after 4h": ("fixed_break", 0.05, "none"),
    "④ smart break instead": ("smart_break", 0.05, "none"),
    "⑤ + rotation 2 groups, +4h": ("smart_break", 0.05, "2 groups, +4h"),
}


def mode_params(mode: str, slope: float, rotation: str = "none", **overrides) -> dict:
    """Parameters for one break mode, one fatigue slope (enabled if > 0) and one rotation."""
    return simulation_params(breaks=BREAK_MODES[mode], fatigue={"enabled": slope > 0, "slope_per_h": slope},
                             rotation=ROTATIONS[rotation], explain=False, **overrides)


def compare_breaks(crises: list, slopes: list[float], strategy: str = "WSPT", n_jobs: int = -1,
                   rotation: str = "none") -> pd.DataFrame:
    cls = resolve_strategy(strategy)
    frames = []
    for slope in slopes:
        for mode in BREAK_MODES:
            kpi, _ = run(crises, {strategy: cls}, params=mode_params(mode, slope, rotation), n_jobs=n_jobs)
            kpi["mode"], kpi["fatigue_slope"], kpi["rotation"] = mode, slope, rotation
            frames.append(kpi)
    return pd.concat(frames, ignore_index=True)


def compare_rotations(crises: list, strategy: str = "WSPT", n_jobs: int = -1, slope: float = 0.05,
                      mode: str = "fixed_break") -> pd.DataFrame:
    """Rotation organisations compared (same break, same fatigue)."""
    cls = resolve_strategy(strategy)
    frames = []
    for rot in ROTATIONS:
        kpi, _ = run(crises, {strategy: cls}, params=mode_params(mode, slope, rot), n_jobs=n_jobs)
        kpi["rotation"], kpi["mode"], kpi["fatigue_slope"] = rot, mode, slope
        frames.append(kpi)
    return pd.concat(frames, ignore_index=True)


def steps(crises: list, strategy: str = "WSPT", n_jobs: int = -1) -> pd.DataFrame:
    """Step-by-step comparison table (one change per row), mean over the crises."""
    cls = resolve_strategy(strategy)
    rows = []
    for label, (mode, slope, rot) in STEPS.items():
        kpi, _ = run(crises, {strategy: cls}, params=mode_params(mode, slope, rot), n_jobs=n_jobs)
        rows.append({"step": label, **kpi.mean(numeric_only=True).to_dict()})
    return pd.DataFrame(rows).set_index("step")


def summarise_breaks(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["customer_hours", "wait_max_sensitive_h", "n_breaks", "max_work_without_break_h", "mean_fatigue",
            "shift_overruns"]
    r = df.groupby(["fatigue_slope", "mode"])[cols].mean().round(2)
    base = r.xs("no_break", level="mode")["customer_hours"]
    r["gain_vs_no_break_pct"] = [round(100 * (1 - v / base.loc[p]), 2) for (p, _), v in r["customer_hours"].items()]
    return r


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Meal breaks, fatigue and crew rotation")
    ap.add_argument("--crises", default="1-30")
    ap.add_argument("--slopes", default="0,0.02,0.05,0.08,0.1")
    ap.add_argument("--strategy", default="WSPT")
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("copilot.simulator").setLevel(logging.ERROR)
    crises = parse_crises(args.crises)
    folder = project_path(load_config()["paths"]["outputs"])
    folder.mkdir(parents=True, exist_ok=True)

    df = compare_breaks(crises, [float(x) for x in args.slopes.split(",")], args.strategy, args.jobs)
    df.to_csv(folder / "break_sensitivity.csv", index=False)
    print("\n=== Meal breaks: mean per fatigue slope and mode ===")
    print(summarise_breaks(df).to_string())
    rot = compare_rotations(crises, args.strategy, args.jobs)
    rot.to_csv(folder / "rotation_comparison.csv", index=False)
    cols = ["customer_hours", "wait_max_sensitive_h", "hours_without_active_crew", "end_h", "shift_overruns"]
    print("\n=== Crew rotation (fixed break, fatigue 5 %/h) ===")
    print(rot.groupby("rotation", sort=False)[cols].mean().round(2).to_string())
    print(f"\nFiles written: {folder / 'break_sensitivity.csv'} and {folder / 'rotation_comparison.csv'}")


if __name__ == "__main__":
    main()
