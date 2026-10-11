"""Brick ① — Early forecast: from the first hour(s) of incident reports, how big will this crisis get?

At H+τ (30 min … 3 h after the first report) the model reads the arrival curve so far and predicts, with an
interval (quantiles 5 / 50 / 95 — its coverage measured on the test crises is shown next to it):
    - total_incidents   how many incidents this crisis will have in the end;
    - total_work_h      total repair work (crew-hours) — what the crews will have to absorb;
    - end_h             when the last customer will be restored, with the current organisation (WSPT,
                        63 crews, no rotation) — learned from simulations of the training crises.
Then deterministic, readable rules turn the forecast into recommendations (second shift / rotation,
reinforcement, department under pressure). The model recommends EARLY; the manager decides.

Why a gradient-boosting model and not a deep network: 80 training crises, a few features, CPU, < 2 s to train,
and each prediction is compared with a naive baseline so we only keep the model if it really helps.
Honest split: crises 1–80 train, 81–100 test (never seen).

    python -m copilot.ai.forecast              # train, evaluate on 81-100, save outputs/forecast_model.joblib
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("copilot.forecast")

TAUS = (30, 60, 90, 120, 180)
TARGETS = ("total_incidents", "total_work_h", "end_h")
QUANTILES = (0.05, 0.5, 0.95)      # nominal 90 % interval; measured coverage is reported
STORM = ("Abnormal stress from wind or rain storm", "Branch fall due to wind", "Lightning strike")
DEPARTMENTS = ("Ariège", "Gers", "Haute-Garonne")
FEATURES = ["tau_min", "n_so_far", "rate_last30", "acceleration", "customers_so_far", "sensitive_so_far",
            "work_h_so_far", "share_8h", "share_storm", "mean_customers"] + [f"share_{d}" for d in DEPARTMENTS]


# --------------------------------------------------------------------------- features
def snapshot_features(inc: pd.DataFrame, tau: int) -> dict:
    """Features of ONE crisis seen at H+τ (only incidents reported at t ≤ τ)."""
    seen = inc[inc["t_reported"] <= tau]
    n = len(seen)
    last = int((seen["t_reported"] > tau - 30).sum())
    avg30 = n / max(tau, 1) * 30
    f = {"tau_min": tau, "n_so_far": n, "rate_last30": last, "acceleration": last / avg30 if avg30 else 0.0,
         "customers_so_far": int(seen["customers"].sum()), "sensitive_so_far": int(seen["sensitive"].sum()),
         "work_h_so_far": float(seen["duration_min"].sum() / 60),
         "share_8h": float((seen["duration_min"] >= 480).mean()) if n else 0.0,
         "share_storm": float(seen["cause_en"].isin(STORM).mean()) if n else 0.0,
         "mean_customers": float(seen["customers"].mean()) if n else 0.0}
    for d in DEPARTMENTS:
        f[f"share_{d}"] = float((seen["department"] == d).mean()) if n else 0.0
    return f


def simulated_end_h(crises: list[str], cache: Path | None = None, n_jobs: int = -1) -> pd.Series:
    """end_h of each crisis with WSPT (nominal) — the 'current organisation' target. Cached in outputs/."""
    from copilot.benchmark import run
    from copilot.policies.wspt import WSPT

    done = pd.Series(dtype=float)
    if cache is not None and cache.exists():
        done = pd.read_csv(cache, index_col=0)["end_h"]
    todo = [c for c in crises if c not in done.index]
    if todo:
        kpi, _ = run(todo, {"WSPT": WSPT}, n_jobs=n_jobs)
        done = pd.concat([done, kpi.set_index("crisis")["end_h"]])
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            done.rename("end_h").to_frame().to_csv(cache)
    return done.loc[crises]


def build_dataset(incidents: pd.DataFrame, crises: list[str], end_h: pd.Series | None = None,
                  taus=TAUS) -> pd.DataFrame:
    rows = []
    for c in crises:
        inc = incidents[incidents["crisis_id"] == c]
        targets = {"total_incidents": len(inc), "total_work_h": float(inc["duration_min"].sum() / 60),
                   "end_h": float(end_h.loc[c]) if end_h is not None else math.nan}
        for tau in taus:
            rows.append({"crisis": c, **snapshot_features(inc, tau), **targets})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- model
@dataclass
class ForecastModel:
    models: dict = field(default_factory=dict)            # (target, q) → fitted regressor
    baseline: dict = field(default_factory=dict)          # naive references learned on the training set
    metrics: pd.DataFrame | None = None

    def fit(self, train: pd.DataFrame, seed: int = 42) -> ForecastModel:
        from sklearn.ensemble import GradientBoostingRegressor

        for target in TARGETS:
            if train[target].isna().all():
                continue
            for q in QUANTILES:
                m = GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=150, max_depth=2,
                                              learning_rate=0.05, subsample=0.8, random_state=seed)
                self.models[(target, q)] = m.fit(train[FEATURES], train[target])
        # naive baseline: "what has arrived so far ÷ the usual fraction at this hour" (end_h: training mean)
        frac = (train["n_so_far"] / train["total_incidents"]).groupby(train["tau_min"]).mean()
        wfrac = (train["work_h_so_far"] / train["total_work_h"]).groupby(train["tau_min"]).mean()
        self.baseline = {"fraction": frac.to_dict(), "work_fraction": wfrac.to_dict(),
                         "end_h": float(train["end_h"].mean())}
        return self

    def predict(self, X: pd.DataFrame | dict) -> pd.DataFrame:
        """One row per snapshot: <target>_low, _p50, _high (quantiles re-sorted so low ≤ p50 ≤ high)."""
        X = pd.DataFrame([X]) if isinstance(X, dict) else X
        out = pd.DataFrame(index=X.index)
        for target in TARGETS:
            if (target, 0.5) not in self.models:
                continue
            qs = np.sort(np.column_stack([self.models[(target, q)].predict(X[FEATURES]) for q in QUANTILES]), axis=1)
            low = X["n_so_far"].to_numpy() if target == "total_incidents" else (
                X["work_h_so_far"].to_numpy() if target == "total_work_h" else X["tau_min"].to_numpy() / 60)
            qs = np.maximum(qs, low[:, None])                 # never below what has already happened
            for k, name in enumerate(("low", "p50", "high")):
                out[f"{target}_{name}"] = qs[:, k]
        return out

    def predict_baseline(self, X: pd.DataFrame) -> pd.DataFrame:
        f = X["tau_min"].map(self.baseline["fraction"])
        wf = X["tau_min"].map(self.baseline["work_fraction"])
        return pd.DataFrame({"total_incidents": X["n_so_far"] / f, "total_work_h": X["work_h_so_far"] / wf,
                             "end_h": self.baseline["end_h"]}, index=X.index)

    def evaluate(self, test: pd.DataFrame) -> pd.DataFrame:
        """MAE of the model vs the naive baseline, and measured coverage of the interval, per τ and target."""
        pred, base = self.predict(test), self.predict_baseline(test)
        rows = []
        for tau, idx in test.groupby("tau_min").groups.items():
            for target in TARGETS:
                if f"{target}_p50" not in pred:
                    continue
                y = test.loc[idx, target]
                mae_m = float((pred.loc[idx, f"{target}_p50"] - y).abs().mean())
                mae_b = float((base.loc[idx, target] - y).abs().mean())
                cover = float(((pred.loc[idx, f"{target}_low"] <= y) & (y <= pred.loc[idx, f"{target}_high"])).mean())
                rows.append({"tau_min": tau, "target": target, "mae_model": round(mae_m, 2),
                             "mae_baseline": round(mae_b, 2),
                             "gain_pct": round(100 * (1 - mae_m / mae_b), 1) if mae_b else 0.0,
                             "coverage_pct": round(100 * cover)})
        self.metrics = pd.DataFrame(rows)
        return self.metrics

    def save(self, path: Path) -> Path:
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)       # plain dict: loadable whatever module saved it
        joblib.dump({"models": self.models, "baseline": self.baseline, "metrics": self.metrics}, path)
        return path

    @staticmethod
    def load(path: Path) -> ForecastModel:
        import joblib

        d = joblib.load(path)
        return ForecastModel(d["models"], d["baseline"], d["metrics"])


def model_path() -> Path:
    from copilot.config import load_config, project_path

    return project_path(load_config()["paths"]["outputs"], "forecast_model.joblib")


def train_default(n_jobs: int = -1, folder=None) -> tuple[ForecastModel, pd.DataFrame]:
    """Train on crises 1–80, evaluate on 81–100, save. Returns (model, metrics)."""
    from copilot.kpi import train_test_split
    from copilot.models import clean_dir, list_crises, read_tables

    _, incidents = read_tables(str(clean_dir(folder)))
    crises = list_crises(folder)
    train_c, test_c = train_test_split(crises)
    end_h = simulated_end_h(crises, model_path().with_name("forecast_end_h.csv"), n_jobs=n_jobs)
    model = ForecastModel().fit(build_dataset(incidents, train_c, end_h))
    metrics = model.evaluate(build_dataset(incidents, test_c, end_h))
    model.save(model_path())
    return model, metrics


# --------------------------------------------------------------------------- recommendations
@dataclass
class Recommendation:
    level: str          # "info" | "warning" | "action"
    title: str
    why: str


def recommendations(pred: dict, seen: pd.DataFrame, crews_per_department: dict[str, int],
                    shift_h: float = 12.0, travel_overhead: float = 1.17,
                    target_shifts: int = 2) -> list[Recommendation]:
    """Readable rules on top of the forecast. Every number in the text comes from `pred` or the inputs.

    travel_overhead 1.17 ≈ driving adds ~15 % of crew time on top of repairs (benchmark, WSPT).
    target_shifts: the crisis should be absorbed within this many shifts per crew (2 ≈ 12h work, 11h rest, 12h work).
    """
    n_crews = sum(crews_per_department.values())
    capacity = n_crews * shift_h
    work50, work90 = pred["total_work_h_p50"] * travel_overhead, pred["total_work_h_high"] * travel_overhead
    shifts50 = work50 / capacity
    out = [Recommendation("info", f"Expected size: {pred['total_incidents_p50']:.0f} incidents "
                          f"({pred['total_incidents_low']:.0f}–{pred['total_incidents_high']:.0f})",
                          f"≈ {work50:.0f} crew-hours of work incl. driving, vs {capacity:.0f} crew-hours "
                          f"per 12h shift for {n_crews} crews ({shifts50:.1f} shifts).")]
    end50, end90 = pred.get("end_h_p50"), pred.get("end_h_high")
    if end50 is not None and (shifts50 > 1.0 or end50 > shift_h + 11):
        out.append(Recommendation(
            "action", "Plan a second shift NOW: stagger crews (rotation)",
            f"Forecast end of restoration H+{end50:.0f} (up to H+{end90:.0f}) with everyone starting together: "
            f"all crews would reach 12h and rest at the same time. Starting part of the crews later keeps "
            f"someone on the field — test it in the What-if page."))
    if work90 > target_shifts * capacity:
        extra = math.ceil((work90 - target_shifts * capacity) / (target_shifts * shift_h))
        out.append(Recommendation(
            "warning", f"Consider asking for {extra} reinforcement crews",
            f"Pessimistic case (high): {work90:.0f} crew-hours vs {target_shifts * capacity:.0f} available in "
            f"{target_shifts} shifts — the crisis would need a third shift."))
    if len(seen):
        work_dep = seen.groupby("department")["duration_min"].sum() / 60
        total = work_dep.sum()
        for d, crews in crews_per_department.items():
            share_work = float(work_dep.get(d, 0) / total) if total else 0.0
            share_crews = crews / n_crews
            if share_crews and share_work / share_crews >= 1.3 and share_work >= 0.2:
                out.append(Recommendation(
                    "warning", f"{d} is under pressure",
                    f"{share_work:.0%} of the work reported so far but {share_crews:.0%} of the crews: "
                    f"pre-position crews from the neighbouring department."))
    return out


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Brick 1 — early forecast of the crisis size")
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("copilot.simulator").setLevel(logging.ERROR)
    _, metrics = train_default(args.jobs)
    print("\n=== Test crises 81-100: model vs naive baseline (MAE, lower is better) ===")
    print(metrics.to_string(index=False))
    print(f"\nModel saved to {model_path()}")


if __name__ == "__main__":
    main()
