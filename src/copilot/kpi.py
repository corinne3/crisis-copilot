"""Indicators computed from a SimResult.

    from copilot.kpi import compute_kpi
    k = compute_kpi(res)          # flat dict, ready for a DataFrame / st.metric

Main objective: customer_hours = Σ customers × outage hours (lower is better).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from copilot.models import SimResult

WORK_TYPES = ("travel", "repair")
NEW_SHIFT_TYPES = ("rest", "out_of_service", "standby")    # a new shift starts after these


def weighted_gini(values, weights=None) -> float:
    """Weighted Gini coefficient (0 = perfect equality, 1 = maximal inequality)."""
    x = np.asarray(values, dtype=float)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
    keep = ~np.isnan(x) & (w > 0)
    x, w = x[keep], w[keep]
    if len(x) == 0 or x.sum() == 0:
        return 0.0
    order = np.argsort(x)
    x, w = x[order], w[order]
    p = w / w.sum()
    lorenz = np.cumsum(x * w) / np.sum(x * w)
    prev = np.concatenate([[0.0], lorenz[:-1]])
    return float(1 - np.sum(p * (lorenz + prev)))


def customers_off_curve(res: SimResult) -> pd.DataFrame:
    s = res.series[["t", "customers_off"]].copy()
    s["t_h"] = s["t"] / 60
    return s


def shift_overruns(res: SimResult) -> int:
    """Number of work activities ending after the end of the 12h shift (must be 0 in the nominal case)."""
    shift = int(res.metadata.get("params", {}).get("shift_max_min", 720))
    n = 0
    for _, g in res.activities.sort_values(["crew_id", "t_start"]).groupby("crew_id"):
        start = 0
        for r in g.itertuples():
            if r.type in NEW_SHIFT_TYPES:
                start = r.t_end
            elif r.type in WORK_TYPES and r.t_end - start > shift:
                n += 1
    return n


def max_work_without_break_h(res: SimResult) -> float:
    """Longest stretch of work (travel + repair) without break or rest, in hours."""
    worst = 0
    for _, g in res.activities.sort_values(["crew_id", "t_start"]).groupby("crew_id"):
        run = 0
        for r in g.itertuples():
            if r.type in ("break", *NEW_SHIFT_TYPES):
                run = 0
            elif r.type in WORK_TYPES:
                run += r.t_end - r.t_start
                worst = max(worst, run)
    return round(worst / 60, 2)


def restoration_time_h(restored: pd.DataFrame, share: float) -> float:
    """Hour (since crisis start) when `share` of the customers cut off are restored."""
    if restored.empty:
        return 0.0
    r = restored.sort_values("t_restored")
    cum = r["customers"].cumsum() / r["customers"].sum()
    return float(r.loc[cum >= share - 1e-12, "t_restored"].iloc[0] / 60)


def compute_kpi(res: SimResult) -> dict:
    inc = res.incidents
    done = inc.dropna(subset=["t_restored"])
    T = (done["t_restored"] - done["t_reported"]) / 60            # outage hours per incident
    N = done["customers"]
    wait_h = (inc["t_assigned"] - inc["t_reported"]) / 60
    sens = inc["sensitive"].astype(bool)

    def pct_restored(h):
        return float(100 * N[T <= h].sum() / N.sum()) if N.sum() else 0.0

    if "town" in done and done["town"].notna().any():               # Gini of mean outage time per town
        by_town = done.assign(NT=T * N).groupby("town").agg(NT=("NT", "sum"), N=("customers", "sum"))
        gini = weighted_gini(by_town["NT"] / by_town["N"], by_town["N"])
    else:
        gini = float("nan")

    s = res.series
    step = int(res.metadata.get("params", {}).get("snapshot_step_min", 30))
    saturated = (s["crews_available"] == 0) & (s["incidents_waiting"] > 0)
    act = res.activities
    dur = act["t_end"] - act["t_start"]
    repair_time, travel_time = dur[act["type"] == "repair"].sum(), dur[act["type"] == "travel"].sum()

    return {
        "crisis": res.metadata.get("crisis"),
        "strategy": res.metadata.get("strategy"),
        "seed": res.metadata.get("seed"),
        "monte_carlo": res.metadata.get("monte_carlo"),
        "customer_hours": float((T * N).sum()),
        "sensitive_customer_hours": float((T * N)[done["sensitive"].astype(bool)].sum()),
        "pct_restored_2h": pct_restored(2),
        "pct_restored_4h": pct_restored(4),
        "pct_restored_8h": pct_restored(8),
        "wait_mean_h": float(wait_h.mean()),
        "wait_max_h": float(wait_h.max()),
        "wait_mean_sensitive_h": float(wait_h[sens].mean()) if sens.any() else 0.0,
        "wait_p90_sensitive_h": float(wait_h[sens].quantile(0.9)) if sens.any() else 0.0,
        "wait_max_sensitive_h": float(wait_h[sens].max()) if sens.any() else 0.0,
        "gini_towns": gini,
        "saturation_pct": float(100 * saturated.mean()),
        "peak_customers_off": int(s["customers_off"].max()),
        "end_h": float(res.metadata.get("t_end_min", s["t"].max()) / 60),
        "restored_95pct_h": restoration_time_h(done, 0.95),
        "travel_share_pct": float(100 * travel_time / (travel_time + repair_time)) if repair_time else 0.0,
        "shift_overruns": shift_overruns(res),
        "unresolved": int(res.metadata.get("unresolved", 0)),
        "n_breaks": int(res.metadata.get("n_breaks", 0)),
        "mean_fatigue": float(res.metadata.get("mean_fatigue", 1.0)),
        "max_work_without_break_h": max_work_without_break_h(res),
        "hours_without_active_crew": float(((s["crews_busy"] == 0) & (s["incidents_waiting"] > 0)).sum() * step / 60),
        "n_incidents": int(len(inc)),
    }


def train_test_split(crises: list[str], n_train: int = 80) -> tuple[list[str], list[str]]:
    """Crises 1..80 = training (tuning α, β, models), 81..100 = test (never seen while tuning)."""
    crises = sorted(crises)
    return crises[:n_train], crises[n_train:]
