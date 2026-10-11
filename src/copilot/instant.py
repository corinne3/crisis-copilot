"""State of the crisis at time t, rebuilt from a SimResult (no re-simulation). Used by the map and the briefing."""
from __future__ import annotations

import numpy as np
import pandas as pd

from copilot.models import SimResult

STATES = ("upcoming", "waiting", "in repair", "restored")


def incidents_at(res: SimResult, t: float) -> pd.DataFrame:
    """Incidents with a `state` column at time t (minutes)."""
    inc = res.incidents.copy()
    ass = inc["t_assigned"].fillna(np.inf)
    rst = inc["t_restored"].fillna(np.inf)
    inc["state"] = np.select([inc["t_reported"] > t, ass > t, rst > t], list(STATES[:3]), STATES[3])
    inc["wait_h"] = ((np.minimum(ass, t) - inc["t_reported"]).clip(lower=0) / 60).round(1)
    return inc


def crews_at(res: SimResult, t: float) -> pd.DataFrame:
    """Position and activity of every crew at time t (linear interpolation while driving)."""
    rows = []
    for cid, g in res.activities.sort_values("t_start").groupby("crew_id"):
        now = g[(g["t_start"] <= t) & (g["t_end"] > t)]
        if not now.empty:
            a = now.iloc[-1]
            dur = max(a["t_end"] - a["t_start"], 1)
            f = (t - a["t_start"]) / dur if a["type"] in ("travel", "return") else 1.0
            rows.append({"crew_id": cid, "activity": a["type"],
                         "lat": a["lat_from"] + f * (a["lat_to"] - a["lat_from"]),
                         "lon": a["lon_from"] + f * (a["lon_to"] - a["lon_from"]),
                         "incident_id": a["incident_id"], "lat_target": a["lat_to"], "lon_target": a["lon_to"]})
            continue
        past = g[g["t_end"] <= t]
        ref = past.iloc[-1] if not past.empty else g.iloc[0]
        lat, lon = (ref["lat_to"], ref["lon_to"]) if not past.empty else (ref["lat_from"], ref["lon_from"])
        rows.append({"crew_id": cid, "activity": "available", "lat": lat, "lon": lon,
                     "incident_id": None, "lat_target": lat, "lon_target": lon})
    return pd.DataFrame(rows)


def series_row(res: SimResult, t: float) -> dict:
    """Last snapshot of the series before t (for the situation briefing)."""
    s = res.series[res.series["t"] <= t]
    return (s.iloc[-1] if not s.empty else res.series.iloc[0]).to_dict()
