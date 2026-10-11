"""Explanation of one decision: factors, contributions, alternative, confidence, limits.

The explanation does not depend on the strategy that decided: every decision is read with the WSPT grid
(score = w / (travel + duration), w = N × (1 + α·sensitive) × (1 + β·wait_h)):
    - raw factors + contribution of each one (decomposition of log(score));
    - alternative: the best OTHER incident this crew could have taken;
    - delta_customer_hours: exact cost (Smith's exchange argument for one crew) of doing the alternative first;
      > 0 ⇒ our choice avoids that many customer-hours;
    - next_best_crew: the next free crew for this incident, and how much later it would arrive;
    - confidence (0–1): score margin between the choice and the alternative;
    - limits: assumptions that weigh on this decision.
Every number here is computed — text generators (templates, LLM) may only quote them, never invent new ones.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

from copilot.models import Crew, Incident
from copilot.travel import TravelMin

ALWAYS_LIMITS = [
    "estimated travel: straight line × 1.3, 55 km/h in the plain, 35 km/h in the mountains",
    "nominal repair duration (2h/4h/8h): the actual duration may differ",
]


def weight(inc: Incident, t: int, params: dict) -> tuple[float, float, float]:
    """(w, sensitive_factor, wait_factor)."""
    alpha = float(params.get("alpha_sensitive", 5.0))
    beta = float(params.get("beta_wait_per_h", 0.1))
    f_s = 1 + alpha * bool(inc.sensitive)
    f_w = 1 + beta * max(0, t - inc.t_reported) / 60
    return inc.customers * f_s * f_w, f_s, f_w


def score(inc: Incident, travel: float, t: int, params: dict) -> float:
    w, _, _ = weight(inc, t, params)
    return w / (travel + inc.nominal_duration)


def explain(crew: Crew, incident: Incident, t: int, params: dict, travel_min: TravelMin,
            waiting: Sequence[Incident], free_crews: Sequence[Crew]) -> dict:
    from copilot.simulator import is_feasible

    tr = travel_min(crew.lat, crew.lon, incident.lat, incident.lon)
    w, f_s, f_w = weight(incident, t, params)
    sc = w / (tr + incident.nominal_duration)

    # contributions: log(score) = log N + log f_s + log f_w − log(travel + duration)
    total_time = tr + incident.nominal_duration
    terms = {
        "customers": math.log(max(incident.customers, 1)),
        "sensitive": math.log(f_s),
        "wait": math.log(f_w),
        "travel": -math.log(total_time) * (tr / total_time) if total_time else 0.0,
        "duration": -math.log(total_time) * (incident.nominal_duration / total_time) if total_time else 0.0,
    }
    norm = sum(abs(v) for v in terms.values()) or 1.0
    contributions = {k: round(100 * v / norm) for k, v in terms.items()}

    # alternative: best other feasible incident for this crew
    alt, alt_sc, alt_tr = None, -1.0, 0.0
    for j in waiting:
        if j.incident_id == incident.incident_id:
            continue
        trj = travel_min(crew.lat, crew.lon, j.lat, j.lon)
        if not is_feasible(crew, j, trj, t, params):
            continue
        s = score(j, trj, t, params)
        if s > alt_sc:
            alt, alt_sc, alt_tr = j, s, trj
    alternative = None
    if alt is not None:
        # Smith exchange: doing alt then i, instead of i then alt (same crew)
        delta_min = (incident.customers * (alt_tr + alt.nominal_duration)
                     - alt.customers * (tr + incident.nominal_duration))
        alternative = {
            "incident_id": alt.incident_id, "score": round(alt_sc, 3), "customers": alt.customers,
            "sensitive": bool(alt.sensitive), "travel_min": round(alt_tr, 1), "duration_min": alt.nominal_duration,
            "delta_customer_hours": round(delta_min / 60, 1),
        }

    # next best free crew for this incident
    other = None
    for c in free_crews:
        if c.crew_id == crew.crew_id:
            continue
        trc = travel_min(c.lat, c.lon, incident.lat, incident.lon)
        if is_feasible(c, incident, trc, t, params) and (other is None or trc < other[1]):
            other = (c, trc)
    next_best = None if other is None else {"crew_id": other[0].crew_id, "delay_min": round(other[1] - tr, 1)}

    if alternative is None:
        confidence = 1.0
    elif sc >= alt_sc:
        confidence = round(0.5 + 0.5 * max(0.0, min(1.0, (sc - alt_sc) / sc)), 2) if sc > 0 else 0.0
    else:
        confidence = round(0.5 * sc / alt_sc, 2)

    limits = list(ALWAYS_LIMITS)
    if incident.lat < float(params.get("travel", {}).get("mountain_lat_threshold", 43.05)):
        limits.append("destination in the mountains: travel time more uncertain")
    if incident.nominal_duration >= 480:
        limits.append("long repair (8h): the crew will be tied up for a long time")

    return {
        "score": round(sc, 3),
        "factors": {
            "customers": incident.customers, "sensitive": bool(incident.sensitive),
            "wait_min": int(t - incident.t_reported), "travel_min": round(tr, 1),
            "duration_min": incident.nominal_duration, "sensitive_weight": round(f_s, 2),
            "wait_weight": round(f_w, 2),
        },
        "contributions_pct": contributions,
        "alternative": alternative,
        "next_best_crew": next_best,
        "confidence": confidence,
        "limits": limits,
    }
