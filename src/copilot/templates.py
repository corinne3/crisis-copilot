"""Template texts (instant, no LLM, never an invented number): decision card + situation briefing.

Every number comes from the explanation JSON or the KPIs. The local LLM (copilot.ai.llm) may only rephrase
these texts, and its output is rejected if it contains a number that is not in the source (number guard).
"""
from __future__ import annotations


def hm(minutes: float) -> str:
    m = int(round(minutes))
    return f"{m // 60}h{m % 60:02d}" if m >= 60 else f"{m} min"


def decision_card(crew_id: str, incident_id: str, expl: dict) -> str:
    """2 to 4 operational sentences from the JSON produced by copilot.explanation.explain()."""
    f = expl.get("factors", {})
    if not f:
        if expl.get("forced") or expl.get("origin") == "human":
            return f"Crew {crew_id} sent to {incident_id} (manager's decision)."
        return f"Crew {crew_id} sent to {incident_id}."
    reasons = [f"{f['customers']} customers off supply"]
    if f.get("sensitive"):
        reasons.append("a sensitive customer is affected")
    if f.get("wait_min", 0) >= 60:
        reasons.append(f"waiting for {hm(f['wait_min'])}")
    text = (f"Crew {crew_id} sent to {incident_id}: {', '.join(reasons)}; "
            f"{hm(f['travel_min'])} of driving, {hm(f['duration_min'])} of repair.")
    alt = expl.get("alternative")
    if alt:
        d = alt["delta_customer_hours"]
        if d > 0:
            text += (f" Starting with {alt['incident_id']} instead ({alt['customers']} customers) "
                     f"would cost {d:.0f} more customer-hours.")
        elif d < 0:
            text += (f" ⚠️ Starting with {alt['incident_id']} ({alt['customers']} customers) would have saved "
                     f"{-d:.0f} customer-hours: worth discussing.")
    nxt = expl.get("next_best_crew")
    if nxt and nxt["delay_min"] > 0:
        text += f" The next closest crew ({nxt['crew_id']}) would arrive {hm(nxt['delay_min'])} later."
    text += f" Confidence: {expl.get('confidence', 0):.0%}."
    return text


def situation_briefing(t_min: int, row: dict, kpi: dict | None = None, sensitive_waiting: int = 0,
                       hotspots: list[str] | None = None) -> str:
    """Situation briefing for the prefecture at time t (row = one line of SimResult.series)."""
    text = (f"Situation at H+{hm(t_min)}: {row['customers_off']} customers without power, "
            f"{row['incidents_waiting']} incidents waiting for a crew. "
            f"{row['crews_busy']} crews on the field, {row['crews_resting']} resting (mandatory rest or break).")
    if sensitive_waiting:
        text += f" {sensitive_waiting} substations with sensitive customers waiting: they are prioritised."
    if hotspots:
        text += f" Areas under pressure: {', '.join(hotspots)}."
    if kpi:
        text += f" Estimated end of restoration: H+{kpi['end_h']:.0f}."
    return text
