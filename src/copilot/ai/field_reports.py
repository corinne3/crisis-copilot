"""Brick ② — Understand field reports: free-text messages from crews → validated, structured updates.

Crews write short, messy messages ("INC_000123 worse than expected, 3 more hours", "tree on the road near Foix,
can't reach INC_000045", "C2 done early"). The small local LLM extracts a JSON update; then DETERMINISTIC checks
decide whether it can be used:
    - the incident / crew must exist in the current crisis;
    - the type must be one of the allowed types; extra minutes in a plausible range;
    - every number in the update must come from the message (minutes may be written as hours: 3 h = 180 min);
    - safety hazards and support requests are NEVER automated: always "manager must handle".
Anything that fails a check is shown as "needs review" — nothing reaches the plan without the manager's tick.

A rule-based extractor (regex + keywords) does the same job without AI; both are measured on a labelled set of
synthetic messages, so the app shows how much the LLM really adds (frugality: keep it only if it helps).
"""
from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass

import numpy as np

log = logging.getLogger("copilot.field_reports")

TYPES = ("duration_update", "access_blocked", "restored", "need_support", "safety_hazard", "info")
HUMAN_ONLY = ("need_support", "safety_hazard")
MAX_EXTRA_MIN, MIN_EXTRA_MIN = 960, -480


@dataclass
class Update:
    type: str = "info"
    incident_id: str | None = None
    crew_id: str | None = None
    extra_minutes: int | None = None
    summary: str = ""
    source: str = "rules"           # rules | llm
    status: str = "ok"              # ok | needs_review | manager_must_handle
    issues: tuple = ()

    def as_dict(self) -> dict:
        d = asdict(self)
        d["issues"] = "; ".join(self.issues)
        return d


# --------------------------------------------------------------------------- rule-based extractor
_INC = re.compile(r"\bINC[\s_-]?0*(\d{1,6})\b", re.I)
_CREW = re.compile(r"\bSITE[\s_-]?0*(\d{1,2})[\s_-]?C(\d{1,2})\b", re.I)
_HM = re.compile(r"\b(\d{1,2})\s*h\s*(\d{2})\b", re.I)
_H = re.compile(r"\b(\d{1,2}(?:[.,]\d)?)\s*(?:[a-zé]+\s+){0,2}?(?:h|hr|hrs|hour|hours|heures?)\b", re.I)
_M = re.compile(r"\b(\d{1,3})\s*(?:[a-zé]+\s+){0,2}?(?:min|mins|minutes|mn)\b", re.I)
KEYWORDS = {
    "safety_hazard": ["danger", "live wire", "fire", "injur", "unsafe", "safety", "câble au sol", "electrocut"],
    "restored": ["fixed", "restored", "done", "back on", "power is back", "rétabli", "finished", "repaired"],
    "access_blocked": ["road", "blocked", "closed", "flood", "can't reach", "cannot reach", "no access",
                       "access", "route coupée", "snow"],
    "need_support": ["need another", "second crew", "backup", "need help", "reinforcement", "extra crew",
                     "crane", "need a truck"],
}
EARLIER = ["earlier", "less", "ahead", "sooner", "faster than", "quicker than"]


def incident_id(num: str) -> str:
    return f"INC_{int(num):06d}"


def crew_id(site: str, k: str) -> str:
    return f"SITE_{int(site):02d}_C{int(k)}"


_WORDS = {"half an hour": "30 min", "half a day": "6 hours", "an hour": "1 hour", "one": "1", "two": "2",
          "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "ten": "10"}


def minutes_in(text: str) -> int | None:
    """Total minutes written in a message ("3 more hours", "1h30", "45 min", "two hours"), else None."""
    text = text.lower()
    for w, d in _WORDS.items():
        text = re.sub(rf"\b{w}\b", d, text)
    m = _HM.search(text)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    total, found = 0, False
    for m in _H.finditer(text):
        total += int(round(float(m.group(1).replace(",", ".")) * 60))
        found = True
    for m in _M.finditer(text):
        total += int(m.group(1))
        found = True
    return total if found else None


def extract_rules(message: str) -> Update:
    low = message.lower()
    u = Update(source="rules", summary=message.strip()[:120])
    m = _INC.search(message)
    u.incident_id = incident_id(m.group(1)) if m else None
    m = _CREW.search(message)
    u.crew_id = crew_id(m.group(1), m.group(2)) if m else None
    kind = next((k for k, kws in KEYWORDS.items() if any(w in low for w in kws)), None)
    minutes = minutes_in(message)
    if kind is None and minutes is not None:
        kind = "duration_update"
    u.type = kind or "info"
    if u.type == "duration_update" and minutes is not None:
        u.extra_minutes = -minutes if any(w in low for w in EARLIER) else minutes
    return u


# --------------------------------------------------------------------------- LLM extractor
SYSTEM = f"""You read short messages sent by repair crews during a power-grid crisis and extract ONE update.
Answer ONLY with JSON: {{"type": one of {list(TYPES)}, "incident_id": "INC_000123" or null,
"crew_id": "SITE_04_C2" or null, "extra_minutes": integer or null, "summary": short English sentence}}.
Rules: extra_minutes only for type duration_update (positive = repair takes longer, negative = shorter),
converted to minutes (2h = 120). Incident ids have 6 digits (INC_45 → INC_000045). Never invent an id or a number.
Live wires, fire, injuries → safety_hazard. Road/tree/flood preventing access → access_blocked."""

EXAMPLES = """Message: INC_000123 worse than expected, need 3 more hours
JSON: {"type": "duration_update", "incident_id": "INC_000123", "crew_id": null, "extra_minutes": 180, "summary": "Repair of INC_000123 needs 3 more hours."}
Message: SITE_04_C2: tree across the road, cannot reach INC 77
JSON: {"type": "access_blocked", "incident_id": "INC_000077", "crew_id": "SITE_04_C2", "extra_minutes": null, "summary": "Road blocked by a tree, INC_000077 unreachable."}
"""


def extract_llm(message: str, client) -> Update | None:
    data = client.generate_json(f"{EXAMPLES}\nMessage: {message}\nJSON:", SYSTEM, task="field_report",
                                max_tokens=200) if client is not None else None
    if not data:
        return None
    u = Update(source="llm")
    u.type = str(data.get("type") or "info")
    u.incident_id = data.get("incident_id") or None
    u.crew_id = data.get("crew_id") or None
    x = data.get("extra_minutes")
    try:
        u.extra_minutes = None if x in (None, "") else int(round(float(x)))
    except (TypeError, ValueError):
        u.extra_minutes = None
        u.issues = ("extra_minutes is not a number",)
    u.summary = str(data.get("summary") or "")[:200]
    if u.incident_id:
        m = _INC.search(str(u.incident_id))
        u.incident_id = incident_id(m.group(1)) if m else str(u.incident_id)
    return u


# --------------------------------------------------------------------------- validation
def validate(u: Update, message: str, known_incidents: set[str] | None = None,
             known_crews: set[str] | None = None, checked_by_human: bool = False) -> Update:
    """Deterministic checks. Sets status = ok / needs_review / manager_must_handle.
    checked_by_human=True: the manager typed / corrected the values, so the "number from the message" check is skipped."""
    issues = list(u.issues)
    if u.type not in TYPES:
        issues.append(f"unknown type {u.type!r}")
        u.type = "info"
    if u.incident_id and u.source == "llm":
        digits = u.incident_id.split("_")[-1]
        written = {int(n) for n in re.findall(r"\d+", message)}
        if not digits.isdigit() or int(digits) not in written:
            issues.append(f"{u.incident_id} does not appear in the message (possible invention)")
    if u.incident_id and known_incidents is not None and u.incident_id not in known_incidents:
        issues.append(f"{u.incident_id} is not an incident of this crisis")
    if u.crew_id and known_crews is not None and u.crew_id not in known_crews:
        issues.append(f"{u.crew_id} is not a crew")
    if u.type in ("duration_update", "access_blocked", "restored") and not u.incident_id:
        issues.append("no incident id: which incident is it?")
    if u.type == "duration_update":
        if u.extra_minutes is None:
            issues.append("duration update without a duration")
        else:
            if not MIN_EXTRA_MIN <= u.extra_minutes <= MAX_EXTRA_MIN:
                issues.append(f"{u.extra_minutes} min is out of the plausible range")
            stated = minutes_in(message)
            if checked_by_human:
                pass
            elif stated is None or abs(abs(u.extra_minutes) - stated) > 1:
                issues.append("the duration does not match the message (possible invention)")
    elif u.extra_minutes is not None:
        u.extra_minutes = None
    u.issues = tuple(issues)
    if u.type in HUMAN_ONLY:
        u.status = "manager_must_handle"
    else:
        u.status = "needs_review" if issues else "ok"
    return u


def understand(message: str, client=None, known_incidents=None, known_crews=None) -> Update:
    """LLM first (if available), rules as fallback; always validated."""
    u = extract_llm(message, client)
    if u is None:
        u = extract_rules(message)
    return validate(u, message, known_incidents, known_crews)


# --------------------------------------------------------------------------- synthetic labelled messages
TEMPLATES = {
    "duration_update": ["{inc} worse than expected, need {h} more hours", "{crew} on {inc}: cable damaged, +{m} min",
                        "{inc}: transformer burnt, {h}h extra at least", "still working on {inc}, another {m} minutes",
                        "{inc} pole broken, repair will take {h} hours longer", "{inc} plus long que prévu, {h}h de plus"],
    "access_blocked": ["{crew}: tree across the road, cannot reach {inc}", "road to {inc} flooded, no access",
                       "{inc} unreachable, route coupée by a landslide", "snow on the pass, can't reach {inc}"],
    "restored": ["{inc} fixed, power is back", "{crew} done on {inc}", "{inc} restored, leaving site",
                 "{inc} rétabli"],
    "need_support": ["{inc}: need a second crew, pole too heavy", "{crew} need a crane on {inc}",
                     "{inc} big job, need backup"],
    "safety_hazard": ["DANGER live wire on the road near {inc}", "fire at the substation {inc}, stay away",
                      "{crew}: colleague injured on {inc}, calling help"],
    "info": ["{crew} on the way", "lunch in 10, then back", "{crew} arrived on site", "battery low on tablet"],
}


def synthetic_messages(n: int = 120, seed: int = 0, incident_ids=None, crew_ids=None) -> list[dict]:
    """Labelled messages: {'message', 'type', 'incident_id', 'crew_id', 'extra_minutes'}."""
    rng = np.random.default_rng(seed)
    incs = list(incident_ids or [f"INC_{k:06d}" for k in rng.integers(1, 16000, 50)])
    crews = list(crew_ids or [f"SITE_{s:02d}_C{c}" for s in range(1, 19) for c in (1, 2, 3)])
    out = []
    for k in range(n):
        kind = TYPES[k % len(TYPES)]
        tpl = str(rng.choice(TEMPLATES[kind]))
        inc, crew = str(rng.choice(incs)), str(rng.choice(crews))
        h, m = int(rng.integers(1, 5)), int(rng.choice([20, 30, 45, 90]))
        num = inc.split("_")[1]
        inc_txt = str(rng.choice([inc, f"INC {int(num)}", inc.lower()]))       # messy ids
        msg = tpl.format(inc=inc_txt, crew=crew, h=h, m=m)
        extra = None
        if kind == "duration_update":
            extra = h * 60 if "{h}" in tpl else m
        out.append({"message": msg, "type": kind, "incident_id": inc if "{inc}" in tpl else None,
                    "crew_id": crew if "{crew}" in tpl else None, "extra_minutes": extra})
    return out


# Hand-written messages that do NOT follow the templates: closer to real life, harder for keyword rules.
HARD_MESSAGES = [
    {"message": "on 4512 now. it's a mess, count half a day more", "type": "duration_update",
     "incident_id": None, "extra_minutes": None},
    {"message": "INC_000318 - wire hanging low over the school entrance, kids around", "type": "safety_hazard",
     "incident_id": "INC_000318", "extra_minutes": None},
    {"message": "INC_002201 should be quicker than planned, about 1 hour less", "type": "duration_update",
     "incident_id": "INC_002201", "extra_minutes": -60},
    {"message": "Gendarmerie stopped us, D117 shut because of the storm, INC_000917 not today",
     "type": "access_blocked", "incident_id": "INC_000917", "extra_minutes": None},
    {"message": "customers on INC_004002 have light again", "type": "restored", "incident_id": "INC_004002",
     "extra_minutes": None},
    {"message": "INC_000777 the transformer must be swapped, we can't lift it with two guys",
     "type": "need_support", "incident_id": "INC_000777", "extra_minutes": None},
    {"message": "INC 1290: add 45 to the estimate", "type": "duration_update", "incident_id": "INC_001290",
     "extra_minutes": 45},
    {"message": "smoke coming out of the cabinet at INC_003310, everyone stepped back", "type": "safety_hazard",
     "incident_id": "INC_003310", "extra_minutes": None},
    {"message": "INC_000650 ok c'est reparti", "type": "restored", "incident_id": "INC_000650",
     "extra_minutes": None},
    {"message": "bridge to INC_005120 is under water", "type": "access_blocked", "incident_id": "INC_005120",
     "extra_minutes": None},
    {"message": "INC_000208 two more hours, insulator chain broken", "type": "duration_update",
     "incident_id": "INC_000208", "extra_minutes": 120},
    {"message": "we're at the depot, coffee then off again", "type": "info", "incident_id": None,
     "extra_minutes": None},
]


def evaluate(messages: list[dict], extractor) -> dict:
    """Field accuracy (%) of an extractor (callable message → Update) on labelled messages."""
    ok = {"type": 0, "incident_id": 0, "extra_minutes": 0, "all": 0}
    for m in messages:
        u = extractor(m["message"])
        hits = {"type": u.type == m["type"], "incident_id": u.incident_id == m["incident_id"],
                "extra_minutes": (u.extra_minutes == m["extra_minutes"]) if m["type"] == "duration_update" else True}
        for k, v in hits.items():
            ok[k] += v
        ok["all"] += all(hits.values())
    return {k: round(100 * v / max(len(messages), 1), 1) for k, v in ok.items()}


# --------------------------------------------------------------------------- effect on the plan
def apply_updates(incidents, updates: list[Update], zone_factor: float = 3.0, zone_half_deg: float = 0.05):
    """Validated updates → (incidents with new durations, slow zones). Only status 'ok' updates are applied."""
    from copilot.travel import SlowZone

    by_id = {i.incident_id: i for i in incidents}
    zones = []
    for u in updates:
        if u.status != "ok" or u.incident_id not in by_id:
            continue
        inc = by_id[u.incident_id]
        if u.type == "duration_update" and u.extra_minutes:
            inc.nominal_duration = max(30, inc.nominal_duration + u.extra_minutes)
            inc.actual_duration = max(30, inc.actual_duration + u.extra_minutes)
        elif u.type == "access_blocked":
            zones.append(SlowZone(inc.lat - zone_half_deg, inc.lat + zone_half_deg, inc.lon - zone_half_deg,
                                  inc.lon + zone_half_deg, zone_factor, f"blocked near {inc.incident_id}"))
    return incidents, zones


def replan_impact(crisis, updates: list[Update], params: dict | None = None) -> dict:
    """Re-plan the crisis with the validated updates and compare with the plan before the messages:
    updated outlook (KPIs) + which assignments change. Simplification: updates are known from the start."""
    from copilot.kpi import compute_kpi
    from copilot.models import load_scenario
    from copilot.policies.wspt import WSPT
    from copilot.simulator import simulate, simulation_params
    from copilot.travel import make_travel

    p = {**(params or simulation_params()), "explain": False}
    sites, crews, incs = load_scenario(crisis)
    before = simulate(sites, crews, incs, WSPT(), p)
    sites, crews, incs = load_scenario(crisis)
    incs, zones = apply_updates(incs, updates)
    after = simulate(sites, crews, incs, WSPT(), p, make_travel({"travel": p["travel"]}, zones=zones))
    a = before.incidents.set_index("incident_id")["crew_id"]
    b = after.incidents.set_index("incident_id")["crew_id"]
    changed = sorted(a.index[a != b.reindex(a.index)])
    kb, ka = compute_kpi(before), compute_kpi(after)
    return {"before": kb, "after": ka, "changed_assignments": changed,
            "customer_hours_change": round(ka["customer_hours"] - kb["customer_hours"], 1)}


def main(argv: list[str] | None = None) -> None:
    """python -m copilot.ai.field_reports — accuracy of rules vs local LLM on 120 labelled messages."""
    from copilot.ai.llm import get_client

    msgs = synthetic_messages()
    print("rules:", evaluate(msgs, extract_rules))
    client = get_client()
    if client is None:
        print("LLM: not available (install Ollama + `ollama pull phi4-mini`) — the app uses the rules.")
    else:
        print("LLM  :", evaluate(msgs, lambda m: extract_llm(m, client) or extract_rules(m)), client.usage())


if __name__ == "__main__":
    main()
