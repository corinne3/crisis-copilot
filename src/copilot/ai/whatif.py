"""Brick ③ — What-if in plain language: "what if we lose 5 crews in Gers from H+6?"

    question  →  (LLM, or rules)  →  structured scenario (JSON, checked)  →  SIMULATOR (base vs scenario)
              →  numbers (KPI deltas)  →  short answer (template, optionally rephrased by the LLM, number guard)

The LLM only TRANSLATES the question into parameters and REPHRASES the answer. Every number comes from the
simulator. The parsed scenario is always shown to the manager, who can correct it before trusting the answer.

Supported changes (kind → fields):
    remove_crews   department | site_id, count, from_h, to_h (None = until the end)
    add_crews      department | site_id, count, from_h            (reinforcement arriving at from_h)
    slow_roads     area ("mountains" or a department), factor     (roads cut / snow)
    travel_speed   factor                                           (all travel times × factor)
    repair_time    factor                                           (all repairs × factor)
    rotation       groups, offset_h
    breaks         mode ("fixed" | "smart" | "none")
    fatigue        slope_per_h
    weights        alpha_sensitive, beta_wait_per_h
"""
from __future__ import annotations

import copy
import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger("copilot.whatif")

DEPARTMENTS = {"ariège": "Ariège", "ariege": "Ariège", "gers": "Gers", "haute-garonne": "Haute-Garonne",
               "haute garonne": "Haute-Garonne", "toulouse": "Haute-Garonne"}
KINDS = ("remove_crews", "add_crews", "slow_roads", "travel_speed", "repair_time", "rotation", "breaks", "fatigue",
         "weights")
_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
              "ten": 10, "twelve": 12, "fifteen": 15, "twenty": 20}


@dataclass
class Scenario:
    crisis: int | None = None
    changes: list[dict] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    source: str = "rules"

    def describe(self) -> list[str]:
        out = []
        for c in self.changes:
            k = c["kind"]
            where = c.get("site_id") or c.get("department") or c.get("area") or "all sites"
            if k == "remove_crews":
                until = "the end" if c.get("to_h") is None else f"H+{c['to_h']:g}"
                out.append(f"{c['count']} crews unavailable in {where} from H+{c.get('from_h', 0):g} to {until}")
            elif k == "add_crews":
                out.append(f"{c['count']} reinforcement crews in {where} from H+{c.get('from_h', 0):g}")
            elif k == "slow_roads":
                out.append(f"roads × {c['factor']:g} slower in {where}")
            elif k == "travel_speed":
                out.append(f"all travel times × {c['factor']:g}")
            elif k == "repair_time":
                out.append(f"all repairs × {c['factor']:g}")
            elif k == "rotation":
                out.append(f"crew rotation: {c['groups']} groups, +{c['offset_h']:g}h")
            elif k == "breaks":
                out.append(f"meal breaks: {c['mode']}")
            elif k == "fatigue":
                out.append(f"fatigue +{100 * c['slope_per_h']:g} %/h after 4h")
            elif k == "weights":
                out.append(f"weights α={c.get('alpha_sensitive')} β={c.get('beta_wait_per_h')}")
        return out


# --------------------------------------------------------------------------- rule parser
def _num(tok: str) -> float:
    tok = tok.lower()
    return float(_NUM_WORDS[tok]) if tok in _NUM_WORDS else float(tok.replace(",", "."))


_N = r"(\d+(?:[.,]\d+)?|" + "|".join(_NUM_WORDS) + r")"


def _where(text: str) -> dict:
    m = re.search(r"site[\s_-]?0*(\d{1,2})", text)
    if m:
        return {"site_id": f"SITE_{int(m.group(1)):02d}"}
    for k, v in DEPARTMENTS.items():
        if k in text:
            return {"department": v}
    return {}


def _window(text: str) -> tuple[float, float | None]:
    m = re.search(r"(?:from|after|at)\s*h\s*\+?\s*(\d+)", text)
    start = float(m.group(1)) if m else 0.0
    m = re.search(r"(?:to|until|till)\s*h\s*\+?\s*(\d+)", text)
    end = float(m.group(1)) if m else None
    m = re.search(rf"for\s+{_N}\s*(?:h|hours?)", text)
    if m and end is None:
        end = start + _num(m.group(1))
    return start, end


def parse_rules(question: str) -> Scenario:
    q = question.lower()
    sc = Scenario(source="rules")
    m = re.search(r"cris[ie]s\s*#?\s*(\d{1,3})", q)
    sc.crisis = int(m.group(1)) if m else None
    start, end = _window(q)
    m = re.search(rf"(?:lose|lost|losing|without|remove|minus|fewer)\s+{_N}\s+(?:\w+\s+)?crews?", q)
    if m:
        sc.changes.append({"kind": "remove_crews", "count": int(_num(m.group(1))), "from_h": start, "to_h": end,
                           **_where(q)})
    m = re.search(rf"(?:add|\+|extra|more|reinforcement of|get)\s+{_N}\s+(?:\w+\s+)?crews?", q) or \
        re.search(rf"{_N}\s+(?:more|extra|reinforcement|additional)\s+crews?", q)
    if m:
        sc.changes.append({"kind": "add_crews", "count": int(_num(m.group(1))), "from_h": start, **_where(q)})
    if re.search(r"(road|roads|pass|passes).*(cut|closed|blocked|snow|slow)|snow|landslide", q):
        f = re.search(rf"(?:x|×|times)\s*{_N}|{_N}\s*(?:x|×|times)", q)
        area = "mountains" if re.search(r"mountain|pyren|pass", q) else _where(q).get("department", "mountains")
        sc.changes.append({"kind": "slow_roads", "area": area,
                           "factor": _num(next(g for g in f.groups() if g)) if f else 3.0})
    m = re.search(r"(?:travel|driving|traffic)[^.]*?(\d+)\s*%\s*(slower|longer|faster)", q)
    if m:
        p = float(m.group(1)) / 100
        sc.changes.append({"kind": "travel_speed", "factor": round(1 + p if m.group(2) != "faster" else 1 - p, 3)})
    m = re.search(r"(?:repairs?|jobs?|work)[^.]*?(\d+)\s*%\s*(longer|slower|shorter|faster)", q)
    if m:
        p = float(m.group(1)) / 100
        sc.changes.append({"kind": "repair_time",
                           "factor": round(1 + p if m.group(2) in ("longer", "slower") else 1 - p, 3)})
    if re.search(r"rotation|stagger|relief|shifts? offset|two shifts|second shift", q):
        g = re.search(rf"{_N}\s+groups", q)
        o = (re.search(rf"(?:offset|\+|apart|later)\s*(?:of\s*)?{_N}\s*h", q)
             or re.search(rf"{_N}\s*h(?:ours?)?\s+(?:later|apart|offset)", q))
        sc.changes.append({"kind": "rotation", "groups": int(_num(g.group(1))) if g else 2,
                           "offset_h": _num(o.group(1)) if o else 4.0})
    if re.search(r"smart break|smart lunch|opportunistic", q):
        sc.changes.append({"kind": "breaks", "mode": "smart"})
    elif re.search(r"break|lunch|meal", q):
        sc.changes.append({"kind": "breaks", "mode": "none" if re.search(r"no (meal |lunch )?break", q) else "fixed"})
    m = re.search(r"fatigue[^.]*?(\d+(?:[.,]\d+)?)\s*%", q)
    if m or "fatigue" in q:
        sc.changes.append({"kind": "fatigue", "slope_per_h": float(m.group(1).replace(",", ".")) / 100 if m else 0.05})
    if not sc.changes:
        sc.issues.append("no change recognised — rephrase, or edit the scenario below")
    return sc


# --------------------------------------------------------------------------- LLM parser
SYSTEM = f"""You translate a crisis manager's what-if question into a JSON scenario for a power-grid crisis
simulator (3 departments: Ariège, Gers, Haute-Garonne; sites SITE_01..SITE_18; times in hours since start, H+0).
Answer ONLY with JSON: {{"crisis": int or null, "changes": [ {{"kind": one of {list(KINDS)}, ...fields}} ]}}.
Fields: remove_crews/add_crews: count, department or site_id, from_h, to_h (null = until the end);
slow_roads: area ("mountains" or a department), factor (3 = three times slower);
travel_speed / repair_time: factor applied to the TIME
(1.1 = 10 % slower/longer, 0.9 = 10 % faster/shorter); rotation: groups, offset_h; breaks: mode (fixed|smart|none);
fatigue: slope_per_h (0.05 = +5 %/h); weights: alpha_sensitive, beta_wait_per_h.
Use ONLY numbers written in the question (or the defaults factor 3, groups 2, offset_h 4). Never invent a change."""


def parse_llm(question: str, client) -> Scenario | None:
    data = client.generate_json(f"Question: {question}\nJSON:", SYSTEM, task="whatif", max_tokens=300) \
        if client is not None else None
    if not data or not isinstance(data.get("changes"), list):
        return None
    sc = Scenario(crisis=data.get("crisis") if isinstance(data.get("crisis"), int) else None,
                  changes=[c for c in data["changes"] if isinstance(c, dict)], source="llm")
    return sc


DEFAULT_NUMBERS = {0.0, 2.0, 3.0, 4.0, 0.05}


def validate(sc: Scenario, question: str, n_crews_per_dep: dict | None = None) -> Scenario:
    """Schema + plausibility + 'no invented number' check. Invalid changes are dropped and reported."""
    from copilot.ai.llm import numbers_in

    allowed = numbers_in(question) | DEFAULT_NUMBERS
    pct = {n / 100 for n in numbers_in(question)} | {1 + n / 100 for n in numbers_in(question)} | {
        1 - n / 100 for n in numbers_in(question)}
    keep = []
    for c in sc.changes:
        k = c.get("kind")
        if k not in KINDS:
            sc.issues.append(f"unknown change {k!r} ignored")
            continue
        if c.get("department") and c["department"] not in DEPARTMENTS.values():
            norm = DEPARTMENTS.get(str(c["department"]).lower())
            if norm is None:
                sc.issues.append(f"unknown department {c['department']!r}: change ignored")
                continue
            c["department"] = norm
        nums = [v for f, v in c.items() if f not in ("kind", "department", "site_id", "area", "mode")
                and isinstance(v, int | float) and not isinstance(v, bool)]
        invented = [v for v in nums if not any(abs(v - a) < 1e-6 for a in allowed | pct)]
        if invented:
            sc.issues.append(f"{k}: number(s) {invented} not in the question — check them")
        if k in ("remove_crews", "add_crews"):
            if not 1 <= int(c.get("count", 0)) <= 63:
                sc.issues.append(f"{k}: count must be between 1 and 63 — ignored")
                continue
            c["count"] = int(c["count"])
            c["from_h"] = float(c.get("from_h") or 0)
            c["to_h"] = None if c.get("to_h") in (None, "") else float(c["to_h"])
        if k in ("slow_roads", "travel_speed", "repair_time") and not 0.2 <= float(c.get("factor", 0)) <= 10:
            sc.issues.append(f"{k}: factor out of range — ignored")
            continue
        if k == "rotation":
            c["groups"], c["offset_h"] = int(c.get("groups", 2)), float(c.get("offset_h", 4))
        if k == "breaks" and c.get("mode") not in ("fixed", "smart", "none"):
            sc.issues.append("breaks: mode must be fixed, smart or none — ignored")
            continue
        keep.append(c)
    sc.changes = keep
    if sc.crisis is not None and not 1 <= int(sc.crisis) <= 100:
        sc.issues.append(f"crisis {sc.crisis} does not exist — using the selected crisis")
        sc.crisis = None
    return sc


def understand(question: str, client=None) -> Scenario:
    """LLM first, then reconciled with the rule-based reading (see reconcile)."""
    return reconcile(parse_llm(question, client), question)


def reconcile(llm_sc: Scenario | None, question: str) -> Scenario:
    """If the LLM reading is missing, empty or flagged (e.g. invented numbers) and the rules give a clean one,
    the rules win. Where both understood the same change, a different number is suspicious (e.g. the direction
    of "10 % slower"): the rule value is kept. The manager always sees which reading was used, and the issues."""
    rules = validate(parse_rules(question), question)
    if llm_sc is None:
        return rules
    sc = validate(llm_sc, question)
    if (not sc.changes or sc.issues) and rules.changes and not rules.issues:
        rules.issues.append("the local LLM's reading had problems: rule-based reading used instead")
        return rules
    by_kind = {c["kind"]: c for c in rules.changes}
    for c in sc.changes:
        r = by_kind.get(c["kind"])
        if not r:
            continue
        for f, v in r.items():
            if isinstance(v, int | float) and not isinstance(v, bool) and isinstance(c.get(f), int | float) \
                    and abs(float(c[f]) - float(v)) > 1e-6:
                sc.issues.append(f"{c['kind']}.{f}: LLM read {c[f]}, rules read {v} — rule value kept")
                c[f] = v
    return sc


# --------------------------------------------------------------------------- run
def build(crisis, sc: Scenario, base_params: dict | None = None):
    """Scenario → (sites, crews, incidents, params, travel_min, options) ready for simulate()."""
    from copilot.models import Crew, clean_dir, load_scenario, read_tables
    from copilot.simulator import Unavailability, simulation_params
    from copilot.travel import MOUNTAIN_ZONE, SlowZone, make_travel

    p = copy.deepcopy(base_params or simulation_params())
    p["explain"] = False
    sites, crews, incs = load_scenario(crisis)
    site_dep = read_tables(str(clean_dir()))[0].set_index("site_id")["department"].to_dict()
    unav, zones, mult = [], [], 1.0

    def pick_sites(c):
        if c.get("site_id"):
            return [c["site_id"]]
        if c.get("department"):
            return [s for s, d in site_dep.items() if d == c["department"]]
        return list(site_dep)

    for c in sc.changes:
        k = c["kind"]
        if k == "remove_crews":
            pool = sorted([cr for cr in crews if cr.site_id in pick_sites(c)],
                          key=lambda cr: (-int(cr.crew_id.rsplit("_C", 1)[1]), cr.site_id))
            for cr in pool[: c["count"]]:
                to = None if c.get("to_h") is None else int(c["to_h"] * 60)
                unav.append(Unavailability(cr.crew_id, int(c.get("from_h", 0) * 60), to))
        elif k == "add_crews":
            targets = [s for s in sites if s.site_id in pick_sites(c)] or sites
            targets = sorted(targets, key=lambda s: -s.n_crews)
            for j in range(c["count"]):
                s = targets[j % len(targets)]
                cr = Crew(f"{s.site_id}_R{j + 1}", s.site_id, s.lat, s.lon)
                crews.append(cr)
                if c.get("from_h", 0) > 0:
                    unav.append(Unavailability(cr.crew_id, 0, int(c["from_h"] * 60)))
        elif k == "slow_roads":
            if c.get("area", "mountains") == "mountains":
                zones.append(SlowZone(MOUNTAIN_ZONE.lat_min, MOUNTAIN_ZONE.lat_max, MOUNTAIN_ZONE.lon_min,
                                      MOUNTAIN_ZONE.lon_max, float(c["factor"]), "mountains"))
            else:
                sub = read_tables(str(clean_dir()))[1]
                d = sub[sub["department"] == c["area"]]
                if len(d):
                    zones.append(SlowZone(d.lat.min(), d.lat.max(), d.lon.min(), d.lon.max(), float(c["factor"]),
                                          c["area"]))
        elif k == "travel_speed":
            mult *= float(c["factor"])
        elif k == "repair_time":
            for i in incs:
                i.nominal_duration = int(round(i.nominal_duration * float(c["factor"])))
                i.actual_duration = int(round(i.actual_duration * float(c["factor"])))
        elif k == "rotation":
            p["rotation"] = {**p["rotation"], "enabled": True, "groups": c["groups"], "offset_min": int(c["offset_h"] * 60)}
        elif k == "breaks":
            p["breaks"] = {**p["breaks"], "enabled": c["mode"] != "none", "mode": c["mode"] if c["mode"] != "none" else "fixed"}
        elif k == "fatigue":
            p["fatigue"] = {**p["fatigue"], "enabled": True, "slope_per_h": float(c["slope_per_h"])}
        elif k == "weights":
            for w in ("alpha_sensitive", "beta_wait_per_h"):
                if c.get(w) is not None:
                    p[w] = float(c[w])
    travel = make_travel({"travel": p["travel"]}, multiplier=mult, zones=zones)
    return sites, crews, incs, p, travel, {"unavailability": unav}


COMPARE = [("customer_hours", "customer-hours", "{:,.0f}"),
           ("wait_p90_sensitive_h", "p90 wait, sensitive (h)", "{:.1f}"),
           ("wait_max_sensitive_h", "max wait, sensitive (h)", "{:.1f}"),
           ("pct_restored_4h", "% customers restored < 4h", "{:.1f}"),
           ("restored_95pct_h", "95 % restored at (h)", "{:.1f}"),
           ("end_h", "end of crisis (h)", "{:.1f}"),
           ("hours_without_active_crew", "hours with no crew working", "{:.1f}"),
           ("shift_overruns", "shift overruns", "{:.0f}")]


def run_whatif(crisis, sc: Scenario, base_params: dict | None = None, strategy=None) -> dict:
    """Simulate base and scenario with the same strategy. Returns KPIs, deltas and an answer text."""
    from copilot.kpi import compute_kpi
    from copilot.policies.wspt import WSPT
    from copilot.simulator import simulate, simulation_params

    strategy = strategy or WSPT
    bp = {**(base_params or simulation_params()), "explain": False}
    s0, c0, i0, p0, t0, o0 = build(crisis, Scenario(), bp)
    base_res = simulate(s0, c0, i0, strategy(), p0, t0, **o0)
    base = compute_kpi(base_res)
    s1, c1, i1, p1, t1, o1 = build(crisis, sc, bp)
    res = simulate(s1, c1, i1, strategy(), p1, t1, **o1)
    new = compute_kpi(res)
    rows = [{"kpi": label, "base": fmt.format(base[k]), "what-if": fmt.format(new[k]),
             "change": (f"{100 * (new[k] / base[k] - 1):+.1f} %" if k == "customer_hours" and base[k]
                        else f"{new[k] - base[k]:+.1f}")} for k, label, fmt in COMPARE]
    return {"base": base, "whatif": new, "table": rows, "answer": answer_text(sc, base, new), "result": res,
            "base_result": base_res}


def answer_text(sc: Scenario, base: dict, new: dict) -> str:
    d = 100 * (new["customer_hours"] / base["customer_hours"] - 1) if base["customer_hours"] else 0.0
    what = "; ".join(sc.describe()) or "no change"
    word = "more" if d > 0 else "fewer"
    txt = (f"Scenario: {what}. Total outage would be {abs(d):.1f} % {word} customer-hours "
           f"({base['customer_hours']:,.0f} → {new['customer_hours']:,.0f}). ")
    txt += (f"Sensitive customers: p90 wait {base['wait_p90_sensitive_h']:.1f} h → {new['wait_p90_sensitive_h']:.1f} h. "
            f"95 % of customers restored at H+{base['restored_95pct_h']:.1f} → H+{new['restored_95pct_h']:.1f}.")
    if any(c["kind"] == "remove_crews" and c.get("to_h") is not None for c in sc.changes) and d < 0:
        txt += (" Note: crews that come back are assumed rested and start a new 12h shift — this staggers the "
                "shifts (like a rotation), which can outweigh the hours lost.")
    if new["shift_overruns"] > 0:
        txt += f" ⚠️ {new['shift_overruns']:.0f} shift overruns: safety rule at risk."
    if new["unresolved"] > 0:
        txt += f" ⚠️ {new['unresolved']:.0f} incidents could not be restored."
    return txt
