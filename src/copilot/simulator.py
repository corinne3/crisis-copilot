"""Discrete-event simulator — event loop T1→T5, returns a SimResult.

Usage:
    from copilot.models import load_scenario
    from copilot.policies.baselines import NearestFCFS
    from copilot.simulator import simulate

    sites, crews, incidents = load_scenario(27)
    res = simulate(sites, crews, incidents, NearestFCFS())
    res.incidents, res.activities, res.decisions, res.series, res.metadata

Simulated rules:
    - one crew = one incident at a time; no reassignment while travelling or repairing;
    - an incident is restored after travel + repair; the crew stays on site, available;
    - shift rule: an assignment is accepted only if travel + NOMINAL duration ≤ remaining shift time;
      when the remaining shift is shorter than the shortest possible repair (MIN_REPAIR_MIN = 2h), or when the
      12h shift is over, the crew drives back to its base (return) and rests for 11h, then starts a new shift;
    - unavailability (stress test): the crew first finishes its current repair (safety), then goes out of service;
    - forced / forbidden assignments (human decision): applied before the strategy;
    - Monte Carlo (option): actual duration and speed are random; the strategy only sees nominal values.

Workforce realism (from V2, disabled by default so the base case is unchanged):
    - fatigue: a repair lasts nominal × (1 + slope × max(0, hours worked since last break − threshold)), capped;
      hours worked = travel + repair since the last break (or the shift start);
    - meal break (45 min, on site, counted in the shift, taken at the END of a repair, never during one):
        mode "fixed": as soon as work since the last break reaches threshold_min (4h, fatigue onset);
        mode "smart": forced at forced_min (6h); opportunistic from opportunistic_min (3h) if no waiting incident
                      is within radius_min of travel and fewer than max_share_site of the site's crews are on break;
    - the shift rule accounts for fatigue (expected duration = nominal × current fatigue factor);
    - crew rotation: without it, all 63 crews start at H0, reach 12h together and all rest at the same time
      (coverage gap); with rotation, each site's crews are split into `groups`; group g starts its shift at
      H0 + g × offset_min (before that: "standby", not available).
"""
from __future__ import annotations

import heapq
import json
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from copilot.models import (
    ACTIVITY_COLUMNS,
    DECISION_COLUMNS,
    Assignment,
    Crew,
    CrewState,
    Incident,
    IncidentState,
    SimResult,
    SimState,
    Site,
)
from copilot.policies.base import Policy
from copilot.travel import TravelMin

log = logging.getLogger("copilot.simulator")

MIN_REPAIR_MIN = 120               # shortest possible repair (2h): below this remaining shift time, the crew goes to rest
MAX_HORIZON_MIN = 14 * 24 * 60     # safeguard: at most 14 simulated days
HANDOVER_MARGIN_MIN = 60           # see is_feasible: jobs longer than a shift

# Processing order of simultaneous events (the most "freeing" first)
_PRIO = {"REPAIR_END": 0, "ARRIVAL": 1, "RETURN_END": 2, "REST_END": 3, "BREAK_END": 3, "OOS_END": 4,
         "UNAVAILABLE": 5, "SHIFT_END": 6, "REPORTED": 7}

BREAKS_DEFAULT = {"enabled": False, "mode": "fixed", "duration_min": 45, "threshold_min": 240,
                  "opportunistic_min": 180, "forced_min": 360, "radius_min": 30, "max_share_site": 0.5}
FATIGUE_DEFAULT = {"enabled": False, "threshold_h": 4.0, "slope_per_h": 0.05, "max": 1.5}
ROTATION_DEFAULT = {"enabled": False, "groups": 2, "offset_min": 240}
NESTED = ("monte_carlo", "breaks", "fatigue", "rotation")


def rotation_groups(crews, params: dict) -> dict[str, int]:
    """Rotation group of each crew: alternation inside each site (C1 → 0, C2 → 1, C3 → 0, …)."""
    r = params.get("rotation") or {}
    if not r.get("enabled") or int(r.get("groups", 1)) <= 1:
        return {c.crew_id: 0 for c in crews}
    n = int(r["groups"])
    rank: dict[str, int] = {}
    out = {}
    for c in sorted(crews, key=lambda q: q.crew_id):
        k = rank.get(c.site_id, 0)
        rank[c.site_id] = k + 1
        out[c.crew_id] = k % n
    return out


def fatigue_factor(work_min: float, params: dict) -> float:
    """Repair lengthening due to fatigue (1.0 = no effect)."""
    f = params.get("fatigue") or {}
    if not f.get("enabled"):
        return 1.0
    h = max(0.0, work_min / 60 - float(f.get("threshold_h", 4.0)))
    return min(float(f.get("max", 1.5)), 1.0 + float(f.get("slope_per_h", 0.05)) * h)


# --------------------------------------------------------------------------- constraints (used by strategies)
def remaining_shift(crew: Crew, t: int, params: dict) -> int:
    """Minutes of work still allowed in the current shift."""
    return int(params["shift_max_min"]) - (t - crew.t_shift_start)


def is_feasible(crew: Crew, incident: Incident, travel: float, t: int, params: dict) -> bool:
    """Shift rule: travel + NOMINAL duration ≤ remaining shift,
    and (crew, incident) pair not forbidden by the manager (params["forbidden"])."""
    if (crew.crew_id, incident.incident_id) in params.get("forbidden", ()):
        return False
    duration = incident.nominal_duration
    work = (params.get("_work_since_break") or {}).get(crew.crew_id)
    if work is not None:                                      # expected duration with the current fatigue
        duration = duration * fatigue_factor(work + travel, params)
    # a job longer than a whole shift (what-if, field report) would never fit: it is given to an almost fresh
    # crew (≥ shift − 1h left) and the overrun is counted in the KPIs (handover between crews is not modelled)
    duration = min(duration, int(params["shift_max_min"]) - HANDOVER_MARGIN_MIN)
    return travel + duration <= remaining_shift(crew, t, params)


# --------------------------------------------------------------------------- parameters
def simulation_params(cfg: dict | None = None, **overrides) -> dict:
    """Flatten the config used by the simulator and strategies (passed as is in SimState.params).

    Nested dicts (monte_carlo, breaks, fatigue, rotation) are MERGED with overrides: breaks={"enabled": True}.
    """
    if cfg is None:
        from copilot.config import load_config

        cfg = load_config()
    p = {
        "shift_max_min": int(cfg["work"]["shift_max_min"]),
        "rest_min": int(cfg["work"]["rest_min"]),
        "snapshot_step_min": int(cfg["display"]["snapshot_step_min"]),
        "alpha_sensitive": float(cfg["score"]["alpha_sensitive"]),
        "beta_wait_per_h": float(cfg["score"]["beta_wait_per_h"]),
        "monte_carlo": {"enabled": False, **cfg.get("monte_carlo", {})},
        "travel": dict(cfg["travel"]),
        "seed": int(cfg.get("seed", 42)),
        "breaks": {**BREAKS_DEFAULT, **cfg.get("breaks", {})},
        "fatigue": {**FATIGUE_DEFAULT, **cfg.get("fatigue", {})},
        "rotation": {**ROTATION_DEFAULT, **cfg.get("rotation", {})},
    }
    for k, v in overrides.items():
        if k in NESTED and isinstance(v, dict):
            p[k] = {**p[k], **v}
        else:
            p[k] = v
    return p


@dataclass
class Unavailability:
    """Crew out of service from t_start to t_end (None = until the end). It first finishes its current repair."""
    crew_id: str
    t_start: int
    t_end: int | None = None


@dataclass(order=True)
class _Event:
    t: int
    prio: int
    seq: int
    kind: str = field(compare=False)
    target: str = field(compare=False)       # crew_id or incident_id


# --------------------------------------------------------------------------- engine
class Simulator:
    def __init__(
        self,
        sites: list[Site],
        crews: list[Crew],
        incidents: list[Incident],
        policy: Policy,
        params: dict | None = None,
        travel_min: TravelMin | None = None,
        unavailability: Iterable[Unavailability] = (),
        forced: dict[str, str] | None = None,
        forbidden: Iterable[tuple[str, str]] = (),
        meta: dict | None = None,
    ):
        self.params = dict(params or simulation_params())
        if travel_min is None:
            from copilot.travel import make_travel

            travel_min = make_travel({"travel": self.params["travel"]})
        self.travel_min = travel_min
        self.policy = policy
        self.sites = {s.site_id: s for s in sites}
        self.crews = {c.crew_id: c for c in crews}
        self.incidents = {i.incident_id: i for i in incidents}
        if len(self.crews) != len(crews) or len(self.incidents) != len(incidents):
            raise ValueError("duplicate crew or incident ids")
        unknown = {c.site_id for c in crews} - set(self.sites)
        if unknown:
            raise ValueError(f"crews attached to unknown sites: {sorted(unknown)}")
        self.forced = dict(forced or {})
        self.forbidden = set(forbidden)
        self.params["forbidden"] = frozenset(self.forbidden)    # visible to strategies through is_feasible
        self.unavailability = list(unavailability)
        self.meta = dict(meta or {})

        self._queue: list[_Event] = []
        self._seq = 0
        self._oos_pending: dict[str, int | None] = {}          # crew → end of unavailability to apply
        self._current_activity: dict[str, dict] = {}
        self.activities: list[dict] = []
        self.decisions: list[dict] = []
        self.rejections: list[dict] = []
        self.rng = np.random.default_rng(self.params.get("seed", 42))
        # work since the last break (visible to strategies through is_feasible when fatigue is on)
        self._work: dict[str, int] = {c.crew_id: 0 for c in crews}
        self._mission_start: dict[str, int] = {}
        self._on_break: set[str] = set()
        self.n_breaks = 0
        self._fatigue_factors: list[float] = []
        if (self.params.get("fatigue") or {}).get("enabled"):
            self.params["_work_since_break"] = self._work
        self._apply_monte_carlo_durations()

    # ------------------------------------------------------------------ helpers
    def _push(self, t: int, kind: str, target: str) -> None:
        self._seq += 1
        heapq.heappush(self._queue, _Event(int(t), _PRIO[kind], self._seq, kind, target))

    def _mc(self) -> dict:
        return self.params.get("monte_carlo") or {}

    def _apply_monte_carlo_durations(self) -> None:
        mc = self._mc()
        if not mc.get("enabled"):
            return
        sigma = float(mc.get("sigma_duration", 0.3))
        for inc in self.incidents.values():
            f = float(np.clip(self.rng.lognormal(0.0, sigma), 0.5, 2.0))
            inc.actual_duration = int(round(inc.nominal_duration * f))

    def _actual_travel(self, nominal: float) -> int:
        mc = self._mc()
        if mc.get("enabled") and nominal > 0:
            v = float(np.clip(self.rng.normal(1.0, float(mc.get("sigma_speed", 0.15))), 0.6, 1.3))
            nominal = nominal / v
        return int(round(nominal))

    def _open(self, crew: Crew, kind: str, t: int, end: int, inc: Incident | None,
              src: tuple[float, float], dst: tuple[float, float]) -> None:
        a = {"crew_id": crew.crew_id, "type": kind, "t_start": int(t), "t_end": int(end),
             "incident_id": inc.incident_id if inc else None,
             "lat_from": src[0], "lon_from": src[1], "lat_to": dst[0], "lon_to": dst[1]}
        self.activities.append(a)
        self._current_activity[crew.crew_id] = a

    # ------------------------------------------------------------------ crew transitions
    def _release(self, crew: Crew, t: int, after_break: bool = False) -> None:
        """The crew just finished something: out of service, rest, break or available?"""
        if crew.crew_id in self._oos_pending:
            t_end = self._oos_pending.pop(crew.crew_id)
            crew.state = CrewState.OUT_OF_SERVICE
            end = t_end if t_end is not None else MAX_HORIZON_MIN
            crew.t_free = end
            self._open(crew, "out_of_service", t, end, None, (crew.lat, crew.lon), (crew.lat, crew.lon))
            if t_end is not None:
                self._push(t_end, "OOS_END", crew.crew_id)
            return
        if remaining_shift(crew, t, self.params) < MIN_REPAIR_MIN:
            self._go_rest(crew, t)
            return
        if not after_break and self._needs_break(crew, t):
            self._take_break(crew, t)
            return
        crew.state = CrewState.AVAILABLE
        crew.t_free = t
        crew.incident_id = None

    # ------------------------------------------------------------------ meal break
    def _needs_break(self, crew: Crew, t: int) -> bool:
        b = self.params.get("breaks") or {}
        if not b.get("enabled"):
            return False
        work = self._work[crew.crew_id]
        if b.get("mode", "fixed") == "fixed":
            return work >= int(b["threshold_min"])
        if work >= int(b["forced_min"]):
            return True
        if work < int(b["opportunistic_min"]):
            return False
        same_site = [c for c in self.crews.values() if c.site_id == crew.site_id]
        on_break = sum(1 for c in same_site if c.crew_id in self._on_break)
        if on_break >= max(1, int(len(same_site) * float(b["max_share_site"]))):
            return False                                     # staggering: not the whole site on break
        nearby = any(self.travel_min(crew.lat, crew.lon, i.lat, i.lon) <= float(b["radius_min"])
                     for i in self.incidents.values() if i.state is IncidentState.WAITING)
        return not nearby                                    # opportunistic: nothing urgent nearby

    def _take_break(self, crew: Crew, t: int) -> None:
        duration = int((self.params.get("breaks") or {}).get("duration_min", 45))
        crew.state = CrewState.RESTING                        # not available to strategies
        crew.t_free = t + duration
        self._on_break.add(crew.crew_id)
        self.n_breaks += 1
        self._open(crew, "break", t, t + duration, None, (crew.lat, crew.lon), (crew.lat, crew.lon))
        self._push(t + duration, "BREAK_END", crew.crew_id)

    def _go_rest(self, crew: Crew, t: int) -> None:
        site = self.sites[crew.site_id]
        duration = self._actual_travel(self.travel_min(crew.lat, crew.lon, site.lat, site.lon))
        crew.state = CrewState.RETURNING
        crew.incident_id = None
        crew.t_free = t + duration + int(self.params["rest_min"])
        self._open(crew, "return", t, t + duration, None, (crew.lat, crew.lon), (site.lat, site.lon))
        self._push(t + duration, "RETURN_END", crew.crew_id)

    def _new_shift(self, crew: Crew, t: int) -> None:
        crew.t_shift_start = t
        self._work[crew.crew_id] = 0
        crew.state = CrewState.AVAILABLE
        crew.t_free = t
        self._push(t + int(self.params["shift_max_min"]), "SHIFT_END", crew.crew_id)

    # ------------------------------------------------------------------ event handling
    def _handle(self, e: _Event) -> None:
        t = e.t
        if e.kind == "REPORTED":
            self.incidents[e.target].state = IncidentState.WAITING
            return

        crew = self.crews[e.target]
        if e.kind == "ARRIVAL":
            inc = self.incidents[crew.incident_id]
            crew.state = CrewState.REPAIRING
            crew.lat, crew.lon = inc.lat, inc.lon
            inc.state = IncidentState.IN_PROGRESS
            inc.t_arrival = t
            if (self.params.get("fatigue") or {}).get("enabled"):    # fatigue lengthens the repair
                travelled = t - self._mission_start.get(crew.crew_id, t)
                f = fatigue_factor(self._work[crew.crew_id] + travelled, self.params)
                self._fatigue_factors.append(f)
                inc.actual_duration = int(round(inc.actual_duration * f))
            end = t + int(inc.actual_duration)
            crew.t_free = end
            self._open(crew, "repair", t, end, inc, (inc.lat, inc.lon), (inc.lat, inc.lon))
            self._push(end, "REPAIR_END", crew.crew_id)
        elif e.kind == "REPAIR_END":
            inc = self.incidents[crew.incident_id]
            inc.state = IncidentState.RESTORED
            inc.t_restored = t
            crew.incident_id = None
            self._work[crew.crew_id] += t - self._mission_start.pop(crew.crew_id, t)
            self._release(crew, t)
        elif e.kind == "RETURN_END":
            site = self.sites[crew.site_id]
            crew.lat, crew.lon = site.lat, site.lon
            crew.state = CrewState.RESTING
            end = t + int(self.params["rest_min"])
            self._open(crew, "rest", t, end, None, (site.lat, site.lon), (site.lat, site.lon))
            self._push(end, "REST_END", crew.crew_id)
        elif e.kind == "REST_END":
            if crew.state is CrewState.RESTING:
                if crew.crew_id in self._oos_pending:
                    self._release(crew, t)
                else:
                    self._new_shift(crew, t)
        elif e.kind == "BREAK_END":
            if crew.crew_id in self._on_break:
                self._on_break.discard(crew.crew_id)
                self._work[crew.crew_id] = 0
                self._release(crew, t, after_break=True)
        elif e.kind == "OOS_END":
            if crew.state is CrewState.OUT_OF_SERVICE:
                self._new_shift(crew, t)              # comes back rested (assumption)
        elif e.kind == "UNAVAILABLE":
            t_end = self._unavailable_end[e.seq]
            if crew.state in (CrewState.AVAILABLE, CrewState.RESTING):
                if crew.state is CrewState.RESTING:   # the rest (or break, or standby) is cut at t
                    self._current_activity[crew.crew_id]["t_end"] = t
                    self._on_break.discard(crew.crew_id)
                self._oos_pending[crew.crew_id] = t_end
                self._release(crew, t)
            else:                                     # travel / repair / return: finish first
                self._oos_pending[crew.crew_id] = t_end
        elif e.kind == "SHIFT_END":
            if crew.state is CrewState.AVAILABLE and t >= crew.t_shift_start + int(self.params["shift_max_min"]):
                self._go_rest(crew, t)

    # ------------------------------------------------------------------ decision
    def _state(self, t: int) -> SimState:
        available = [c for c in self.crews.values() if c.state is CrewState.AVAILABLE]
        waiting = sorted((i for i in self.incidents.values() if i.state is IncidentState.WAITING),
                         key=lambda i: (i.t_reported, i.incident_id))
        return SimState(t=t, available_crews=available, waiting_incidents=waiting,
                        all_crews=list(self.crews.values()), params=self.params)

    def _dispatch(self, crew: Crew, inc: Incident, t: int, detail: dict, origin: str) -> None:
        analysis = {}
        if self.params.get("explain", True):          # explanation computed for ANY strategy
            from copilot.explanation import explain

            analysis = explain(
                crew, inc, t, self.params, self.travel_min,
                [i for i in self.incidents.values() if i.state is IncidentState.WAITING],
                [c for c in self.crews.values() if c.state is CrewState.AVAILABLE],
            )
        duration = self._actual_travel(self.travel_min(crew.lat, crew.lon, inc.lat, inc.lon))
        self._mission_start[crew.crew_id] = t
        crew.state = CrewState.TRAVELLING
        crew.incident_id = inc.incident_id
        crew.t_free = t + duration + inc.actual_duration
        inc.state = IncidentState.ASSIGNED
        inc.crew_id = crew.crew_id
        inc.t_assigned = t
        self._open(crew, "travel", t, t + duration, inc, (crew.lat, crew.lon), (inc.lat, inc.lon))
        self._push(t + duration, "ARRIVAL", crew.crew_id)
        self.decisions.append({"t": t, "crew_id": crew.crew_id, "incident_id": inc.incident_id,
                               "explanation": json.dumps({"origin": origin, **analysis,
                                                          "strategy_detail": detail or {}},
                                                         ensure_ascii=False, default=str)})

    def _validate(self, a: Assignment, t: int, free: dict, waiting: dict) -> str | None:
        if a.crew_id not in free:
            return "crew not available"
        if a.incident_id not in waiting:
            return "incident not waiting"
        if (a.crew_id, a.incident_id) in self.forbidden:
            return "assignment forbidden by the manager"
        crew, inc = free[a.crew_id], waiting[a.incident_id]
        if not is_feasible(crew, inc, self.travel_min(crew.lat, crew.lon, inc.lat, inc.lon), t, self.params):
            return "not enough shift time left"
        return None

    def _decide(self, t: int) -> None:
        state = self._state(t)
        if not state.available_crews or not state.waiting_incidents:
            return
        free = {c.crew_id: c for c in state.available_crews}
        waiting = {i.incident_id: i for i in state.waiting_incidents}

        # 1) assignments forced by the manager (as soon as both crew and incident are ready)
        for cid, iid in list(self.forced.items()):
            if cid in free and iid in waiting:
                reason = self._validate(Assignment(cid, iid, {"forced": True}), t, free, waiting)
                if reason and reason != "assignment forbidden by the manager":
                    self.rejections.append({"t": t, "crew_id": cid, "incident_id": iid, "reason": reason})
                    continue
                self._dispatch(free.pop(cid), waiting.pop(iid), t, {"forced": True}, "human")
                del self.forced[cid]
        if not free or not waiting:
            return

        # 2) strategy (it sees neither crews already sent nor incidents reserved by a forced assignment)
        reserved = set(self.forced.values())
        state.available_crews = [c for c in state.available_crews if c.crew_id in free and c.crew_id not in self.forced]
        state.waiting_incidents = [i for i in state.waiting_incidents
                                   if i.incident_id in waiting and i.incident_id not in reserved]
        if not state.available_crews or not state.waiting_incidents:
            return
        excluded: set[str] = set()
        for _pass in range(3):      # passes 2 and 3: only if the strategy proposed a forbidden pair
            forbidden_seen = False
            for a in self.policy.decide(state, self.travel_min) or []:
                reason = self._validate(a, t, free, waiting)
                if reason:
                    self.rejections.append({"t": t, "crew_id": a.crew_id, "incident_id": a.incident_id,
                                            "reason": reason})
                    if reason.startswith("assignment forbidden"):
                        forbidden_seen = True
                        excluded.add(a.crew_id)
                    continue
                self._dispatch(free.pop(a.crew_id), waiting.pop(a.incident_id), t, a.explanation,
                               getattr(self.policy, "name", type(self.policy).__name__))
            if not forbidden_seen:
                break
            state.available_crews = [c for c in state.available_crews
                                     if c.crew_id in free and c.crew_id not in excluded]
            state.waiting_incidents = [i for i in state.waiting_incidents if i.incident_id in waiting]
            if not state.available_crews or not state.waiting_incidents:
                break

    # ------------------------------------------------------------------ main loop
    def run(self) -> SimResult:
        clock = time.perf_counter()
        for inc in self.incidents.values():
            inc.state = IncidentState.NOT_YET
            self._push(inc.t_reported, "REPORTED", inc.incident_id)
        groups = rotation_groups(list(self.crews.values()), self.params)
        offset = int((self.params.get("rotation") or {}).get("offset_min", 240))
        for crew in self.crews.values():
            start = crew.t_shift_start + groups[crew.crew_id] * offset
            if start > crew.t_shift_start:                 # rotation: the crew is on standby until `start`
                crew.state = CrewState.RESTING
                crew.t_free = start
                self._open(crew, "standby", crew.t_shift_start, start, None, (crew.lat, crew.lon), (crew.lat, crew.lon))
                self._push(start, "REST_END", crew.crew_id)
            else:
                self._new_shift(crew, crew.t_shift_start)
        self._unavailable_end: dict[int, int | None] = {}
        for u in self.unavailability:
            if u.crew_id not in self.crews:
                raise ValueError(f"unavailability: unknown crew {u.crew_id}")
            self._push(u.t_start, "UNAVAILABLE", u.crew_id)
            self._unavailable_end[self._seq] = u.t_end

        t = 0
        while self._queue:
            t = self._queue[0].t
            if t > MAX_HORIZON_MIN:
                log.warning("max horizon reached (%s min): simulation stopped", MAX_HORIZON_MIN)
                break
            while self._queue and self._queue[0].t == t:          # T1/T2: all events at instant t
                self._handle(heapq.heappop(self._queue))
            self._decide(t)                                       # T3/T4: one decision per instant
            if all(i.state is IncidentState.RESTORED for i in self.incidents.values()):
                break                                             # T5: crisis over

        t_end = t
        for a in self.activities:                                 # cut rest / out-of-service beyond the end
            a["t_end"] = min(a["t_end"], max(t_end, a["t_start"]))
        return self._result(t_end, time.perf_counter() - clock)

    # ------------------------------------------------------------------ outputs
    def _result(self, t_end: int, elapsed_s: float) -> SimResult:
        df_inc = pd.DataFrame([{
            "incident_id": i.incident_id, "substation_id": i.substation_id, "lat": i.lat, "lon": i.lon,
            "customers": i.customers, "sensitive": i.sensitive, "town": None, "department": None,
            "t_reported": i.t_reported, "t_assigned": i.t_assigned, "t_arrival": i.t_arrival,
            "t_restored": i.t_restored, "crew_id": i.crew_id, "actual_duration": i.actual_duration,
            "nominal_duration": i.nominal_duration, "cause": i.cause,
        } for i in self.incidents.values()])
        df_act = pd.DataFrame(self.activities, columns=ACTIVITY_COLUMNS)
        df_dec = pd.DataFrame(self.decisions, columns=DECISION_COLUMNS)
        series = build_series(df_inc, df_act, len(self.crews), t_end, int(self.params["snapshot_step_min"]))

        unresolved = int(df_inc["t_restored"].isna().sum())
        done = df_inc.dropna(subset=["t_restored"])
        customer_hours = float(((done["t_restored"] - done["t_reported"]) * done["customers"]).sum() / 60)
        meta = {
            **self.meta,
            "strategy": getattr(self.policy, "name", type(self.policy).__name__),
            "n_crews": len(self.crews),
            "n_incidents": len(self.incidents),
            "unresolved": unresolved,
            "t_end_min": int(t_end),
            "customer_hours": round(customer_hours, 1),
            "n_decisions": len(self.decisions),
            "rejections": self.rejections,
            "monte_carlo": bool(self._mc().get("enabled")),
            "seed": self.params.get("seed"),
            "compute_s": round(elapsed_s, 3),
            "n_breaks": self.n_breaks,
            "mean_fatigue": round(float(np.mean(self._fatigue_factors)), 3) if self._fatigue_factors else 1.0,
            "params": {k: (sorted(v) if isinstance(v, frozenset) else v)
                       for k, v in self.params.items() if k not in ("travel", "_work_since_break")},
        }
        if unresolved:
            log.warning("%s incidents not restored (no crew can take them any more)", unresolved)
        return SimResult(incidents=df_inc, activities=df_act, decisions=df_dec, series=series, metadata=meta)


BUSY_TYPES = ("travel", "repair")
RESTING_TYPES = ("return", "rest", "break", "standby")


def build_series(df_inc: pd.DataFrame, df_act: pd.DataFrame, n_crews: int, t_end: int, step: int) -> pd.DataFrame:
    """Snapshots every `step` minutes (0, step, 2·step, …, ≥ t_end).

    customers_off counts each SUBSTATION once (two open incidents on the same substation = same customers).
    """
    ts = np.arange(0, t_end + step, step)
    rep = df_inc["t_reported"].to_numpy()
    res = df_inc["t_restored"].fillna(np.inf).to_numpy(dtype=float)
    ass = df_inc["t_assigned"].fillna(np.inf).to_numpy(dtype=float)
    subs = df_inc["substation_id"].to_numpy()
    cust = df_inc["customers"].to_numpy()
    a_start, a_end, a_type = (df_act["t_start"].to_numpy(), df_act["t_end"].to_numpy(),
                              df_act["type"].to_numpy())

    rows = []
    for t in ts:
        open_ = (rep <= t) & (res > t)
        seen = dict(zip(subs[open_], cust[open_], strict=True))
        active = a_type[(a_start <= t) & (a_end > t)]
        busy = int(np.isin(active, BUSY_TYPES).sum())
        resting = int(np.isin(active, RESTING_TYPES).sum())
        out = int((active == "out_of_service").sum())
        rows.append({"t": int(t), "customers_off": int(sum(seen.values())),
                     "incidents_waiting": int(((rep <= t) & (ass > t)).sum()),
                     "crews_busy": busy, "crews_resting": resting, "crews_out": out,
                     "crews_available": n_crews - busy - resting - out})
    return pd.DataFrame(rows)


def simulate(sites: list[Site], crews: list[Crew], incidents: list[Incident], policy: Policy,
             params: dict | None = None, travel_min: TravelMin | None = None, **options) -> SimResult:
    """Shortcut for Simulator(...).run(). Options: unavailability, forced, forbidden, meta."""
    return Simulator(sites, crews, incidents, policy, params, travel_min, **options).run()


def simulate_crisis(crisis, policy: Policy, params: dict | None = None, scenario: dict | None = None,
                    travel_min: TravelMin | None = None, folder=None, **options) -> SimResult:
    """Load the crisis (load_scenario) then simulate it. `scenario` = load_scenario params."""
    from copilot.models import clean_dir, load_scenario, normalise_crisis_id, read_tables

    sites, crews, incidents = load_scenario(crisis, scenario, folder)
    meta = {"crisis": normalise_crisis_id(crisis), **options.pop("meta", {})}
    res = simulate(sites, crews, incidents, policy, params, travel_min, meta=meta, **options)
    _, df_all = read_tables(str(clean_dir(folder)))         # add town / department (map, Gini, briefing)
    geo = df_all.set_index("incident_id")[["town", "department"]]
    res.incidents["town"] = res.incidents["incident_id"].map(geo["town"])
    res.incidents["department"] = res.incidents["incident_id"].map(geo["department"])
    return res
