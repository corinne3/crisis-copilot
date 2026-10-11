"""WSPT score + Hungarian assignment — the recommended strategy.

score_ij = w_i / (travel_ij + duration_i)                      (Smith's rule: weighted shortest processing time)
w_i      = N_i × (1 + α·sensitive_i) × (1 + β·wait_i_in_hours)
Global assignment maximising Σ score (scipy.optimize.linear_sum_assignment).

α and β come from state.params (alpha_sensitive, beta_wait_per_h) — the learning loop (brick ④) only ever
proposes new values for them; it never changes this code.
"""
from __future__ import annotations

import numpy as np

from copilot.models import Assignment, SimState
from copilot.policies.base import Policy
from copilot.travel import TravelMin

INFEASIBLE = -1e9


def weight(customers: int, sensitive: bool, wait_min: float, alpha: float, beta: float) -> float:
    return customers * (1 + alpha * bool(sensitive)) * (1 + beta * max(0.0, wait_min) / 60)


class WSPT(Policy):
    name = "WSPT"

    def decide(self, state: SimState, travel_min: TravelMin) -> list[Assignment]:
        from scipy.optimize import linear_sum_assignment

        from copilot.simulator import is_feasible

        crews, incs, p = state.available_crews, state.waiting_incidents, state.params
        if not crews or not incs:
            return []
        alpha, beta = float(p.get("alpha_sensitive", 5.0)), float(p.get("beta_wait_per_h", 0.1))
        S = np.full((len(crews), len(incs)), INFEASIBLE)
        T = np.zeros_like(S)
        for a, c in enumerate(crews):
            for b, i in enumerate(incs):
                tr = travel_min(c.lat, c.lon, i.lat, i.lon)
                T[a, b] = tr
                if is_feasible(c, i, tr, state.t, p):
                    S[a, b] = weight(i.customers, i.sensitive, state.t - i.t_reported, alpha, beta) / (
                        tr + i.nominal_duration)
        rows, cols = linear_sum_assignment(-S)
        out = []
        for a, b in zip(rows, cols, strict=True):
            if S[a, b] <= INFEASIBLE / 2:
                continue
            i = incs[b]
            others = np.sort(S[a][S[a] > INFEASIBLE / 2])[::-1]
            margin = float((others[0] - others[1]) / others[0]) if len(others) > 1 and others[0] > 0 else 1.0
            out.append(Assignment(crews[a].crew_id, i.incident_id, {
                "rule": self.name, "score": round(float(S[a, b]), 3), "customers": i.customers,
                "sensitive": bool(i.sensitive), "wait_min": state.t - i.t_reported,
                "travel_min": round(float(T[a, b]), 1), "duration_min": i.nominal_duration,
                "margin": round(margin, 2),
            }))
        return out


STRATEGIES = {"R1": "copilot.policies.baselines:NearestFCFS",
              "R2": "copilot.policies.baselines:MostCustomers",
              "WSPT": "copilot.policies.wspt:WSPT"}


def resolve_strategy(name: str) -> type[Policy]:
    """'R1', 'R2', 'WSPT' or 'module:Class'."""
    import importlib

    spec = STRATEGIES.get(name.strip().upper(), name.strip())
    if ":" not in spec:
        raise ValueError(f"unknown strategy: {name} (expected one of {list(STRATEGIES)} or 'module:Class')")
    module, cls = spec.split(":", 1)
    return getattr(importlib.import_module(module), cls)
