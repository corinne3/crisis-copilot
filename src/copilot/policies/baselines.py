"""Reference strategies R1 and R2 — the WSPT strategy must beat them.

R1 — NearestFCFS    : incidents in order of arrival; each one takes the nearest available crew.
R2 — MostCustomers  : incidents by decreasing number of customers; each one takes the nearest crew.

Both respect the shift rule (is_feasible): a crew without enough shift time left is never proposed.
"""
from __future__ import annotations

from copilot.models import Assignment, Incident, SimState
from copilot.policies.base import Policy
from copilot.travel import TravelMin


class _Nearest(Policy):
    name = "abstract"

    def order(self, incidents: list[Incident]) -> list[Incident]:
        raise NotImplementedError

    def decide(self, state: SimState, travel_min: TravelMin) -> list[Assignment]:
        from copilot.simulator import is_feasible  # local import: avoids a circular import

        free = list(state.available_crews)
        out: list[Assignment] = []
        for inc in self.order(state.waiting_incidents):
            if not free:
                break
            candidates = []
            for crew in free:
                tr = travel_min(crew.lat, crew.lon, inc.lat, inc.lon)
                if is_feasible(crew, inc, tr, state.t, state.params):
                    candidates.append((tr, crew.crew_id, crew))
            if not candidates:
                continue
            tr, _, crew = min(candidates)
            free.remove(crew)
            out.append(Assignment(crew.crew_id, inc.incident_id, {
                "rule": self.name, "travel_min": round(tr, 1), "customers": inc.customers,
                "wait_min": state.t - inc.t_reported,
            }))
        return out


class NearestFCFS(_Nearest):
    """R1: first come, first served + nearest crew."""
    name = "R1"

    def order(self, incidents):
        return sorted(incidents, key=lambda i: (i.t_reported, i.incident_id))


class MostCustomers(_Nearest):
    """R2: most customers first (sensitive first on ties) + nearest crew."""
    name = "R2"

    def order(self, incidents):
        return sorted(incidents, key=lambda i: (-i.customers, not i.sensitive, i.t_reported, i.incident_id))
