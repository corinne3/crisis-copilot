"""Common interface of all dispatch strategies."""
from __future__ import annotations

from copilot.models import Assignment, SimState
from copilot.travel import TravelMin


class Policy:
    name: str = "abstract"

    def decide(self, state: SimState, travel_min: TravelMin) -> list[Assignment]:
        """Called at every decision point (incident reported, crew released).
        Returns the assignments to start now.
        Must NEVER propose an assignment that breaks the remaining shift time (use simulator.is_feasible)."""
        raise NotImplementedError
