"""Small synthetic scenarios: tests run without the challenge data."""
from __future__ import annotations

from copilot.config import load_config, project_path
from copilot.models import Crew, Incident, Site
from copilot.simulator import simulation_params

HAS_DATA = (project_path(load_config()["paths"]["clean"]) / "incidents.parquet").exists()

BASE = (43.60, 1.44)      # Toulouse (plain)


def site(sid="SITE_01", n=1, lat=BASE[0], lon=BASE[1]) -> Site:
    return Site(sid, "Town", lat, lon, n)


def crews_for(*sites: Site) -> list[Crew]:
    from copilot.models import make_crews

    return make_crews(list(sites))


def incident(iid, t=0, customers=100, duration=120, sensitive=False, dlat=0.0, dlon=0.0) -> Incident:
    return Incident(iid, f"SUB_{iid}", t, duration, duration, "lightning", customers, sensitive,
                    BASE[0] + dlat, BASE[1] + dlon)


def params(**overrides) -> dict:
    return simulation_params(**{"explain": True, **overrides})
