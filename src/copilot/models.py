"""Domain objects and scenario loading.

Units: time in MINUTES since the start of the crisis (int), distances in km, WGS84 coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

import pandas as pd


class CrewState(StrEnum):
    AVAILABLE = "available"
    TRAVELLING = "travelling"
    REPAIRING = "repairing"
    RETURNING = "returning"          # back to base before resting
    RESTING = "resting"              # rest, break or standby: not available
    OUT_OF_SERVICE = "out_of_service"


class IncidentState(StrEnum):
    NOT_YET = "not_yet"
    WAITING = "waiting"
    ASSIGNED = "assigned"            # crew on its way
    IN_PROGRESS = "in_progress"      # repair under way
    RESTORED = "restored"


@dataclass
class Site:
    site_id: str
    town: str
    lat: float
    lon: float
    n_crews: int


@dataclass
class Incident:
    incident_id: str
    substation_id: str
    t_reported: int
    nominal_duration: int            # minutes (120/240/480) — what strategies see
    actual_duration: int             # minutes — what the simulator plays (= nominal without Monte Carlo)
    cause: str
    customers: int
    sensitive: bool
    lat: float
    lon: float
    state: IncidentState = IncidentState.NOT_YET
    crew_id: str | None = None
    t_assigned: int | None = None
    t_arrival: int | None = None
    t_restored: int | None = None


@dataclass
class Crew:
    crew_id: str                     # e.g. "SITE_04_C2"
    site_id: str                     # home base (rest location)
    lat: float                       # current position
    lon: float
    state: CrewState = CrewState.AVAILABLE
    t_shift_start: int = 0
    t_free: int = 0
    incident_id: str | None = None


@dataclass
class Assignment:
    crew_id: str
    incident_id: str
    explanation: dict = field(default_factory=dict)


@dataclass
class SimState:
    """What a strategy receives at each decision point."""
    t: int
    available_crews: list[Crew]
    waiting_incidents: list[Incident]
    all_crews: list[Crew]
    params: dict


@dataclass
class SimResult:
    """Simulator output: 4 tables + metadata."""
    incidents: pd.DataFrame
    activities: pd.DataFrame
    decisions: pd.DataFrame
    series: pd.DataFrame
    metadata: dict = field(default_factory=dict)


ACTIVITY_COLUMNS = ["crew_id", "type", "t_start", "t_end", "incident_id", "lat_from", "lon_from", "lat_to", "lon_to"]
DECISION_COLUMNS = ["t", "crew_id", "incident_id", "explanation"]


# --------------------------------------------------------------------------- scenario loading
def normalise_crisis_id(crisis: str | int) -> str:
    """27, "27", "crisis_027", "CRISE_027" → "CRISIS_027"."""
    if isinstance(crisis, int):
        return f"CRISIS_{crisis:03d}"
    s = str(crisis).strip().upper()
    for prefix in ("CRISIS_", "CRISE_"):
        if s.startswith(prefix):
            s = s[len(prefix):]
    if s.isdigit():
        return f"CRISIS_{int(s):03d}"
    raise ValueError(f"invalid crisis id: {crisis!r} (expected e.g. 27 or 'CRISIS_027')")


def clean_dir(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    from copilot.config import load_config, project_path

    return project_path(load_config()["paths"]["clean"])


@lru_cache(maxsize=8)
def read_tables(folder: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read sites + incidents once per folder (the benchmark loads hundreds of scenarios)."""
    d = Path(folder)
    missing = [f for f in ("sites.parquet", "incidents.parquet") if not (d / f).exists()]
    if missing:
        raise FileNotFoundError(f"{missing} missing in {d} — run first: python -m copilot.data.cleaning")
    return pd.read_parquet(d / "sites.parquet"), pd.read_parquet(d / "incidents.parquet")


def list_crises(folder: str | Path | None = None) -> list[str]:
    _, inc = read_tables(str(clean_dir(folder)))
    return sorted(inc["crisis_id"].unique().tolist())


def make_crews(sites: list[Site], start_positions: dict[str, tuple[float, float]] | None = None) -> list[Crew]:
    """One Crew per unit of n_crews: SITE_04_C1, SITE_04_C2, …"""
    start_positions = start_positions or {}
    valid = {f"{s.site_id}_C{k}" for s in sites for k in range(1, s.n_crews + 1)}
    unknown = set(start_positions) - valid
    if unknown:
        raise ValueError(f"start_positions: unknown crews {sorted(unknown)[:5]}")
    crews = []
    for s in sites:
        for k in range(1, s.n_crews + 1):
            cid = f"{s.site_id}_C{k}"
            lat, lon = start_positions.get(cid, (s.lat, s.lon))
            crews.append(Crew(crew_id=cid, site_id=s.site_id, lat=float(lat), lon=float(lon)))
    return crews


def load_scenario(crisis: str | int, params: dict | None = None, folder: str | Path | None = None,
                  ) -> tuple[list[Site], list[Crew], list[Incident]]:
    """(sites, crews, incidents) for one crisis — fresh objects at each call.

    params: {"start_positions": {crew_id: (lat, lon)}, "max_crews": int (round-robin across sites)}
    """
    params = params or {}
    cid = normalise_crisis_id(crisis)
    df_sites, df_inc = read_tables(str(clean_dir(folder)))
    sites = [Site(r.site_id, r.town, float(r.lat), float(r.lon), int(r.n_crews))
             for r in df_sites.itertuples(index=False)]
    crews = make_crews(sites, params.get("start_positions"))
    if params.get("max_crews") is not None:
        order = sorted(crews, key=lambda c: (int(c.crew_id.rsplit("_C", 1)[1]), c.site_id))
        keep = {c.crew_id for c in order[: int(params["max_crews"])]}
        crews = [c for c in crews if c.crew_id in keep]
    sel = df_inc[df_inc["crisis_id"] == cid].sort_values(["t_reported", "incident_id"])
    if sel.empty:
        raise ValueError(f"{cid} not found in incidents.parquet")
    incidents = [Incident(r.incident_id, r.substation_id, int(r.t_reported), int(r.duration_min),
                          int(r.duration_min), r.cause_en, int(r.customers), bool(r.sensitive),
                          float(r.lat), float(r.lon))
                 for r in sel.itertuples(index=False)]
    return sites, crews, incidents
