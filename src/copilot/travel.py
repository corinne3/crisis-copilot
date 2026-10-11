"""Travel times: great-circle distance × detour factor, speed depends on the destination (plain / mountain)."""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

TravelMin = Callable[[float, float, float, float], float]


@dataclass(frozen=True)
class SlowZone:
    """Area where travel is slowed down (stress test "roads cut"); applies when the DESTINATION is inside."""
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    factor: float = 3.0
    name: str = ""

    def contains(self, lat: float, lon: float) -> bool:
        return self.lat_min <= lat <= self.lat_max and self.lon_min <= lon <= self.lon_max


MOUNTAIN_ZONE = SlowZone(42.4, 43.05, -0.5, 2.4, factor=3.0, name="Pyrenees (lat < 43.05)")


def zone_factor(lat: float, lon: float, zones: Sequence[SlowZone] = ()) -> float:
    return max((z.factor for z in zones if z.contains(lat, lon)), default=1.0)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    if math.isclose(lat1, lat2, abs_tol=1e-7) and math.isclose(lon1, lon2, abs_tol=1e-7):
        return 0.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * math.atan2(math.sqrt(min(1.0, a)), math.sqrt(max(0.0, 1 - a)))


def travel_minutes(lat1: float, lon1: float, lat2: float, lon2: float, detour_factor: float = 1.3,
                   speed_plain_kmh: float = 55.0, speed_mountain_kmh: float = 35.0,
                   mountain_lat_threshold: float = 43.05, multiplier: float = 1.0,
                   zones: Sequence[SlowZone] = ()) -> float:
    d = haversine_km(lat1, lon1, lat2, lon2)
    if d == 0.0:
        return 0.0
    speed = speed_mountain_kmh if lat2 < mountain_lat_threshold else speed_plain_kmh
    speed = max(1.0, speed / (multiplier * zone_factor(lat2, lon2, zones)))
    return d * detour_factor / speed * 60.0


def travel_params(cfg: dict) -> dict:
    t = cfg.get("travel", cfg)
    return {k: float(t[k]) for k in ("detour_factor", "speed_plain_kmh", "speed_mountain_kmh",
                                     "mountain_lat_threshold")}


def make_travel(cfg: dict | None = None, multiplier: float = 1.0, zones: Sequence[SlowZone] = ()) -> TravelMin:
    """Build the travel_min(lat1, lon1, lat2, lon2) function given to strategies."""
    if cfg is None:
        from copilot.config import load_config

        cfg = load_config()
    p, zones = travel_params(cfg), tuple(zones)

    def _travel(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        return travel_minutes(lat1, lon1, lat2, lon2, **p, multiplier=multiplier, zones=zones)

    _travel.params = {**p, "multiplier": multiplier, "zones": zones}
    return _travel


def travel_matrix(crews, incidents, cfg: dict | None = None) -> np.ndarray:
    """Vectorised crews × incidents travel-time matrix (minutes), no slow zones."""
    if cfg is None:
        from copilot.config import load_config

        cfg = load_config()
    p = travel_params(cfg)
    if not crews or not incidents:
        return np.zeros((len(crews), len(incidents)))
    la1 = np.radians([c.lat for c in crews])[:, None]
    lo1 = np.radians([c.lon for c in crews])[:, None]
    lat2 = np.array([i.lat for i in incidents])[None, :]
    la2, lo2 = np.radians(lat2), np.radians([i.lon for i in incidents])[None, :]
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    d = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    speed = np.where(lat2 < p["mountain_lat_threshold"], p["speed_mountain_kmh"], p["speed_plain_kmh"])
    return d * p["detour_factor"] / speed * 60.0
