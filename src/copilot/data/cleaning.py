"""Raw data cleaning → data/clean/*.parquet.

Run:
    python -m copilot.data.cleaning
    python -m copilot.data.cleaning --raw data/raw --clean data/clean

Inputs (data/raw/, ';'-separated CSV with UTF-8 BOM, synthetic data provided by Enedis):
    agents_v2.csv (sites & crews), poste_v2.csv (substations), insee_v2.csv (towns),
    coupures_v2_100_crises.csv (100 simulated crises)

Outputs (data/clean/):
    sites.parquet, substations.parquet, towns.parquet, incidents.parquet, cleaning_report.json

What is fixed / checked:
    - everything is read as text then typed explicitly (decimal comma tolerated)
    - geographic bounding box check (Ariège, Haute-Garonne, Gers)
    - INSEE codes padded to 5 characters (9001 → 09001)
    - repair duration "4h" → 240 minutes; dates DD/MM/YYYY HH:MM
    - incidents pointing to an unknown substation (5 in the data set) are EXCLUDED and logged
    - same substation hit several times in one crisis (43 cases) is KEPT and flagged
    - French outage causes get an English label (cause_en)
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from copilot.config import load_config, project_path

log = logging.getLogger("copilot.cleaning")

LAT_MIN, LAT_MAX = 42.4, 44.3
LON_MIN, LON_MAX = -0.5, 2.4
ALLOWED_DURATIONS_MIN = {120, 240, 480}

DEFAULT_FILES = {
    "sites": "agents_v2.csv",
    "substations": "poste_v2.csv",
    "towns": "insee_v2.csv",
    "outages": "coupures_v2_100_crises.csv",
}

CAUSES_EN = {
    "Effort anormal par tempête de vent ou de pluie": "Abnormal stress from wind or rain storm",
    "Chute de branche par vent": "Branch fall due to wind",
    "Coup de foudre": "Lightning strike",
    "Défaillance de matériel : autre défaillance": "Equipment failure (other)",
    "Élagage insuffisant": "Insufficient tree trimming",
    "Usure naturelle": "Natural wear",
    "Défaillance protection": "Protection failure",
    "Cause inconnue avec circonstances atmosphériques normales": "Unknown cause, normal weather",
    "Autres travaux de tiers": "Third-party works",
    "Véhicule": "Vehicle",
}


class InvalidData(ValueError):
    """Raised when a consistency check fails (explicit message for the team)."""


# --------------------------------------------------------------------------- helpers
def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path} — put the provided files in data/raw/")
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    return df.apply(lambda s: s.str.strip())


def _check_columns(df: pd.DataFrame, expected: list[str], name: str) -> None:
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise InvalidData(f"{name}: missing columns {missing} (read: {list(df.columns)})")


def _to_float(s: pd.Series, name: str) -> pd.Series:
    out = pd.to_numeric(s.str.replace(",", ".", regex=False), errors="coerce")
    if out.isna().any():
        raise InvalidData(f"{name}: {int(out.isna().sum())} non-numeric values, e.g. {s[out.isna()].head(3).tolist()}")
    return out


def _to_int(s: pd.Series, name: str) -> pd.Series:
    return _to_float(s, name).round().astype("int64")


def _check_bbox(df: pd.DataFrame, name: str) -> None:
    out = ~(df["lat"].between(LAT_MIN, LAT_MAX) & df["lon"].between(LON_MIN, LON_MAX))
    if out.any():
        raise InvalidData(f"{name}: {int(out.sum())} rows outside the area "
                          f"(lat {LAT_MIN}–{LAT_MAX}, lon {LON_MIN}–{LON_MAX}):\n{df.loc[out].head(5)}")


def _check_unique(df: pd.DataFrame, key: str, name: str) -> None:
    dup = df[key].duplicated()
    if dup.any():
        raise InvalidData(f"{name}: duplicate key {key}, e.g. {df.loc[dup, key].head(3).tolist()}")


def _duration_minutes(s: pd.Series) -> pd.Series:
    hours = pd.to_numeric(s.str.lower().str.replace("h", "", regex=False).str.replace(",", "."), errors="coerce")
    minutes = (hours * 60).round()
    bad = minutes.isna() | ~minutes.isin(ALLOWED_DURATIONS_MIN)
    if bad.any():
        raise InvalidData(f"dureeDepannage: unexpected values {s[bad].unique()[:5].tolist()}")
    return minutes.astype("int64")


# --------------------------------------------------------------------------- readers
def read_sites(path: Path) -> pd.DataFrame:
    df = _read_csv(path)
    _check_columns(df, ["id_site", "commune", "departement", "longitude", "latitude", "nb_equipes"], "sites")
    out = pd.DataFrame({
        "site_id": df["id_site"], "town": df["commune"], "department": df["departement"],
        "lat": _to_float(df["latitude"], "sites.latitude"), "lon": _to_float(df["longitude"], "sites.longitude"),
        "n_crews": _to_int(df["nb_equipes"], "sites.nb_equipes"),
    })
    _check_unique(out, "site_id", "sites")
    _check_bbox(out, "sites")
    if (out["n_crews"] < 1).any():
        raise InvalidData("sites: a site has no crew")
    return out.sort_values("site_id").reset_index(drop=True)


def read_substations(path: Path) -> pd.DataFrame:
    df = _read_csv(path)
    _check_columns(df, ["id_poste", "code_insee", "commune", "departement", "longitude", "latitude", "x_l93",
                        "y_l93", "nb_clients", "poste_avec_client_sensible"], "substations")
    out = pd.DataFrame({
        "substation_id": df["id_poste"], "insee_code": df["code_insee"].str.zfill(5), "town": df["commune"],
        "department": df["departement"],
        "lat": _to_float(df["latitude"], "substations.latitude"),
        "lon": _to_float(df["longitude"], "substations.longitude"),
        "x_l93": _to_float(df["x_l93"], "substations.x_l93"), "y_l93": _to_float(df["y_l93"], "substations.y_l93"),
        "customers": _to_int(df["nb_clients"], "substations.nb_clients"),
        "sensitive": _to_int(df["poste_avec_client_sensible"], "substations.sensitive").astype(bool),
    })
    _check_unique(out, "substation_id", "substations")
    _check_bbox(out, "substations")
    if (out["customers"] <= 0).any():
        raise InvalidData("substations: customers must be > 0")
    return out


def read_towns(path: Path) -> pd.DataFrame:
    df = _read_csv(path)
    _check_columns(df, ["code_insee", "commune", "departement", "latitude", "longitude", "nb_postes"], "towns")
    out = pd.DataFrame({
        "insee_code": df["code_insee"].str.zfill(5), "town": df["commune"], "department": df["departement"],
        "lat": _to_float(df["latitude"], "towns.latitude"), "lon": _to_float(df["longitude"], "towns.longitude"),
        "n_substations": _to_int(df["nb_postes"], "towns.nb_postes"),
    })
    _check_unique(out, "insee_code", "towns")
    _check_bbox(out, "towns")
    return out


def read_outages(path: Path) -> pd.DataFrame:
    df = _read_csv(path)
    _check_columns(df, ["numero_crise", "id_incident", "id_poste", "causeInterruption", "dateHeureIncident",
                        "dureeDepannage"], "outages")
    date = pd.to_datetime(df["dateHeureIncident"], format="%d/%m/%Y %H:%M", errors="coerce")
    if date.isna().any():
        raise InvalidData(f"outages: unreadable dates, e.g. {df.loc[date.isna(), 'dateHeureIncident'].head(3).tolist()}")
    out = pd.DataFrame({
        "crisis_id": df["numero_crise"].str.replace("CRISE_", "CRISIS_", regex=False),
        "incident_id": df["id_incident"], "substation_id": df["id_poste"],
        "cause": df["causeInterruption"], "cause_en": df["causeInterruption"].map(CAUSES_EN).fillna("Other"),
        "reported_at": date, "duration_min": _duration_minutes(df["dureeDepannage"]),
    })
    _check_unique(out, "incident_id", "outages")
    return out


# --------------------------------------------------------------------------- assembly
def build_incidents(outages: pd.DataFrame, substations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join outages ↔ substations. Returns (valid incidents, excluded incidents)."""
    known = outages["substation_id"].isin(substations["substation_id"])
    excluded = outages.loc[~known].copy()
    inc = outages.loc[known].merge(
        substations[["substation_id", "insee_code", "town", "department", "lat", "lon", "customers", "sensitive"]],
        on="substation_id", how="left", validate="many_to_one")
    start = inc.groupby("crisis_id")["reported_at"].transform("min")
    inc["crisis_start"] = start
    inc["t_reported"] = ((inc["reported_at"] - start).dt.total_seconds() // 60).astype("int64")
    inc = inc.sort_values(["crisis_id", "reported_at", "incident_id"]).reset_index(drop=True)
    inc["rank_on_substation"] = inc.groupby(["crisis_id", "substation_id"]).cumcount() + 1
    inc["n_incidents_substation_crisis"] = inc.groupby(["crisis_id", "substation_id"])["incident_id"].transform("size")
    cols = ["crisis_id", "incident_id", "substation_id", "insee_code", "town", "department", "lat", "lon",
            "customers", "sensitive", "cause", "cause_en", "reported_at", "crisis_start", "t_reported",
            "duration_min", "rank_on_substation", "n_incidents_substation_crisis"]
    return inc[cols], excluded


def global_checks(substations, towns, incidents) -> None:
    unknown = ~substations["insee_code"].isin(towns["insee_code"])
    if unknown.any():
        raise InvalidData(f"substations: {int(unknown.sum())} INSEE codes missing from the town table")
    if incidents["t_reported"].lt(0).any():
        raise InvalidData("incidents: negative t_reported")
    if incidents[["lat", "lon", "customers"]].isna().any().any():
        raise InvalidData("incidents: missing values after the join")


def clean(raw: Path, out_dir: Path, files: dict | None = None) -> dict:
    f = {**DEFAULT_FILES, **(files or {})}
    raw, out_dir = Path(raw), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sites = read_sites(raw / f["sites"])
    substations = read_substations(raw / f["substations"])
    towns = read_towns(raw / f["towns"])
    outages = read_outages(raw / f["outages"])
    incidents, excluded = build_incidents(outages, substations)
    global_checks(substations, towns, incidents)

    sites.to_parquet(out_dir / "sites.parquet", index=False)
    substations.to_parquet(out_dir / "substations.parquet", index=False)
    towns.to_parquet(out_dir / "towns.parquet", index=False)
    incidents.to_parquet(out_dir / "incidents.parquet", index=False)

    multi = incidents[incidents["n_incidents_substation_crisis"] > 1]
    per_crisis = incidents.groupby("crisis_id").agg(n=("incident_id", "size"), window=("t_reported", "max"))
    report = {
        "sites": len(sites), "crews": int(sites["n_crews"].sum()),
        "crews_per_department": sites.groupby("department")["n_crews"].sum().astype(int).to_dict(),
        "substations": len(substations), "customers_total": int(substations["customers"].sum()),
        "sensitive_substations": int(substations["sensitive"].sum()), "towns": len(towns),
        "crises": int(incidents["crisis_id"].nunique()),
        "incidents_read": len(outages), "incidents_valid": len(incidents),
        "excluded_unknown_substation": excluded[["crisis_id", "incident_id", "substation_id"]].to_dict("records"),
        "substations_hit_several_times": int(multi.groupby(["crisis_id", "substation_id"]).ngroups),
        "incidents_per_crisis": {"min": int(per_crisis["n"].min()), "mean": round(float(per_crisis["n"].mean()), 1),
                                 "max": int(per_crisis["n"].max())},
        "reporting_window_h": {"min": round(per_crisis["window"].min() / 60, 2),
                               "mean": round(per_crisis["window"].mean() / 60, 2),
                               "max": round(per_crisis["window"].max() / 60, 2)},
        "durations_min": {str(k): int(v) for k, v in incidents["duration_min"].value_counts().sort_index().items()},
    }
    with open(out_dir / "cleaning_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    log.info("Sites: %s (%s crews) — substations: %s (%s sensitive) — towns: %s", report["sites"], report["crews"],
             report["substations"], report["sensitive_substations"], report["towns"])
    log.info("Incidents: %s read, %s valid, %s excluded (unknown substation)", report["incidents_read"],
             report["incidents_valid"], len(report["excluded_unknown_substation"]))
    log.info("Files written to %s", out_dir)
    return report


def main(argv: list[str] | None = None) -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Clean the Enedis challenge data")
    ap.add_argument("--raw", default=str(project_path(cfg["paths"]["raw"])))
    ap.add_argument("--clean", default=str(project_path(cfg["paths"]["clean"])))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    clean(Path(args.raw), Path(args.clean), cfg.get("raw_files"))


if __name__ == "__main__":
    main()
