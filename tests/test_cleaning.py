from pathlib import Path

import pandas as pd
import pytest

from copilot.config import load_config, project_path
from copilot.data import cleaning as C

BOM = "﻿"
RAW = project_path(load_config()["paths"]["raw"])
HAS_RAW = all((RAW / f).exists() for f in C.DEFAULT_FILES.values())


def _write(folder: Path, name: str, lines: list[str]) -> None:
    (folder / name).write_text(BOM + "\r\n".join(lines) + "\r\n", encoding="utf-8")


def raw_mini(folder: Path) -> Path:
    _write(folder, "agents_v2.csv", [
        "id_site;commune;departement;longitude;latitude;nb_equipes",
        "SITE_01;Foix;Ariège;1.6060;42.9654;2",
        "SITE_02;Auch;Gers;0,5855;43.6464;1",          # decimal comma accepted
    ])
    _write(folder, "poste_v2.csv", [
        "id_poste;code_insee;commune;code_departement;departement;region;longitude;latitude;x_l93;y_l93;"
        "nb_clients;poste_avec_client_sensible",
        "POSTE_000001;9122;Foix;9;Ariège;Occitanie;1.60;42.96;575000;6205000;100;1",
        "POSTE_000002;32013;Auch;32;Gers;Occitanie;0.58;43.64;505000;6283000;40;0",
    ])
    _write(folder, "insee_v2.csv", [
        "code_insee;commune;code_departement;departement;latitude;longitude;nb_postes",
        "09122;Foix;09;Ariège;42.96;1.60;1",
        "32013;Auch;32;Gers;43.64;0.58;1",
    ])
    _write(folder, "coupures_v2_100_crises.csv", [
        "numero_crise;id_incident;id_poste;causeInterruption;dateHeureIncident;dureeDepannage",
        "CRISE_001;INC_1;POSTE_000001;Coup de foudre;09/01/2016 14:08;4h",
        "CRISE_001;INC_2;POSTE_000002;Chute de branche par vent;09/01/2016 15:38;2h",
        "CRISE_001;INC_3;POSTE_000001;Coup de foudre;09/01/2016 18:08;8h",   # same substation, same crisis
        "CRISE_001;INC_4;POSTE_999999;Véhicule;09/01/2016 16:00;2h",         # unknown substation → excluded
        "CRISE_002;INC_5;POSTE_000002;Usure naturelle;10/02/2016 00:30;4h",
    ])
    return folder


def test_pipeline_mini(tmp_path: Path):
    (tmp_path / "raw").mkdir()
    raw = raw_mini(tmp_path / "raw")
    out = tmp_path / "clean"
    r = C.clean(raw, out)
    assert r["sites"] == 2 and r["crews"] == 3
    assert r["incidents_read"] == 5 and r["incidents_valid"] == 4
    assert [e["substation_id"] for e in r["excluded_unknown_substation"]] == ["POSTE_999999"]
    assert r["substations_hit_several_times"] == 1
    for f in ["sites", "substations", "towns", "incidents"]:
        assert (out / f"{f}.parquet").exists()

    sites = pd.read_parquet(out / "sites.parquet")
    assert sites.loc[sites.site_id == "SITE_02", "lon"].item() == pytest.approx(0.5855)
    subs = pd.read_parquet(out / "substations.parquet")
    assert subs.loc[subs.substation_id == "POSTE_000001", "insee_code"].item() == "09122"
    inc = pd.read_parquet(out / "incidents.parquet").set_index("incident_id")
    assert inc.loc["INC_1", "crisis_id"] == "CRISIS_001"
    assert inc.loc["INC_2", "t_reported"] == 90             # 15:38 − 14:08
    assert inc.loc["INC_3", "duration_min"] == 480
    assert inc.loc["INC_3", "rank_on_substation"] == 2
    assert inc.loc["INC_5", "t_reported"] == 0              # relative to ITS crisis
    assert inc.loc["INC_1", "cause_en"] == "Lightning strike"
    assert bool(inc.loc["INC_1", "sensitive"])


def test_invalid_duration():
    with pytest.raises(C.InvalidData):
        C._duration_minutes(pd.Series(["3h"]))


@pytest.mark.skipif(not HAS_RAW, reason="challenge data not in data/raw")
def test_real_data(tmp_path: Path):
    r = C.clean(RAW, tmp_path)
    assert r["sites"] == 18 and r["crews"] == 63
    assert r["crises"] == 100 and r["incidents_valid"] == 15676
