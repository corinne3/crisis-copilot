import numpy as np
import pandas as pd

from copilot.ai import forecast as F


def fake_incidents(n_crises=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_crises):
        n = int(rng.integers(50, 250))
        for k in range(n):
            rows.append({"crisis_id": f"CRISIS_{c + 1:03d}", "incident_id": f"I{c}_{k}",
                         "t_reported": int(rng.triangular(0, 60, 400)), "customers": int(rng.integers(5, 200)),
                         "sensitive": bool(rng.random() < 0.05), "duration_min": int(rng.choice([120, 240, 480])),
                         "cause_en": "Lightning strike", "department": rng.choice(list(F.DEPARTMENTS))})
    return pd.DataFrame(rows)


def test_features_only_use_the_past():
    inc = fake_incidents(1)
    f = F.snapshot_features(inc, 60)
    assert f["n_so_far"] == int((inc.t_reported <= 60).sum())
    assert set(F.FEATURES) <= set(f)


def test_fit_predict_evaluate():
    inc = fake_incidents()
    crises = sorted(inc.crisis_id.unique())
    end_h = pd.Series(np.linspace(20, 50, len(crises)), index=crises)
    train, test = F.build_dataset(inc, crises[:24], end_h), F.build_dataset(inc, crises[24:], end_h)
    m = F.ForecastModel().fit(train)
    p = m.predict(test)
    assert (p["total_incidents_low"] <= p["total_incidents_p50"]).all()
    assert (p["total_incidents_p50"] <= p["total_incidents_high"]).all()
    assert (p["total_incidents_low"] >= test["n_so_far"]).all()          # never below what is already known
    met = m.evaluate(test)
    assert set(met["target"]) == set(F.TARGETS) and (met["mae_model"] >= 0).all()


def test_recommendations_rules():
    pred = {"total_incidents_p50": 200, "total_incidents_low": 150, "total_incidents_high": 260,
            "total_work_h_p50": 900, "total_work_h_low": 700, "total_work_h_high": 1500,
            "end_h_p50": 55, "end_h_high": 60}
    seen = pd.DataFrame({"department": ["Gers"] * 8 + ["Haute-Garonne"] * 2, "duration_min": [240] * 10})
    recs = F.recommendations(pred, seen, {"Ariège": 14, "Gers": 14, "Haute-Garonne": 35})
    titles = " | ".join(r.title for r in recs)
    assert "second shift" in titles and "reinforcement" in titles and "Gers is under pressure" in titles
