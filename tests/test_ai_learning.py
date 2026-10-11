import json

import numpy as np
import pandas as pd
from _helpers import crews_for, incident, params, site
from test_ai_llm import FakeClient

from copilot.ai import learning as L
from copilot.simulator import simulate


def synthetic_log(n=600, alpha=10.0, beta=0.5, seed=1):
    rng = np.random.default_rng(seed)
    mgr = L.SimulatedManager(alpha, beta, temperature=0.1, seed=seed)
    rows = []
    for d in range(n):
        k = int(rng.integers(2, 6))
        cands = [{"incident_id": f"I{d}_{j}", "customers": int(rng.integers(10, 800)),
                  "sensitive": bool(rng.random() < 0.2), "wait_min": int(rng.integers(0, 600)),
                  "travel_min": float(rng.integers(5, 90)), "duration_min": int(rng.choice([120, 240, 480]))}
                 for j in range(k)]
        rec = int(np.argmax(L.log_scores(pd.DataFrame(cands), 5.0, 0.1)))
        ch = mgr.choose(cands)
        rows.append({"source": "test", "crisis": "X", "t": 0, "crew_id": "C", "recommended": cands[rec]["incident_id"],
                     "chosen": cands[ch]["incident_id"], "accepted": ch == rec,
                     "reason": "" if ch == rec else mgr.reason(cands[rec], cands[ch]), "candidates": json.dumps(cands)})
    return pd.DataFrame(rows, columns=L.LOG_COLUMNS)


def test_fit_recovers_hidden_preferences():
    df = synthetic_log()
    fit = L.fit_preferences(df, n_boot=5)
    assert abs(fit.alpha - 10) < 3 and abs(fit.beta - 0.5) < 0.2
    assert fit.alpha_ci[0] <= fit.alpha <= fit.alpha_ci[1] + 1e-6
    assert L.agreement_rate(df, fit.alpha, fit.beta) > L.agreement_rate(df, 5.0, 0.1)


def test_not_enough_decisions():
    import pytest

    with pytest.raises(ValueError):
        L.fit_preferences(synthetic_log(n=5), n_boot=0)


def test_group_reasons_keywords_and_llm():
    reasons = ["Hospital on INC_1", "waiting for ages", "too far for this crew", "gut feeling"]
    g, src = L.group_reasons(reasons)
    assert src == "keywords" and g["sensitive customers"] == [0] and g["other"] == [3]
    good = json.dumps({"themes": [{"name": "medical", "ids": [0]}, {"name": "rest", "ids": [1, 2, 3]}]})
    g, src = L.group_reasons(reasons, FakeClient([good]))
    assert src == "llm" and g["medical"] == [0]
    bad = json.dumps({"themes": [{"name": "medical", "ids": [0, 0]}]})       # ids missing / duplicated
    assert L.group_reasons(reasons, FakeClient([bad]))[1] == "keywords"


def test_manager_in_the_loop_logs_every_decision():
    s = site(n=2)
    incs = [incident(f"I{k}", t=5 * k, customers=50 + 40 * k, sensitive=k == 2, dlat=0.03 * k) for k in range(6)]
    recs = []
    res = simulate([s], crews_for(s), incs, L.ManagerInTheLoop(L.SimulatedManager(seed=3), records=recs), params())
    assert res.metadata["unresolved"] == 0
    assert len(recs) == len(res.decisions)
    for r in recs:
        assert r["chosen"] in [c["incident_id"] for c in json.loads(r["candidates"])]


def test_versions(tmp_path):
    fit = L.Fit(9.5, 0.45, 10.0, 500, 120)
    assert L.approved_score(tmp_path) is None
    L.save_version(fit, None, "proposed", folder=tmp_path)
    L.save_version(fit, None, "approved", approved_by="expert", folder=tmp_path)
    assert [v["version"] for v in L.list_versions(tmp_path)] == [1, 2]
    assert L.approved_score(tmp_path) == {"alpha_sensitive": 9.5, "beta_wait_per_h": 0.45, "version": 2}


def test_record_human_override(tmp_path):
    from copilot.policies.wspt import WSPT

    s = site(n=2)
    incs = lambda: [incident("A", customers=500), incident("B", customers=100, dlat=0.05)]  # noqa: E731
    res = simulate([s], crews_for(s), incs(), WSPT(), params(), forbidden=[("SITE_01_C1", "A")])
    path = tmp_path / "log.csv"
    row = L.record_human_override(res, "X", "SITE_01_C1", 0, "A", "nursing home on B", path=path)
    assert row["chosen"] == "B" and row["accepted"] is False
    df = L.read_decision_log(path)
    assert len(df) == 1 and df.loc[0, "reason"] == "nursing home on B" and not df.loc[0, "accepted"]


def test_group_reasons_repairs_small_gaps():
    reasons = [f"hospital {k}" for k in range(10)]
    almost = json.dumps({"themes": [{"name": "medical", "ids": list(range(9)) + [3, 42]}]})   # 9 missing, dup, out of range
    g, src = L.group_reasons(reasons, FakeClient([almost]))
    assert src == "llm" and sorted(g["medical"]) == list(range(9)) and g["other"] == [9]
    half = json.dumps({"themes": [{"name": "medical", "ids": [0, 1, 2]}]})
    assert L.group_reasons(reasons, FakeClient([half]))[1] == "keywords"
