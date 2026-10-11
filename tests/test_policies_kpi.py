import json

import pytest
from _helpers import HAS_DATA, crews_for, incident, params, site

from copilot.kpi import compute_kpi, train_test_split, weighted_gini
from copilot.policies.wspt import WSPT, resolve_strategy
from copilot.simulator import simulate
from copilot.templates import decision_card, situation_briefing


def test_wspt_prefers_sensitive_and_short_jobs():
    s = site()
    incs = [incident("LONG", customers=300, duration=480), incident("SHORT", customers=200, duration=120)]
    res = simulate([s], crews_for(s), incs, WSPT(), params())
    first = res.incidents.sort_values("t_assigned").iloc[0]
    assert first.incident_id == "SHORT"                       # 200/120 > 300/480
    incs = [incident("N", customers=300), incident("S", customers=100, sensitive=True)]
    res = simulate([s], crews_for(s), incs, WSPT(), params())
    assert res.incidents.sort_values("t_assigned").iloc[0].incident_id == "S"   # ×(1+5)


def test_alpha_zero_ignores_sensitive():
    s = site()
    incs = [incident("N", customers=300), incident("S", customers=100, sensitive=True)]
    res = simulate([s], crews_for(s), incs, WSPT(), params(alpha_sensitive=0.0))
    assert res.incidents.sort_values("t_assigned").iloc[0].incident_id == "N"


def test_explanation_and_card():
    s = site()
    incs = [incident("A", customers=500), incident("B", customers=100, dlat=0.1)]
    res = simulate([s], crews_for(s), incs, WSPT(), params())
    d = res.decisions.iloc[0]
    e = json.loads(d.explanation)
    assert e["factors"]["customers"] == 500
    assert e["alternative"]["incident_id"] == "B" and e["alternative"]["delta_customer_hours"] > 0
    assert 0.5 <= e["confidence"] <= 1
    card = decision_card(d.crew_id, d.incident_id, e)
    assert "500 customers" in card and "B" in card


def test_kpi_values():
    s = site()
    res = simulate([s], crews_for(s), [incident("A", customers=60)], WSPT(), params())
    k = compute_kpi(res)
    hours = (res.incidents.t_restored.iloc[0] - 0) / 60
    assert k["customer_hours"] == pytest.approx(60 * hours)
    assert k["pct_restored_8h"] == 100
    txt = situation_briefing(0, res.series.iloc[0].to_dict(), k)
    assert "60 customers without power" in txt


def test_gini_and_split():
    assert weighted_gini([1, 1, 1]) == pytest.approx(0, abs=1e-9)
    assert weighted_gini([0, 0, 1]) > 0.5
    tr, te = train_test_split([f"CRISIS_{k:03d}" for k in range(100, 0, -1)])
    assert tr[0] == "CRISIS_001" and te[0] == "CRISIS_081" and len(te) == 20


def test_resolve_strategy():
    assert resolve_strategy("wspt") is WSPT
    with pytest.raises(ValueError):
        resolve_strategy("nope")


@pytest.mark.skipif(not HAS_DATA, reason="clean data missing (python -m copilot.data.cleaning)")
def test_reference_numbers_crisis_24():
    from copilot.simulator import simulate_crisis

    k = compute_kpi(simulate_crisis(24, WSPT()))
    assert k["customer_hours"] == pytest.approx(199004.57, abs=1)
    assert k["shift_overruns"] == 0 and k["unresolved"] == 0
