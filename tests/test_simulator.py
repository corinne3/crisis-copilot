import pytest
from _helpers import crews_for, incident, params, site

from copilot.kpi import compute_kpi
from copilot.models import Crew
from copilot.policies.baselines import MostCustomers, NearestFCFS
from copilot.policies.wspt import WSPT
from copilot.simulator import MIN_REPAIR_MIN, Unavailability, is_feasible, simulate


def test_single_incident_restored_after_travel_and_repair():
    s = site()
    res = simulate([s], crews_for(s), [incident("I1", t=10, dlat=0.1)], NearestFCFS(), params())
    r = res.incidents.iloc[0]
    assert r.t_assigned == 10 and r.t_arrival > 10
    assert r.t_restored == r.t_arrival + 120
    assert res.metadata["unresolved"] == 0
    assert set(res.activities["type"]) == {"travel", "repair"}


def test_one_crew_one_incident_at_a_time():
    s = site()
    incs = [incident(f"I{k}", t=0, customers=10 * k) for k in range(1, 4)]
    res = simulate([s], crews_for(s), incs, MostCustomers(), params())
    order = res.incidents.sort_values("t_assigned")["incident_id"].tolist()
    assert order == ["I3", "I2", "I1"]                          # most customers first
    rep = res.activities[res.activities.type == "repair"].sort_values("t_start")
    assert (rep["t_start"].iloc[1:].values >= rep["t_end"].iloc[:-1].values).all()


def test_shift_rule_and_rest():
    s = site()
    incs = [incident(f"I{k}", t=0, duration=240) for k in range(4)]   # 16h of work for 1 crew
    res = simulate([s], crews_for(s), incs, NearestFCFS(), params())
    k = compute_kpi(res)
    assert k["unresolved"] == 0 and k["shift_overruns"] == 0
    assert "rest" in set(res.activities["type"])
    rest = res.activities[res.activities.type == "rest"].iloc[0]
    assert rest.t_end - rest.t_start == 660


def test_is_feasible():
    p = params()
    c = Crew("SITE_01_C1", "SITE_01", 43.6, 1.44, t_shift_start=0)
    i = incident("I1", duration=480)
    assert is_feasible(c, i, 30, 0, p)
    assert not is_feasible(c, i, 30, 720 - 480, p)
    assert not is_feasible(c, i, 0, 0, {**p, "forbidden": {("SITE_01_C1", "I1")}})
    assert MIN_REPAIR_MIN == 120


def test_forced_and_forbidden():
    s = site(n=2)
    incs = [incident("BIG", customers=1000), incident("SMALL", customers=1, dlat=0.2)]
    res = simulate([s], crews_for(s), incs, WSPT(), params(), forced={"SITE_01_C2": "SMALL"},
                   forbidden=[("SITE_01_C1", "SMALL")])
    got = res.incidents.set_index("incident_id")["crew_id"].to_dict()
    assert got == {"BIG": "SITE_01_C1", "SMALL": "SITE_01_C2"}
    import json
    origins = {d.incident_id: json.loads(d.explanation)["origin"] for d in res.decisions.itertuples()}
    assert origins["SMALL"] == "human"


def test_forbidden_pair_does_not_deadlock():
    s = site(n=2)
    incs = [incident("A", customers=500), incident("B", customers=400, dlat=0.05)]
    res = simulate([s], crews_for(s), incs, WSPT(), params(), forbidden=[("SITE_01_C1", "A")])
    assert res.metadata["unresolved"] == 0
    assert res.incidents.set_index("incident_id").loc["A", "crew_id"] == "SITE_01_C2"


def test_unavailability_finishes_current_repair_first():
    s = site()
    incs = [incident("I1", t=0, duration=240), incident("I2", t=10)]
    res = simulate([s], crews_for(s), incs, NearestFCFS(), params(),
                   unavailability=[Unavailability("SITE_01_C1", 60, 600)])
    r = res.incidents.set_index("incident_id")
    assert r.loc["I1", "t_restored"] == 240                     # safety: repair not abandoned
    assert r.loc["I2", "t_assigned"] >= 600
    assert "out_of_service" in set(res.activities["type"])


def test_monte_carlo_is_reproducible_and_changes_durations():
    s = site(n=2)
    incs = lambda: [incident(f"I{k}", t=10 * k, dlat=0.05 * k) for k in range(6)]  # noqa: E731
    p = params(monte_carlo={"enabled": True}, seed=7)
    a = simulate([s], crews_for(s), incs(), WSPT(), p).incidents["actual_duration"].tolist()
    b = simulate([s], crews_for(s), incs(), WSPT(), p).incidents["actual_duration"].tolist()
    assert a == b and a != [120] * 6


def test_series_counts_each_substation_once():
    s = site(n=2)
    i1, i2 = incident("I1", customers=50), incident("I2", customers=50)
    i2.substation_id = i1.substation_id
    res = simulate([s], crews_for(s), [i1, i2], NearestFCFS(), params())
    assert res.series.loc[0, "customers_off"] == 50


def test_duplicate_ids_rejected():
    s = site()
    with pytest.raises(ValueError):
        simulate([s], crews_for(s), [incident("I1"), incident("I1")], NearestFCFS(), params())
