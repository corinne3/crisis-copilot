from _helpers import crews_for, incident, params, site

from copilot.kpi import compute_kpi
from copilot.policies.wspt import WSPT
from copilot.simulator import fatigue_factor, rotation_groups, simulate


def _long_day():
    return [incident(f"I{k}", t=0, duration=120, dlat=0.01 * k) for k in range(5)]


def test_fatigue_factor():
    p = params(fatigue={"enabled": True, "threshold_h": 4, "slope_per_h": 0.05, "max": 1.5})
    assert fatigue_factor(3 * 60, p) == 1.0
    assert abs(fatigue_factor(6 * 60, p) - 1.10) < 1e-9
    assert fatigue_factor(40 * 60, p) == 1.5
    assert fatigue_factor(10 * 60, params()) == 1.0            # disabled by default


def test_fixed_break_after_4h():
    s = site()
    res = simulate([s], crews_for(s), _long_day(), WSPT(), params(breaks={"enabled": True, "mode": "fixed"}))
    k = compute_kpi(res)
    assert k["n_breaks"] >= 1 and k["max_work_without_break_h"] <= 4.5
    b = res.activities[res.activities.type == "break"].iloc[0]
    assert b.t_end - b.t_start == 45
    assert k["shift_overruns"] == 0


def test_smart_break_is_opportunistic_when_nothing_waits():
    s = site()
    incs = [incident("I1", t=0, duration=240), incident("I2", t=600)]   # nothing waiting after I1
    p = params(breaks={"enabled": True, "mode": "smart"})
    res = simulate([s], crews_for(s), incs, WSPT(), p)
    assert compute_kpi(res)["n_breaks"] == 1


def test_fatigue_lengthens_repairs():
    s = site()
    base = simulate([s], crews_for(s), _long_day(), WSPT(), params())
    tired = simulate([s], crews_for(s), _long_day(), WSPT(), params(fatigue={"enabled": True}))
    assert compute_kpi(tired)["customer_hours"] > compute_kpi(base)["customer_hours"]
    assert tired.metadata["mean_fatigue"] > 1.0


def test_rotation_groups_and_standby():
    s = site(n=4)
    crews = crews_for(s)
    p = params(rotation={"enabled": True, "groups": 2, "offset_min": 240})
    g = rotation_groups(crews, p)
    assert sorted(g.values()) == [0, 0, 1, 1]
    res = simulate([s], crews, [incident("I1", t=0)], WSPT(), p)
    standby = res.activities[res.activities.type == "standby"]
    assert len(standby) == 2 and (standby.t_end <= 240).all()
