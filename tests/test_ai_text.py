import json

from _helpers import HAS_DATA
from test_ai_llm import FakeClient

from copilot.ai import field_reports as FR
from copilot.ai import whatif as W


# ----------------------------------------------------------------- field reports
def test_rules_extract_messy_messages():
    u = FR.extract_rules("inc 45 worse than expected, need 3 more hours")
    assert (u.type, u.incident_id, u.extra_minutes) == ("duration_update", "INC_000045", 180)
    u = FR.extract_rules("SITE_04_C2: tree across the road, cannot reach INC_000077")
    assert (u.type, u.incident_id, u.crew_id) == ("access_blocked", "INC_000077", "SITE_04_C2")
    assert FR.minutes_in("1h30 more") == 90 and FR.minutes_in("two more hours") == 120


def test_rules_are_perfect_on_templates_but_weak_on_real_phrasing():
    assert FR.evaluate(FR.synthetic_messages(60), FR.extract_rules)["all"] == 100
    assert FR.evaluate(FR.HARD_MESSAGES, FR.extract_rules)["type"] < 60


def test_validation_blocks_inventions_and_unknown_ids():
    msg = "INC_000045 need 2 more hours"
    good = json.dumps({"type": "duration_update", "incident_id": "INC_000045", "extra_minutes": 120})
    u = FR.understand(msg, FakeClient([good]), known_incidents={"INC_000045"})
    assert u.source == "llm" and u.status == "ok"
    invented = json.dumps({"type": "duration_update", "incident_id": "INC_000045", "extra_minutes": 300})
    u = FR.understand(msg, FakeClient([invented]), known_incidents={"INC_000045"})
    assert u.status == "needs_review" and "invention" in "; ".join(u.issues)
    u = FR.understand(msg, None, known_incidents={"INC_000001"})
    assert u.source == "rules" and u.status == "needs_review"


def test_safety_is_never_automated():
    u = FR.understand("DANGER live wire on the road near INC_000318")
    assert u.type == "safety_hazard" and u.status == "manager_must_handle"


def test_apply_updates():
    from _helpers import incident

    incs = [incident("INC_000001"), incident("INC_000002")]
    ups = [FR.Update("duration_update", "INC_000001", extra_minutes=60, status="ok"),
           FR.Update("access_blocked", "INC_000002", status="ok"),
           FR.Update("duration_update", "INC_000002", extra_minutes=999, status="needs_review")]
    incs, zones = FR.apply_updates(incs, ups)
    assert incs[0].nominal_duration == 180 and incs[1].nominal_duration == 120
    assert len(zones) == 1 and zones[0].contains(incs[1].lat, incs[1].lon)


# ----------------------------------------------------------------- what-if
def test_whatif_rules():
    sc = W.understand("What if we lose 5 crews in Gers from H+6 to H+18?")
    assert sc.changes == [{"kind": "remove_crews", "count": 5, "from_h": 6.0, "to_h": 18.0, "department": "Gers"}]
    sc = W.understand("crisis 27: rotation with 3 groups 4h apart, and repairs take 20% longer")
    kinds = {c["kind"]: c for c in sc.changes}
    assert sc.crisis == 27 and kinds["rotation"]["groups"] == 3 and kinds["repair_time"]["factor"] == 1.2
    assert W.understand("tell me a joke").issues


def test_whatif_llm_checked():
    q = "What if we lose 5 crews in Gers?"
    ok = json.dumps({"crisis": None, "changes": [{"kind": "remove_crews", "count": 5, "department": "gers"}]})
    sc = W.understand(q, FakeClient([ok]))
    assert sc.source == "llm" and sc.changes[0]["department"] == "Gers" and not sc.issues
    bad = json.dumps({"changes": [{"kind": "remove_crews", "count": 12, "department": "Gers"},
                                  {"kind": "teleport"}]})
    sc = W.understand(q, FakeClient([bad]))                       # flagged LLM reading → clean rules reading wins
    assert sc.source == "rules" and sc.changes[0]["count"] == 5
    sc = W.validate(W.parse_llm(q, FakeClient([bad])), q)
    assert any("not in the question" in i for i in sc.issues) and any("teleport" in i for i in sc.issues)


if HAS_DATA:
    def test_whatif_runs_on_real_crisis():
        r = W.run_whatif(24, W.understand("add 10 extra crews in Haute-Garonne"))
        assert r["whatif"]["customer_hours"] < r["base"]["customer_hours"]
        assert "customer-hours" in r["answer"]


def test_llm_incident_id_must_come_from_the_message():
    msg = "on 4512 now, count half a day more"
    bad = json.dumps({"type": "duration_update", "incident_id": "INC_000451", "extra_minutes": 360})
    u = FR.understand(msg, FakeClient([bad]))
    assert u.status == "needs_review" and "does not appear" in "; ".join(u.issues)


def test_whatif_cross_check_keeps_the_rule_direction():
    q = "What if repairs take 20% longer and travel is 10% slower?"
    wrong = json.dumps({"changes": [{"kind": "repair_time", "factor": 1.2}, {"kind": "travel_speed", "factor": 0.9}]})
    sc = W.understand(q, FakeClient([wrong]))
    f = {c["kind"]: c["factor"] for c in sc.changes}
    assert f == {"repair_time": 1.2, "travel_speed": 1.1} and any("rule value kept" in i for i in sc.issues)
