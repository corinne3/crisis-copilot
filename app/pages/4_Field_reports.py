"""Brick ② — Field reports: crews' free-text messages → checked updates → the plan adapts."""
from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from copilot.ai import field_reports as FR
from copilot.models import load_scenario
from copilot.ui import ai_status_sidebar, crisis_selector, fmt_int, require_data

logging.getLogger("copilot.simulator").setLevel(logging.ERROR)
st.set_page_config(page_title="Field reports", layout="wide")
st.title("② Field reports — understand what the crews write")
st.caption("A small local LLM turns messy messages into structured updates; deterministic checks decide if an "
           "update can be used (known incident, plausible duration, no invented number). Safety hazards and "
           "requests for help are never automated. Without the LLM, keyword rules do the job (less well).")
require_data()
with st.sidebar:
    crisis = crisis_selector(key="fr_crisis")
client = ai_status_sidebar()

_, crews, incs = load_scenario(crisis)
ids, crew_ids = [i.incident_id for i in incs], [c.crew_id for c in crews]
sample = [f"{ids[3]} worse than expected, need 4 more hours",
          f"{crew_ids[7]}: tree across the road, cannot reach {ids[12]}",
          f"customers on {ids[1]} have light again",
          f"DANGER live wire on the road near {ids[5]}",
          f"{ids[20]} transformer burnt, 6h extra at least",
          f"on {int(ids[30].split('_')[1])} now. it's a mess, count half a day more",
          f"{ids[25]} the pole must be replaced, we can't lift it with two guys",
          "we're at the depot, coffee then off again"]
text = st.text_area("Messages received (one per line)", "\n".join(sample), height=210)
messages = [m.strip() for m in text.splitlines() if m.strip()]

if st.button("📨 Read the messages", type="primary") or "fr_updates" not in st.session_state:
    with st.spinner("Reading…"):
        st.session_state["fr_updates"] = [FR.understand(m, client, set(ids), set(crew_ids)) for m in messages]
        st.session_state["fr_messages"] = messages
updates = st.session_state["fr_updates"]

STATUS = {"ok": "✅ usable", "needs_review": "🟠 needs review", "manager_must_handle": "🔴 manager must handle"}
table = pd.DataFrame([{"message": m, **u.as_dict()} for m, u in zip(st.session_state["fr_messages"], updates,
                                                                    strict=True)])
table["status"] = table["status"].map(STATUS)
table.insert(0, "apply", [u.status == "ok" and u.type in ("duration_update", "access_blocked") for u in updates])
edited = st.data_editor(
    table[["apply", "message", "type", "incident_id", "crew_id", "extra_minutes", "status", "issues", "source"]],
    use_container_width=True, hide_index=True, disabled=["message", "status", "issues", "source", "type"],
    column_config={"apply": st.column_config.CheckboxColumn("apply to plan", help="Only usable updates are applied")})
st.caption("You can correct the incident id or the minutes before applying. 🔴 lines go straight to the manager: "
           "the copilot never decides on safety.")

if any(u.status == "manager_must_handle" for u in updates):
    for u, m in zip(updates, st.session_state["fr_messages"], strict=True):
        if u.status == "manager_must_handle":
            st.error(f"🔴 {u.type.replace('_', ' ')} — {m}")

if st.button("🔁 Re-plan the crisis with the ticked updates"):
    chosen = []
    for row, u in zip(edited.itertuples(), updates, strict=True):
        if not row.apply:
            continue
        v = FR.validate(FR.Update(u.type, row.incident_id or None, row.crew_id or None,
                                  None if pd.isna(row.extra_minutes) else int(row.extra_minutes), u.summary, u.source),
                        row.message, set(ids), set(crew_ids), checked_by_human=True)
        if v.status == "ok":
            chosen.append(v)
        else:
            st.warning(f"Not applied ({'; '.join(v.issues)}): {row.message}")
    if chosen:
        with st.spinner("Re-planning…"):
            r = FR.replan_impact(crisis, chosen)
        a, b = r["before"], r["after"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Customer-hours outlook", fmt_int(b["customer_hours"]), fmt_int(r["customer_hours_change"]),
                  delta_color="inverse")
        m2.metric("95 % restored at", f"H+{b['restored_95pct_h']:.1f}",
                  f"{b['restored_95pct_h'] - a['restored_95pct_h']:+.1f} h", delta_color="inverse")
        m3.metric("Assignments changed", len(r["changed_assignments"]))
        st.caption(f"{len(chosen)} update(s) applied. Simplification: the plan is recomputed as if the information "
                   "had been known from the start. Small changes ripple: one longer repair moves many later "
                   "assignments.")
    else:
        st.info("Nothing to apply.")

with st.expander("Is the LLM worth it? Accuracy on labelled messages"):
    r_tpl = FR.evaluate(FR.synthetic_messages(), FR.extract_rules)
    r_hard = FR.evaluate(FR.HARD_MESSAGES, FR.extract_rules)
    rows = [{"extractor": "keyword rules", "set": "120 template messages", **r_tpl},
            {"extractor": "keyword rules", "set": "12 real-life phrasings", **r_hard}]
    if client is not None and st.button("Measure the local LLM on the 12 real-life phrasings (≈ 30 s)"):
        rows.append({"extractor": f"LLM {client.model}", "set": "12 real-life phrasings",
                     **FR.evaluate(FR.HARD_MESSAGES, lambda m: FR.extract_llm(m, client) or FR.extract_rules(m))})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("% of messages where the field is right ('all' = every field right). Rules are perfect on the "
               "phrasings they were written for and weak on anything else: that gap is the only reason to use an LLM.")
