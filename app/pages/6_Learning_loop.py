"""Brick ④ — Learn from overrides: what priorities does the manager really apply? An expert validates."""
from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from copilot.ai import learning as L
from copilot.ui import ai_status_sidebar, fmt_int, require_data

logging.getLogger("copilot.simulator").setLevel(logging.ERROR)
st.set_page_config(page_title="Learning loop", layout="wide")
st.title("④ Learning loop — learn from the manager's overrides")
st.caption("Every accept / refuse is logged. A small statistical model (conditional logit, scipy, < 1 s) finds the "
           "two weights of the score that best explain the manager's choices: α (how much a sensitive customer "
           "counts) and β (how much waiting counts). The proposal is replayed on crises never used for learning, "
           "with hard safety checks, and applied only if an expert approves it. Nothing changes silently.")
require_data()
client = ai_status_sidebar()

human_log = L.read_decision_log()
with st.sidebar:
    src = st.radio("Decisions to learn from", ["simulated manager", f"my logged decisions ({len(human_log)})"])
    if src == "simulated manager":
        st.caption("No real manager data yet: a simulated manager with HIDDEN preferences plays the human. "
                   "Can the loop find them back?")
        h_alpha = st.slider("Hidden α of the manager", 0.0, 20.0, 10.0, 0.5)
        h_beta = st.slider("Hidden β of the manager", 0.0, 1.5, 0.5, 0.05)
        noise = st.slider("Decision noise", 0.0, 0.3, 0.08, 0.02)
        n_crises = st.slider("Training crises (from 1–80)", 5, 40, 20, 5)


@st.cache_data(show_spinner=False)
def collect(alpha: float, beta: float, noise: float, n: int) -> pd.DataFrame:
    return L.collect_log(list(range(1, n + 1)), L.SimulatedManager(alpha, beta, noise))


@st.cache_data(show_spinner=False)
def fit(df: pd.DataFrame) -> L.Fit:
    return L.fit_preferences(df, n_boot=10)


if src == "simulated manager":
    with st.spinner(f"Playing {n_crises} crises with the manager in the loop…"):
        df = collect(h_alpha, h_beta, noise, n_crises)
else:
    df = human_log
    if len(df) < 20:
        st.info("Fewer than 20 logged decisions: refuse some recommendations in the Crisis room (with a reason) "
                "or use the simulated manager.")
        st.stop()

# --------------------------------------------------------------------------- 1. the log
st.subheader("1 · Decision log")
n_over = int((~df["accepted"].astype(bool)).sum())
c1, c2, c3 = st.columns(3)
c1.metric("Decisions logged", fmt_int(len(df)))
c2.metric("Overrides", fmt_int(n_over), f"{100 * n_over / max(len(df), 1):.0f} % of decisions", delta_color="off")
c3.metric("Crises", df["crisis"].nunique())
with st.expander("See the log"):
    st.dataframe(df[["crisis", "t", "crew_id", "recommended", "chosen", "accepted", "reason"]].head(500),
                 use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------- 2. reasons
st.subheader("2 · Why did the manager override? (free-text reasons, grouped)")
reasons = df.loc[~df["accepted"].astype(bool), "reason"].fillna("").tolist()
reasons = [r for r in reasons if r]
if reasons:
    groups, how = L.group_reasons(reasons, client)
    st.caption("Grouped by the local LLM (a sample of 40 reasons)" if how == "llm"
               else "Grouped by keyword rules (local LLM off, or its answer was incomplete)")
    rows = [{"theme": t, "reasons": len(ix), "example": reasons[ix[0]]} for t, ix in
            sorted(groups.items(), key=lambda kv: -len(kv[1]))]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Themes tell the EXPERT what the weights alone cannot: e.g. 'road risky at night' is not in the "
               "score — it may call for a new rule, decided by people, not learned silently.")
else:
    st.info("No free-text reasons in this log.")

# --------------------------------------------------------------------------- 3. learned weights
st.subheader("3 · What weights does the manager actually use?")
try:
    f = fit(df)
except ValueError as e:
    st.error(str(e))
    st.stop()
c1, c2, c3 = st.columns(3)
c1.metric("α — sensitive customer", f"{f.alpha:.1f}", f"interval {f.alpha_ci[0]:.1f} – {f.alpha_ci[1]:.1f}",
          delta_color="off")
c2.metric("β — waiting (per hour)", f"{f.beta:.2f}", f"interval {f.beta_ci[0]:.2f} – {f.beta_ci[1]:.2f}",
          delta_color="off")
agree0, agree1 = L.agreement_rate(df, 5.0, 0.1), L.agreement_rate(df, f.alpha, f.beta)
c3.metric("Agreement with the manager", f"{agree1:.0%}", f"{100 * (agree1 - agree0):+.0f} pts vs default weights")
if src == "simulated manager":
    st.caption(f"Hidden truth: α = {h_alpha}, β = {h_beta}. Default weights: α = 5, β = 0.1. "
               f"Learned from {f.n_decisions} decisions with a real choice.")

# --------------------------------------------------------------------------- 4. validation gate
st.subheader("4 · Validation gate — replay on crises 81–100 (never used for learning)")
max_inc = st.slider("Accepted increase of total customer-hours (%)", 0.0, 10.0, 3.0, 0.5,
                    help="Respecting the manager's priorities may cost a little in total; the expert sets the limit.")


@st.cache_data(show_spinner=False)
def gate(alpha: float, beta: float, max_inc: float):
    return L.validation_gate(alpha, beta, list(range(81, 101)), max_customer_hours_increase_pct=max_inc, n_jobs=2)


if st.button("Run the validation gate (≈ 20 s)") or st.session_state.get("gate_done") == (f.alpha, f.beta, max_inc):
    st.session_state["gate_done"] = (f.alpha, f.beta, max_inc)
    with st.spinner("Replaying 20 test crises twice…"):
        g = gate(f.alpha, f.beta, max_inc)
    st.dataframe(pd.DataFrame([{"": "✅" if c["ok"] else "❌", "check": c["check"], "current": c["current"],
                                "proposed": c["proposed"]} for c in g.checks]), hide_index=True,
                 use_container_width=True)
    kpis = pd.DataFrame({"current weights": g.kpi_current, "proposed weights": g.kpi_proposed}).round(2)
    st.dataframe(kpis, use_container_width=True)
    if g.passed:
        st.success("Gate PASSED — the proposal can be submitted to the expert.")
    else:
        st.error("Gate FAILED — the proposal is blocked (it can still be saved as rejected, for the record).")

    # ----------------------------------------------------------------------- 5. expert decision
    st.subheader("5 · Expert decision")
    who = st.text_input("Expert name / role", placeholder="e.g. regional crisis lead")
    note = st.text_area("Note (why approve or reject)", height=80)
    themes = {t: len(ix) for t, ix in (L.group_reasons(reasons, None)[0].items() if reasons else [])}
    b1, b2 = st.columns(2)
    if b1.button("✅ Approve and version", disabled=not (g.passed and who)):
        p = L.save_version(f, g, "approved", who, note, themes)
        st.success(f"Saved {p.name}. The Crisis room can now use these weights (sidebar → Score weights).")
    if b2.button("❌ Reject (keep current weights)", disabled=not who):
        p = L.save_version(f, g, "rejected", who, note, themes)
        st.info(f"Saved {p.name} as rejected.")

versions = L.list_versions()
if versions:
    st.subheader("Versions")
    st.dataframe(pd.DataFrame([{"version": v["version"], "status": v["status"], "date": v["date"],
                                "by": v["approved_by"], **v["score"], "note": v["note"]} for v in versions]),
                 hide_index=True, use_container_width=True)
    st.caption("Files in config/learned/ — versioned, never overwritten: every change of weights is traceable.")
