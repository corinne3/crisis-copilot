"""Brick ③ — What-if in plain language. The LLM translates, the simulator computes, the manager checks."""
from __future__ import annotations

import json
import logging

import pandas as pd
import streamlit as st

from copilot.ai import whatif as W
from copilot.ai.llm import rephrase
from copilot.ui import ai_status_sidebar, crisis_selector, customers_off_chart, require_data

logging.getLogger("copilot.simulator").setLevel(logging.ERROR)
st.set_page_config(page_title="What-if", layout="wide")
st.title("③ What-if — ask in plain language")
st.caption("Your question is translated into simulator parameters (local LLM, or rules without it). You SEE the "
           "parsed scenario and can correct it. Then the simulator replays the crisis: every number of the answer "
           "comes from the simulation, never from the language model.")
require_data()
with st.sidebar:
    crisis = crisis_selector(key="wi_crisis")
client = ai_status_sidebar()

EXAMPLES = ["What if we lose 5 crews in Gers from H+6 to H+18?",
            "What if we get 10 extra crews in Ariège from H+4?",
            "Snow on the passes: mountain roads 3 times slower",
            "What if repairs take 20% longer and travel is 10% slower?",
            "Rotation with 2 groups 6h apart and smart breaks?"]
ex = st.selectbox("Examples", ["(type your own)"] + EXAMPLES)
question = st.text_input("Your question", value="" if ex.startswith("(") else ex,
                         placeholder="e.g. what if we lose 3 crews on site 4 for 8 hours?")
if not question:
    st.stop()

sc = W.understand(question, client)
crisis_run = f"CRISIS_{sc.crisis:03d}" if sc.crisis else crisis
st.markdown(f"**Understood ({'local LLM' if sc.source == 'llm' else 'rules'})** — crisis {crisis_run}:")
for line in sc.describe() or ["(nothing)"]:
    st.markdown(f"- {line}")
for issue in sc.issues:
    st.warning(issue)

with st.expander("Check / correct the scenario (JSON)"):
    raw = st.text_area("changes", json.dumps(sc.changes, indent=1, ensure_ascii=False), height=180)
    try:
        sc = W.validate(W.Scenario(sc.crisis, json.loads(raw), [], "edited" if raw != json.dumps(
            sc.changes, indent=1, ensure_ascii=False) else sc.source), question + " " + raw)
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {e}")
        st.stop()
if not sc.changes:
    st.stop()


@st.cache_data(show_spinner=False)
def compute(crisis: str, changes_json: str):
    return W.run_whatif(crisis, W.Scenario(changes=json.loads(changes_json)))


with st.spinner("Simulating base and what-if…"):
    r = compute(crisis_run, json.dumps(sc.changes, sort_keys=True))
answer = W.answer_text(sc, r["base"], r["whatif"])
if client is not None:
    answer, _src = rephrase(answer, client, "a crisis manager in a hurry")
st.success(answer)
st.dataframe(pd.DataFrame(r["table"]).set_index("kpi"), use_container_width=True)
st.plotly_chart(customers_off_chart({"base": r["base_result"], "what-if": r["result"]}), use_container_width=True)
st.caption("Same strategy (WSPT), same crisis, only the scenario changes. Assumptions: crews that come back "
           "from unavailability start a fresh 12h shift; reinforcement crews start at the biggest sites of the "
           "department.")
