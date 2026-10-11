"""Crisis copilot — home page.   Run:  streamlit run app/Home.py"""
from __future__ import annotations

import streamlit as st

from copilot.models import clean_dir
from copilot.ui import ai_status_sidebar

st.set_page_config(page_title="Crisis copilot", layout="wide")
st.title("⚡ Crisis copilot — dispatching repair crews after a storm")
st.markdown("""
After a storm, hundreds of incidents cut power to thousands of homes. **63 repair crews** from 18 sites must be
sent, one incident at a time, under strict safety rules (12h shifts, 11h rest). This copilot **recommends**;
the crisis manager **decides**.

**Frugal AI — AI only where it truly helps, small models, on a laptop CPU, no cloud:**

| Step | What | How | AI? |
|---|---|---|---|
| ① **Predict** | How big will this crisis get? | gradient boosting on the first reports | small ML |
| ② **Understand** | What are the crews writing? | local LLM → JSON, strict checks, rule fallback | small LLM |
| **Decide** | Which crew goes where, now? | WSPT score + Hungarian assignment, safety rules | **none** — deterministic |
| **Explain** | Why this choice? | factor contributions, best alternative, exact cost | templates (LLM may rephrase) |
| ③ **Simulate** | What if…? | question → parameters → simulator | LLM translates only |
| **Human** | Accept / refuse / force | every decision replayed and logged | — |
| ④ **Learn** | What does the manager really prioritise? | conditional logit on overrides + expert gate | small stats |

The decision core never depends on a language model: if the LLM is off, every page still works, with plainer words.
""")
st.info("Pages (left menu): Crisis room · Workforce (breaks, fatigue, rotation) · ① Early forecast · "
        "② Field reports · ③ What-if · ④ Learning loop · Benchmark.")
if not (clean_dir() / "incidents.parquet").exists():
    st.error("Data not found: put the 4 challenge CSV files in data/raw/ and run `python -m copilot.data.cleaning`.")
ai_status_sidebar()
st.caption("Built after the Hackathon IA for Impact 2026 (La Mêlée, Toulouse) on the utility's synthetic data. "
           "Open source, MIT licence.")
