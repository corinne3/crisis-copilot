"""Benchmark over the 100 crises (results of `python -m copilot.benchmark`)."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from copilot.benchmark import summarise
from copilot.config import load_config, project_path
from copilot.ui import ai_status_sidebar

st.set_page_config(page_title="Benchmark", layout="wide")
st.title("Benchmark — every strategy on every crisis")
ai_status_sidebar()
f = project_path(load_config()["paths"]["outputs"]) / "benchmark_kpi.csv"
if not f.exists():
    st.info("Run `python -m copilot.benchmark` (≈ 1–2 min) to compare the strategies on the 100 crises.")
    st.stop()
kpi = pd.read_csv(f)
st.dataframe(summarise(kpi).T, use_container_width=True)
st.caption("Mean over the crises. gain_vs_R1_pct / gain_vs_R2_pct = mean % of customer-hours saved per crisis; "
           "crises_won = number of crises where the strategy beats the reference.")
st.plotly_chart(px.box(kpi, x="strategy", y="customer_hours", points="all", height=400,
                       labels={"customer_hours": "Customer-hours per crisis", "strategy": ""}),
                use_container_width=True)
pivot = kpi.pivot_table(index="crisis", columns="strategy", values="customer_hours")
if {"WSPT", "R2"} <= set(pivot.columns):
    pivot["gain_vs_R2_pct"] = 100 * (1 - pivot["WSPT"] / pivot["R2"])
    st.plotly_chart(px.histogram(pivot, x="gain_vs_R2_pct", nbins=30, height=320,
                                 labels={"gain_vs_R2_pct": "WSPT vs R2: customer-hours saved per crisis (%)"}),
                    use_container_width=True)
