"""Small helpers shared by the Streamlit pages (plots, colours, AI status). Imported only by the app."""
from __future__ import annotations

import pandas as pd

ACTIVITY_COLOURS = {"travel": "#f2a541", "repair": "#d1495b", "break": "#2a9d8f", "standby": "#cdb4db",
                    "return": "#9db4c0", "rest": "#5c6b73", "out_of_service": "#222222"}
STRATEGY_NAMES = {"R1": "R1 · first come + nearest crew", "R2": "R2 · most customers first",
                  "WSPT": "WSPT + Hungarian (recommended)"}


def fmt_int(x: float) -> str:
    return f"{x:,.0f}".replace(",", " ")


def gantt(activities: pd.DataFrame, height: int = 900):
    import plotly.express as px

    act = activities[activities["t_end"] > activities["t_start"]].copy()
    act["start"] = pd.Timestamp("2026-01-01") + pd.to_timedelta(act["t_start"], unit="m")
    act["end"] = pd.Timestamp("2026-01-01") + pd.to_timedelta(act["t_end"], unit="m")
    fig = px.timeline(act, x_start="start", x_end="end", y="crew_id", color="type", height=height,
                      color_discrete_map=ACTIVITY_COLOURS)
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
    return fig


def customers_off_chart(results: dict, height: int = 360):
    """results: {label: SimResult}."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for label, res in results.items():
        s = res.series
        fig.add_trace(go.Scatter(x=s["t"] / 60, y=s["customers_off"], mode="lines", name=label))
    fig.update_layout(xaxis_title="Hours since the start of the crisis", yaxis_title="Customers without power",
                      height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
    return fig


def ai_status_sidebar():
    """Shows whether the local LLM is reachable + what it has cost so far. Returns the client or None."""
    import streamlit as st

    from copilot.ai import llm

    client = llm.get_client()
    with st.sidebar:
        st.divider()
        if client is None:
            st.caption("🧠 Local LLM: **off** — rules and templates are used (same numbers, plainer words). "
                       "To enable: install Ollama, then `ollama pull phi4-mini`.")
        else:
            u = client.usage()
            st.caption(f"🧠 Local LLM: **{client.model}** on this computer (no cloud). "
                       f"This session: {u['calls']} calls, {u['seconds']} s, {u['tokens']} tokens.")
    return client


def crisis_selector(default: str = "CRISIS_024", key: str = "crisis"):
    import streamlit as st

    from copilot.models import list_crises

    crises = list_crises()
    idx = crises.index(default) if default in crises else 0
    return st.selectbox("Crisis", crises, index=idx, key=key)


def require_data():
    import streamlit as st

    from copilot.models import clean_dir

    if not (clean_dir() / "incidents.parquet").exists():
        st.error("Clean data not found. Put the 4 challenge CSV files in data/raw/, then run "
                 "`python -m copilot.data.cleaning`.")
        st.stop()
