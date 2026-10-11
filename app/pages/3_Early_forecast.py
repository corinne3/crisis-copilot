"""Brick ① — Early forecast: at H+1, how big will this crisis get, and what should we prepare now?"""
from __future__ import annotations

import logging

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from copilot.ai import forecast as F
from copilot.models import clean_dir, read_tables
from copilot.ui import ai_status_sidebar, crisis_selector, require_data

logging.getLogger("copilot.simulator").setLevel(logging.ERROR)
st.set_page_config(page_title="Early forecast", layout="wide")
st.title("① Early forecast — how big will this crisis get?")
st.caption("A small gradient-boosting model (scikit-learn, CPU, trained in seconds) reads the first reports and "
           "predicts the final size of the crisis with an interval. Trained on crises 1–80, evaluated on 81–100 "
           "(never seen). Simple, readable rules turn the forecast into early recommendations.")
require_data()

path = F.model_path()
if not path.exists():
    st.warning("No forecast model yet.")
    if st.button("Train the model now (≈ 30 s: 80 crises simulated once, then training)"):
        with st.spinner("Training…"):
            F.train_default(n_jobs=2)
        st.rerun()
    st.stop()


@st.cache_resource
def load_model(mtime: float):
    return F.ForecastModel.load(path)


model = load_model(path.stat().st_mtime)
sites, incidents = read_tables(str(clean_dir()))
crews_per_dep = sites.groupby("department")["n_crews"].sum().astype(int).to_dict()

with st.sidebar:
    crisis = crisis_selector("CRISIS_090", key="fc_crisis")
    tau = st.select_slider("We are at H+… (minutes since the first report)", options=list(F.TAUS), value=60)
    reveal = st.checkbox("Reveal what really happened", value=False)
    if st.button("Retrain the model"):
        with st.spinner("Training…"):
            F.train_default(n_jobs=2)
        st.cache_resource.clear()
        st.rerun()
ai_status_sidebar()

if int(crisis.split("_")[1]) <= 80:
    st.warning("This crisis was used to TRAIN the model: pick 81–100 for an honest test.")

inc = incidents[incidents["crisis_id"] == crisis].sort_values("t_reported")
feat = F.snapshot_features(inc, tau)
pred = model.predict(feat).iloc[0].to_dict()
base = model.predict_baseline(pd.DataFrame([feat])).iloc[0].to_dict()
seen = inc[inc["t_reported"] <= tau]

c1, c2, c3 = st.columns(3)
labels = {"total_incidents": ("Total incidents", "{:.0f}"), "total_work_h": ("Repair work (crew-hours)", "{:.0f}"),
          "end_h": ("End of restoration (H+)", "{:.0f}")}
truth = {"total_incidents": len(inc), "total_work_h": inc["duration_min"].sum() / 60}
for col, (t, (lab, fmt)) in zip((c1, c2, c3), labels.items(), strict=True):
    if f"{t}_p50" not in pred:
        continue
    col.metric(lab, fmt.format(pred[f"{t}_p50"]),
               f"interval {fmt.format(pred[f'{t}_low'])} – {fmt.format(pred[f'{t}_high'])}", delta_color="off")
    col.caption(f"naive extrapolation: {fmt.format(base[t])}"
                + (f" · **actual: {fmt.format(truth[t])}**" if reveal and t in truth else ""))
st.caption(f"Seen so far at H+{tau} min: {len(seen)} incidents, {int(seen['customers'].sum())} customers, "
           f"{seen['duration_min'].sum() / 60:.0f} crew-hours of repairs.")

fig = go.Figure()
cum = inc.assign(n=range(1, len(inc) + 1))
known = cum[cum["t_reported"] <= tau]
fig.add_trace(go.Scatter(x=known["t_reported"] / 60, y=known["n"], mode="lines", name="reported so far",
                         line=dict(width=3)))
if reveal:
    fig.add_trace(go.Scatter(x=cum["t_reported"] / 60, y=cum["n"], mode="lines", name="what happened next",
                             line=dict(dash="dot")))
x_end = max(8.0, cum["t_reported"].max() / 60)
fig.add_trace(go.Scatter(x=[tau / 60, x_end, x_end, tau / 60], fill="toself", mode="none", name="forecast interval",
                         y=[len(known), pred["total_incidents_high"], pred["total_incidents_low"], len(known)],
                         fillcolor="rgba(42,157,143,0.2)"))
fig.add_hline(y=pred["total_incidents_p50"], line_dash="dash", annotation_text="forecast (median)")
fig.add_vline(x=tau / 60, line_dash="dot")
fig.update_layout(height=360, xaxis_title="Hours since the first report", yaxis_title="Cumulative incidents",
                  margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

st.subheader("What to prepare now")
icons = {"info": "ℹ️", "warning": "⚠️", "action": "🟢"}
for r in F.recommendations(pred, seen, crews_per_dep):
    (st.success if r.level == "action" else st.warning if r.level == "warning" else st.info)(
        f"{icons[r.level]} **{r.title}** — {r.why}")
st.caption("Rules, not AI: every number in these sentences comes from the forecast above or from the crew table. "
           "The manager decides.")

with st.expander("Is the model worth it? Test crises 81–100, model vs naive extrapolation"):
    if model.metrics is not None:
        m = model.metrics.rename(columns={"tau_min": "H+ (min)", "mae_model": "error model (MAE)",
                                          "mae_baseline": "error naive (MAE)", "gain_pct": "gain %",
                                          "coverage_pct": "interval coverage %"})
        st.dataframe(m, use_container_width=True, hide_index=True)
        st.caption("Frugality rule: the model is kept only because it beats the naive extrapolation on crises it "
                   "never saw. Only 20 test crises: the coverage figures are rough.")
st.info("With real utility data, the strongest early signal would come BEFORE the first report: weather forecasts "
        "(wind gusts, lightning, snow) crossed with the vulnerability of the network (overhead lines, trees) and "
        "past storms. This page shows the mechanism with the arrival curve only.")
