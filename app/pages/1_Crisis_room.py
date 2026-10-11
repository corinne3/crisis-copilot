"""Crisis room — compare strategies on one crisis, animated map, explained decisions, human overrides.

The tool recommends, the manager decides: every refusal is replayed (the rest of the crisis is re-planned) and
logged, with an optional reason, for the learning loop (page 6).
"""
from __future__ import annotations

import json
import logging

import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from copilot.ai import learning
from copilot.ai.llm import rephrase
from copilot.instant import crews_at, incidents_at, series_row
from copilot.kpi import compute_kpi
from copilot.policies.wspt import resolve_strategy
from copilot.simulator import Unavailability, simulate_crisis, simulation_params
from copilot.templates import decision_card, situation_briefing
from copilot.travel import MOUNTAIN_ZONE, make_travel
from copilot.ui import (
    STRATEGY_NAMES,
    ai_status_sidebar,
    crisis_selector,
    customers_off_chart,
    fmt_int,
    gantt,
    require_data,
)

logging.getLogger("copilot.simulator").setLevel(logging.ERROR)
st.set_page_config(page_title="Crisis room", layout="wide")
st.title("Crisis room")
st.caption("The tool recommends, the manager decides. Synthetic data from the utility's challenge — "
           "Hackathon IA for Impact 2026.")
require_data()

approved = learning.approved_score()
with st.sidebar:
    crisis = crisis_selector()
    chosen = st.multiselect("Strategies to compare", list(STRATEGY_NAMES), default=list(STRATEGY_NAMES),
                            format_func=STRATEGY_NAMES.get)
    st.subheader("Score weights")
    options = ["default (α=5, β=0.1)"] + ([f"learned v{approved['version']:03d} (approved)"] if approved else [])
    weights = st.radio("Weights", options, label_visibility="collapsed")
    a0, b0 = (approved["alpha_sensitive"], approved["beta_wait_per_h"]) if weights.startswith("learned") else (5.0, 0.1)
    alpha = st.slider("Weight of a sensitive customer α", 0.0, 20.0, float(a0), 0.5)
    beta = st.slider("Weight of waiting time β (per hour)", 0.0, 1.0, float(b0), 0.05)
    st.subheader("Stress tests")
    roads_cut = st.checkbox("Mountain roads cut (travel × 3)")
    n_crews = st.slider("Number of crews", 10, 63, 63)
    oos_pct = st.slider("% of crews out of service at H+3", 0, 50, 0, 10)
client = ai_status_sidebar()


@st.cache_data(show_spinner=False)
def run(crisis: str, code: str, alpha: float, beta: float, roads: bool, n_crews: int, oos: int,
        forbidden: tuple = (), forced: tuple = ()):
    from copilot.models import load_scenario

    p = simulation_params(alpha_sensitive=alpha, beta_wait_per_h=beta)
    travel = make_travel(zones=[MOUNTAIN_ZONE] if roads else ())
    scenario = {"max_crews": n_crews} if n_crews < 63 else None
    opts = {"forbidden": set(forbidden), "forced": dict(forced)}
    if oos:
        _, crews, _ = load_scenario(crisis, scenario)
        n = round(len(crews) * oos / 100)
        opts["unavailability"] = [Unavailability(c.crew_id, 180) for c in crews[:: max(1, len(crews) // n)][:n]]
    res = simulate_crisis(crisis, resolve_strategy(code)(), p, scenario=scenario, travel_min=travel, **opts)
    return res, compute_kpi(res)


if not chosen:
    st.info("Pick at least one strategy.")
    st.stop()
with st.spinner("Simulating…"):
    results = {c: run(crisis, c, alpha, beta, roads_cut, n_crews, oos_pct) for c in chosen}

# --------------------------------------------------------------------------- KPIs
ref = "R2" if "R2" in results else chosen[0]
st.subheader("Indicators")
for col, c in zip(st.columns(len(chosen)), chosen, strict=True):
    k, k_ref = results[c][1], results[ref][1]
    with col:
        st.markdown(f"**{STRATEGY_NAMES[c]}**")
        delta = None if c == ref else f"{100 * (k['customer_hours'] / k_ref['customer_hours'] - 1):+.1f} % vs {ref}"
        st.metric("Customer-hours without power", fmt_int(k["customer_hours"]), delta, delta_color="inverse")
        st.metric("Sensitive customers: 90 % reached within", f"{k['wait_p90_sensitive_h']:.1f} h")
        st.metric("Customers restored within 4h", f"{k['pct_restored_4h']:.0f} %")
        st.metric("95 % of customers restored at", f"H+{k['restored_95pct_h']:.0f}")
        st.caption(f"Saturation {k['saturation_pct']:.0f} % of the time · Gini {k['gini_towns']:.2f} · "
                   f"shift overruns: {k['shift_overruns']}")

st.subheader("Customers without power over time")
st.plotly_chart(customers_off_chart({STRATEGY_NAMES[c]: results[c][0] for c in chosen}), use_container_width=True)
st.caption("Area under each curve = customer-hours. The gap between two curves = homes back on earlier.")

detail = st.selectbox("Strategy detail", chosen, index=len(chosen) - 1, format_func=STRATEGY_NAMES.get)
res, _ = results[detail]
s = res.series
fig = go.Figure()
for col_, name in [("crews_busy", "travelling / repairing"), ("crews_resting", "return / rest / break"),
                   ("crews_available", "available"), ("crews_out", "out of service")]:
    fig.add_trace(go.Scatter(x=s["t"] / 60, y=s[col_], stackgroup="a", name=name, mode="none"))
fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="Hours", legend=dict(orientation="h"))
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------- map
st.subheader("Map of the crisis at time t")
end_h = float(s["t"].max() / 60)
t_h = st.slider("Hours since the start of the crisis", 0.0, end_h, min(6.0, end_h), 0.5)
t_min = t_h * 60
inc_t, crew_t = incidents_at(res, t_min), crews_at(res, t_min)
COL = {"upcoming": [180, 180, 180, 60], "waiting": [214, 40, 40, 220], "in repair": [242, 165, 65, 230],
       "restored": [42, 157, 143, 140]}
inc_t["colour"] = inc_t["state"].map(COL)
inc_t["radius"] = 400 + 25 * inc_t["customers"]
inc_t["outline"] = inc_t["sensitive"].map(lambda v: [128, 0, 200, 255] if v else [0, 0, 0, 0])
crew_t["colour"] = crew_t["activity"].map({"rest": [90, 90, 90, 255], "return": [120, 140, 160, 255],
                                           "out_of_service": [0, 0, 0, 255], "break": [42, 157, 143, 255],
                                           "standby": [180, 150, 210, 255]}).apply(
    lambda c: c if isinstance(c, list) else [30, 90, 200, 255])
for col_, state in zip(st.columns(4), ("waiting", "in repair", "restored", "upcoming"), strict=True):
    m = inc_t["state"].eq(state)
    col_.metric(state.capitalize(), f"{int(m.sum())} incidents", f"{int(inc_t.loc[m, 'customers'].sum())} customers",
                delta_color="off")
if st.checkbox("Plain view without base map (if the map stays grey)"):
    f3 = px.scatter(inc_t[inc_t["state"] != "upcoming"], x="lon", y="lat", color="state", size="customers",
                    size_max=14, hover_name="town", height=520,
                    color_discrete_map={"waiting": "#d62828", "in repair": "#f2a541", "restored": "#2a9d8f"})
    f3.add_scatter(x=crew_t["lon"], y=crew_t["lat"], mode="markers", name="crews",
                   marker=dict(symbol="triangle-up", size=9, color="#1e5ac8"), text=crew_t["crew_id"])
    f3.update_yaxes(scaleanchor="x", scaleratio=1.37)
    st.plotly_chart(f3, use_container_width=True)
else:
    trips = crew_t[crew_t["activity"] == "travel"]
    layers = [
        pdk.Layer("ScatterplotLayer", inc_t, get_position="[lon, lat]", get_fill_color="colour",
                  get_line_color="outline", line_width_min_pixels=2, stroked=True, get_radius="radius", pickable=True),
        pdk.Layer("LineLayer", trips, get_source_position="[lon, lat]", get_target_position="[lon_target, lat_target]",
                  get_color=[30, 90, 200, 160], get_width=2),
        pdk.Layer("ScatterplotLayer", crew_t, get_position="[lon, lat]", get_fill_color="colour", get_radius=1200,
                  radius_min_pixels=4, pickable=True),
    ]
    st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=pdk.ViewState(latitude=43.3, longitude=1.1, zoom=7.2),
                             tooltip={"text": "{town}{crew_id}\n{state}{activity}\n{customers} customers"}, height=520))
st.caption("Red = waiting · orange = in repair · green = restored · purple outline = sensitive customer · "
           "blue = crews (lines = current trips). Size = number of customers.")
briefing = situation_briefing(int(t_min), series_row(res, t_min),
                              sensitive_waiting=int((inc_t["state"].eq("waiting") & inc_t["sensitive"]).sum()),
                              hotspots=inc_t[inc_t["state"] == "waiting"]["department"].value_counts()
                              .head(2).index.tolist())
st.info(briefing)
if client is not None and st.button("✍️ Rephrase for the prefecture (local LLM, numbers checked)"):
    txt, src = rephrase(briefing, client, "the prefecture (non-technical)")
    st.success(txt if src == "llm" else "The rephrasing changed a number: template kept.\n\n" + txt)

with st.expander("Crew Gantt chart"):
    st.plotly_chart(gantt(res.activities), use_container_width=True)

# --------------------------------------------------------------------------- explained decisions + human override
st.subheader("Explained decisions — the tool recommends, the manager decides")
key = f"{crisis}|{detail}|{alpha}|{beta}|{roads_cut}|{n_crews}|{oos_pct}"
human = st.session_state.setdefault("human", {}).setdefault(key, {"forbidden": [], "forced": []})
dec = res.decisions.copy()
dec["e"] = dec["explanation"].map(json.loads)
dec["hour"] = (dec["t"] / 60).round(1)
dec["confidence"] = dec["e"].map(lambda e: e.get("confidence"))
dec["delta"] = dec["e"].map(lambda e: (e.get("alternative") or {}).get("delta_customer_hours"))
dec["debatable"] = dec["delta"].fillna(0) < 0
show = st.radio("Show", ["all decisions", "only debatable decisions"], horizontal=True)
view = dec[dec["debatable"]] if show.startswith("only") else dec
if view.empty:
    st.info("No debatable decision with these settings.")
else:
    idx = st.selectbox("Decision", view.index, format_func=lambda k: (
        f"H+{dec.at[k, 'hour']} · {dec.at[k, 'crew_id']} → {dec.at[k, 'incident_id']} · "
        f"confidence {dec.at[k, 'confidence'] or 0:.0%}" + (" · ⚠️ debatable" if dec.at[k, "debatable"] else "")))
    e, crew_sel, inc_sel, t_sel = dec.at[idx, "e"], dec.at[idx, "crew_id"], dec.at[idx, "incident_id"], dec.at[idx, "t"]
    c1, c2 = st.columns([3, 2])
    with c1:
        card = decision_card(crew_sel, inc_sel, e)
        st.info(card)
        if client is not None and st.button("✍️ Plain-language version (local LLM)"):
            txt, src = rephrase(card, client)
            st.write(txt if src == "llm" else "(the rephrasing changed a number — template kept)")
        for lim in e.get("limits", []):
            st.caption(f"Limit: {lim}")
    with c2:
        contrib = e.get("contributions_pct") or {}
        if contrib:
            f5 = px.bar(x=list(contrib.values()), y=list(contrib.keys()), orientation="h", height=220,
                        labels={"x": "contribution to the score (%)", "y": ""}, color=[v > 0 for v in contrib.values()],
                        color_discrete_map={True: "#2a9d8f", False: "#e76f51"})
            f5.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(f5, use_container_width=True)
    reason = st.text_input("Why do you refuse? (optional, free text — it feeds the learning loop)",
                           placeholder="e.g. nursing home on the other incident, the mayor keeps calling…")
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("❌ Refuse this assignment"):
            human["forbidden"].append((crew_sel, inc_sel))
            res_h, _ = run(crisis, detail, alpha, beta, roads_cut, n_crews, oos_pct,
                           tuple(human["forbidden"]), tuple(human["forced"]))
            learning.record_human_override(res_h, crisis, crew_sel, int(t_sel), inc_sel, reason)
            st.rerun()
    with b2:
        other = st.selectbox("Crew to send instead", sorted(set(res.activities["crew_id"]) - {crew_sel}))
        if st.button("✋ Force this crew"):
            human["forced"] = [(q, i) for q, i in human["forced"] if q != other] + [(other, inc_sel)]
            st.rerun()
    with b3:
        if st.button("↺ Undo my decisions"):
            human["forbidden"], human["forced"] = [], []
            st.rerun()

if human["forbidden"] or human["forced"]:
    res_h, k_h = run(crisis, detail, alpha, beta, roads_cut, n_crews, oos_pct,
                     tuple(human["forbidden"]), tuple(human["forced"]))
    k0 = results[detail][1]
    st.markdown(f"**Impact of your decisions** ({len(human['forbidden'])} refusal(s), "
                f"{len(human['forced'])} forced assignment(s))")
    m1, m2, m3 = st.columns(3)
    m1.metric("Customer-hours (recommendation)", fmt_int(k0["customer_hours"]))
    m2.metric("Customer-hours (with your decisions)", fmt_int(k_h["customer_hours"]),
              fmt_int(k_h["customer_hours"] - k0["customer_hours"]), delta_color="inverse")
    m3.metric("Sensitive: 90 % reached within", f"{k_h['wait_p90_sensitive_h']:.1f} h",
              f"{k_h['wait_p90_sensitive_h'] - k0['wait_p90_sensitive_h']:+.1f} h", delta_color="inverse")
    st.caption("Your decisions are applied in the replay; the rest of the crisis is re-planned by the strategy. "
               "Refusals are logged in outputs/decision_log.csv for the learning loop.")
