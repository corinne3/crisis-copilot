"""Workforce realism: meal breaks, fatigue and crew rotation — people as a performance lever."""
from __future__ import annotations

import logging

import pandas as pd
import plotly.express as px
import streamlit as st

from copilot.config import load_config, project_path
from copilot.kpi import compute_kpi
from copilot.policies.wspt import resolve_strategy
from copilot.simulator import simulate_crisis
from copilot.ui import ai_status_sidebar, crisis_selector, customers_off_chart, fmt_int, gantt, require_data
from copilot.workforce import BREAK_MODES, ROTATIONS, mode_params

logging.getLogger("copilot.simulator").setLevel(logging.ERROR)
st.set_page_config(page_title="Workforce", layout="wide")
st.title("Meal breaks, fatigue and crew rotation")
st.caption("Assumption: beyond 4h of work without a break, every extra hour makes repairs longer. A break (45 min, "
           "on site, never during a repair) resets the counter. Labour law already requires a break after 6h of "
           "work. Rotation: part of the crews start later so that not everyone rests at the same time.")
require_data()

NAMES = {"no_break": "No break", "fixed_break": "Fixed break after 4h", "smart_break": "Smart break"}
with st.sidebar:
    crisis = crisis_selector()
    strategy = st.selectbox("Strategy", ["WSPT", "R2", "R1"])
    slope = st.slider("Efficiency loss per hour beyond 4h (%)", 0, 15, 5) / 100
    break_min = st.slider("Break duration (min)", 20, 60, 45, 5)
    rotation = st.selectbox("Crew rotation", list(ROTATIONS), index=1,
                            help="Without rotation, all crews start at H0 and rest at the same time.")
    chosen_break = st.selectbox("Break kept for the full model", ["smart_break", "fixed_break"], format_func=NAMES.get)
ai_status_sidebar()


@st.cache_data(show_spinner=False)
def run(crisis: str, strategy: str, mode: str, slope: float, duration: int, rotation: str = "none"):
    p = mode_params(mode, slope, rotation)
    p["breaks"] = {**p["breaks"], "duration_min": duration}
    res = simulate_crisis(crisis, resolve_strategy(strategy)(), p)
    return res, compute_kpi(res)


STEPS = [
    ("① base (no break, no fatigue)", "no_break", 0.0, "none"),
    (f"② + fatigue {slope:.0%}/h (more realistic)", "no_break", slope, "none"),
    (f"③ + {NAMES[chosen_break].lower()}", chosen_break, slope, "none"),
    (f"④ + rotation “{rotation}” (no break)", "no_break", slope, rotation),
    ("⑤ full model (break + rotation)", chosen_break, slope, rotation),
]
KPIS = {"customer_hours": ("Customer-hours", "{:,.0f}"),
        "wait_mean_sensitive_h": ("Mean wait, sensitive (h)", "{:.1f}"),
        "wait_p90_sensitive_h": ("p90 wait, sensitive (h)", "{:.1f}"),
        "wait_max_sensitive_h": ("Max wait, sensitive (h)", "{:.1f}"),
        "pct_restored_4h": ("Restored < 4h (%)", "{:.0f}"),
        "restored_95pct_h": ("95 % restored at (h)", "{:.0f}"),
        "end_h": ("End of crisis (h)", "{:.0f}"),
        "hours_without_active_crew": ("Hours with no crew working", "{:.1f}")}


def steps_table(kpis: list[dict]) -> pd.DataFrame:
    rows = []
    for (name, *_), k in zip(STEPS, kpis, strict=True):
        row = {"Step": name, **{lab: fmt.format(k[c]).replace(",", " ") for c, (lab, fmt) in KPIS.items()}}
        row["vs ①"] = f"{100 * (k['customer_hours'] / kpis[0]['customer_hours'] - 1):+.1f} %"
        rows.append(row)
    return pd.DataFrame(rows).set_index("Step")


st.subheader(f"Step by step — {crisis}, strategy {strategy}")
with st.spinner("Simulating…"):
    step_kpis = [run(crisis, strategy, m, sl, break_min, rot)[1] for _, m, sl, rot in STEPS]
st.dataframe(steps_table(step_kpis), use_container_width=True)
st.caption("Row ① is identical to the Crisis room (same rules, same strategy). “Max wait” and “End of crisis” "
           "depend on ONE incident (taken just before or just after a rest: ±11h): on a single crisis, prefer the "
           "mean wait, the p90 wait and “95 % restored”.")


@st.cache_data(show_spinner=False)
def mean_steps(strategy: str, slope: float, duration: int, rotation: str, chosen_break: str, n: int) -> list[dict]:
    from copilot.benchmark import run as bench

    cls = resolve_strategy(strategy)
    out = []
    for _, m, sl, rot in STEPS:
        p = mode_params(m, sl, rot)
        p["breaks"] = {**p["breaks"], "duration_min": duration}
        kpi, _ = bench(list(range(1, n + 1)), {strategy: cls}, params=p, n_jobs=2)
        out.append(kpi.mean(numeric_only=True).to_dict())
    return out


if st.button("Compute the same table as a mean over 20 crises (≈ 1 min)"):
    with st.spinner("20 crises…"):
        st.dataframe(steps_table(mean_steps(strategy, slope, break_min, rotation, chosen_break, 20)),
                     use_container_width=True)
    st.caption("Mean over crises 1 to 20: these are the numbers to quote.")

st.subheader(f"The 3 break modes — fatigue {slope:.0%}/h, rotation “{rotation}”")
modes = {m: run(crisis, strategy, m, slope, break_min, rotation) for m in BREAK_MODES}
base = run(crisis, strategy, "no_break", 0.0, break_min, "none")[1]
ref = modes["no_break"][1]["customer_hours"]
for col, m in zip(st.columns(3), BREAK_MODES, strict=True):
    k = modes[m][1]
    with col:
        st.markdown(f"**{NAMES[m]}**")
        st.metric("Customer-hours", fmt_int(k["customer_hours"]),
                  None if m == "no_break" else f"{100 * (k['customer_hours'] / ref - 1):+.1f} %", delta_color="inverse")
        st.metric("Breaks taken", k["n_breaks"])
        st.metric("Mean repair efficiency", f"{100 / k['mean_fatigue']:.0f} %", help="100 % = no fatigue")
        st.metric("Hours with no crew working", f"{k['hours_without_active_crew']:.1f} h",
                  f"{k['hours_without_active_crew'] - base['hours_without_active_crew']:+.1f} h vs ①",
                  delta_color="inverse")
        st.caption(f"Mean sensitive wait {k['wait_mean_sensitive_h']:.1f} h · 95 % restored at "
                   f"{k['restored_95pct_h']:.0f} h · shift overruns: {k['shift_overruns']}")
st.plotly_chart(customers_off_chart({NAMES[m]: modes[m][0] for m in BREAK_MODES}), use_container_width=True)
g = st.selectbox("Crew Gantt chart", list(BREAK_MODES), index=2, format_func=NAMES.get)
st.plotly_chart(gantt(modes[g][0].activities), use_container_width=True)

out = project_path(load_config()["paths"]["outputs"])
st.subheader("Sensitivity: from which fatigue level does the break pay off?")
f = out / "break_sensitivity.csv"
if f.exists():
    df = pd.read_csv(f)
    mean = df.groupby(["fatigue_slope", "mode"])["customer_hours"].mean().unstack()
    gain = (100 * (1 - mean.div(mean["no_break"], axis=0))).drop(columns="no_break").reset_index()
    gain = gain.melt(id_vars="fatigue_slope", var_name="mode", value_name="gain_pct")
    gain["mode"] = gain["mode"].map(NAMES)
    fig = px.line(gain, x="fatigue_slope", y="gain_pct", color="mode", markers=True, height=360,
                  labels={"fatigue_slope": "Efficiency loss per hour beyond 4h",
                          "gain_pct": "Customer-hours saved vs no break (%)"})
    fig.add_hline(y=0, line_dash="dot")
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Mean over {df['crisis'].nunique()} crises. Above 0: the break saves customers time; below: it costs.")
else:
    st.info("Run `python -m copilot.workforce` to compute the sensitivity curve.")

st.subheader("Crew rotation: organisations compared")
f = out / "rotation_comparison.csv"
if f.exists():
    rot = pd.read_csv(f)
    cols = {"customer_hours": "Customer-hours", "wait_max_sensitive_h": "Max wait, sensitive (h)",
            "hours_without_active_crew": "Hours with no crew working", "end_h": "End of crisis (h)"}
    st.dataframe(rot.groupby("rotation", sort=False)[list(cols)].mean().round(1).rename(columns=cols),
                 use_container_width=True)
    st.caption(f"Mean over {rot['crisis'].nunique()} crises, fixed break, fatigue 5 %/h. More groups = better "
               "night coverage but fewer crews at the initial peak: the manager's trade-off.")
else:
    st.info("Run `python -m copilot.workforce` to compare rotation organisations.")
