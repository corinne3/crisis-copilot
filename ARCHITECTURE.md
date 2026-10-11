# Architecture

```
            ┌────────────── ① PREDICT ──────────────┐
 first      │ forecast.py  gradient boosting (q5/50/95)│──► recommendations (rules): second shift,
 reports ──►│ features of the arrival curve at H+τ    │    reinforcement, department under pressure
            └──────────────────────────────────────────┘
 crew       ┌────────────── ② UNDERSTAND ────────────┐
 messages ─►│ field_reports.py  LLM → JSON | rules     │──► validate(): known ids, plausible minutes,
            └──────────────────────────────────────────┘    numbers from the message, safety → human
                                │ ticked by the manager
                                ▼
            ┌────────────── DECIDE (no AI) ───────────┐
 incidents ►│ simulator.py  discrete events            │   policies/wspt.py
 crews      │  shift 12h / rest 11h / breaks / fatigue │◄──score = N(1+α·s)(1+β·wait)/(travel+duration)
            │  rotation / unavailability / overrides   │   Hungarian assignment (scipy)
            └──────────────────────────────────────────┘
                 │ every decision                │ KPIs (kpi.py)
                 ▼                               ▼
            explanation.py ──► templates.py ──► (llm.rephrase, number guard)
                 │
                 ▼
            HUMAN: accept / refuse (+ reason) / force  ──► replay + outputs/decision_log.csv
                 │
                 ▼
            ┌────────────── ④ LEARN ─────────────────┐
            │ learning.py  conditional logit → α, β   │──► validation gate (crises 81–100, hard safety
            │ reasons grouped (LLM | keywords)        │    checks) ──► expert approves ──► config/learned/
            └─────────────────────────────────────────┘                                  scoring_vNNN.yaml
            ③ SIMULATE: whatif.py  question → (LLM | rules) → scenario JSON (checked, editable) → simulator
```

## Design rules

1. **The decision core is deterministic.** Same inputs → same recommendation, computed in milliseconds,
   explainable term by term. No model can override a safety rule (12h shift, 11h rest, a crew never leaves a
   repair unfinished).
2. **AI only where text or uncertainty is involved**, and each AI brick has a no-AI fallback *and* a measured
   comparison with it. If the AI does not beat the fallback, it is not worth its energy.
3. **The LLM never produces a number.** Numbers come from the simulator, the KPIs or the model; the LLM may
   translate (question → parameters, message → JSON) or rephrase, and its output is checked.
4. **Nothing is learned silently.** The learning loop proposes two readable weights with intervals, replays them
   on unseen crises, and an expert approves; versions are files, never overwritten.
5. **Small and local.** One ~3B-parameter model on CPU via Ollama, scikit-learn and scipy for the rest. No GPU,
   no cloud, no paid API.

## Simulator contract

- Time in minutes since the crisis start; one decision round per event instant (T1 events → T3 decide → T5 end).
- A strategy implements `Policy.decide(state, travel_min) -> list[Assignment]` and must use
  `simulator.is_feasible()` (travel + nominal duration × fatigue ≤ remaining shift, not forbidden).
- Output `SimResult`: `incidents`, `activities` (travel, repair, return, rest, break, standby, out_of_service),
  `decisions` (with the explanation JSON), `series` (snapshots every 30 min), `metadata`.

## Train / test discipline

Crises 1–80 are used for anything that is tuned or learned (score weights, forecast model, learning loop);
crises 81–100 are only used to evaluate (`kpi.train_test_split`).

## From the hackathon to this repository

- **V1** (hackathon, team repo): simulator, R1/R2/WSPT, explanations, crisis room.
- **V2**: meal breaks, fatigue, crew rotation (`workforce.py`, Workforce page).
- **V3** (this repo): clean English rewrite of V1+V2 and the four frugal-AI bricks. With the same inputs,
  the V3 core reproduces the V1 numbers exactly (e.g. crisis 24, WSPT: 199 004.6 customer-hours — tested).
