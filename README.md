# ⚡ Crisis Copilot — dispatching power-grid repair crews after a storm

**Frugal AI around a deterministic optimisation core.** The tool recommends, the crisis manager decides.

Built after the **Hackathon IA for Impact 2026** (La Mêlée, Toulouse, 9–10 Oct 2026) on the **Enedis** challenge:
after a storm, hundreds of incidents cut power across Ariège, Haute-Garonne and Gers; **63 repair crews from 18
sites** must be sent one incident at a time, under strict safety rules (12h shifts, 11h rest). Goal: minimise
**Σ customers × hours without power**, sensitive customers first, every decision explainable.

> Data: synthetic files provided by Enedis for the hackathon (100 simulated crises, 15 676 incidents).
> They are **not** in this repository.

---

## Why "frugal AI"

AI is used **only where rules or optimisation cannot do the job**, with small models that run on a laptop CPU,
no GPU, no cloud, no paid API. The decision core has **no AI at all**: it is fast, exact, auditable — and the
whole app keeps working if the language model is switched off.

| Step | Question | Method | AI? |
|---|---|---|---|
| ① **Predict** | How big will this crisis get? | gradient boosting on the first reports (scikit-learn) | small ML |
| ② **Understand** | What are the crews writing? | local LLM → JSON + strict checks, keyword fallback | small LLM |
| **Decide** | Which crew goes where, now? | WSPT score + Hungarian assignment + shift rules | **none** |
| **Explain** | Why this choice? | factor contributions, best alternative, exact cost (Smith exchange) | templates |
| ③ **Simulate** | What if we lose 5 crews in Gers? | question → parameters → discrete-event simulator | LLM translates only |
| **Human** | Accept / refuse / force | every decision replayed and logged | — |
| ④ **Learn** | What does the manager really prioritise? | conditional logit on overrides + validation gate + expert approval | small stats |

One small local model (**Microsoft Phi-4-mini, 3.8B parameters, MIT licence, via Ollama**, ~2.5 GB) is shared by
every AI brick. Any other Ollama model can be set in `config/default.yaml` (`llm.model`). Its outputs are always
checked: JSON schema, known ids, plausible ranges, and a **number guard** — a sentence produced by the LLM is
rejected if it contains a number that is not in the source. Every call is metered (time, tokens) in the app.

## Results (100 crises, nominal simulation)

| Strategy | Customer-hours (mean) | Sensitive customers: p90 wait | Restored < 4h |
|---|---:|---:|---:|
| R1 — first come, nearest crew | 154 428 | 13.5 h | 14.3 % |
| R2 — most customers first | 130 653 | 14.0 h | 17.0 % |
| **WSPT + Hungarian** | **128 672** | **6.6 h** | **20.8 %** |

WSPT saves 11 % of customer-hours vs R1, matches R2 on the total and **halves the wait of sensitive customers**,
with **0 shift-rule violations**.

**Workforce realism** (mean of crises 1–20, WSPT): with a realistic fatigue assumption (+5 %/h after 4h), smart
meal breaks + a 2-group rotation bring customer-hours back near the no-fatigue level and cut the p90 wait of
sensitive customers from **8.4 h to 2.8 h** — because without rotation all crews reach 12h and rest *at the same
time* (11 h with nobody on the field).

**AI bricks — each one measured against a no-AI baseline:**
- ① Forecast at H+1h, test crises 81–100: error −15 % to −40 % vs naive extrapolation (total incidents, total
  work, end of restoration), intervals with measured coverage.
- ② Field reports, 12 real-life phrasings: messages fully understood **25 % with keyword rules → 83 % with
  phi4-mini** (message type 33 % → 100 %), ≈ 9 s per message on a laptop CPU. The remaining errors are caught
  by the checks (e.g. an incident id not written in the message → "needs review").
- ③ What-if: 5/5 example questions translated correctly once cross-checked with the rules (the LLM alone
  inverted "10 % slower" once: the cross-check caught it).
- ④ Learning loop: a simulated manager with **hidden** weights (α = 10, β = 0.5) — the loop recovers
  α ≈ 10.3, β ≈ 0.49 from decisions alone; agreement with the manager 65 % → 72 %; on the 20 test crises the
  proposal cuts the p90 wait of sensitive customers 6.5 h → 4.5 h for +1.5 % customer-hours, passes the
  safety gate, and still waits for an expert's approval.

## Quick start (Windows PowerShell — same on macOS/Linux with `source .venv/bin/activate`)

```powershell
git clone https://github.com/<you>/crisis-copilot.git
cd crisis-copilot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# put the 4 challenge files in data\raw\ : agents_v2.csv, poste_v2.csv, insee_v2.csv, coupures_v2_100_crises.csv
python -m copilot.data.cleaning          # → data/clean/*.parquet
python -m copilot.benchmark              # 100 crises × 3 strategies (≈ 1–2 min)
python -m copilot.workforce --crises 1-20
python -m copilot.ai.forecast            # train ① (≈ 30 s)
python -m copilot.ai.learning            # ④ end to end, in the terminal
streamlit run app/Home.py
```

Optional — the local language model (the app works without it):
```powershell
# install Ollama from https://ollama.com, then:
ollama pull phi4-mini
python -m copilot.ai.check_llm           # tests every AI brick with the model (≈ 1–3 min on CPU)
```

Measured with `python -m copilot.ai.check_llm` (phi4-mini, CPU only, no GPU): 26 calls, ~4.5 min in total,
first call ~14 s (model loading), then ~1.5 s for a short answer and ~9 s for a JSON extraction.

Tests: `pytest -q` · lint: `ruff check .`

## Project structure

```
src/copilot/
  data/cleaning.py     raw CSV → clean parquet, with checks and a report
  models.py            domain objects, scenario loading
  travel.py            travel times (detour factor, plain/mountain speed, slow zones)
  simulator.py         discrete-event simulator: shifts, rest, breaks, fatigue, rotation, Monte Carlo, overrides
  policies/            R1, R2 baselines and WSPT + Hungarian (the recommended strategy)
  explanation.py       why this decision: contributions, alternative, exact cost, confidence, limits
  templates.py         decision card and situation briefing (no LLM, no invented number)
  kpi.py · instant.py · benchmark.py · workforce.py
  ai/llm.py            shared local LLM client: JSON mode, timeouts, number guard, usage meter, fallback
  ai/forecast.py       ① early forecast
  ai/field_reports.py  ② field reports
  ai/whatif.py         ③ what-if
  ai/learning.py       ④ learning loop
app/                   Streamlit app (Home + 7 pages)
config/default.yaml    every parameter in one place; config/learned/ = versioned, approved score weights
```
More in [ARCHITECTURE.md](ARCHITECTURE.md).

## What real data would change

The mechanisms are in place; the numbers would become operational with the utility's real data:
- **Forecast before the first incident**: weather forecasts (gusts, lightning, snow) crossed with network
  vulnerability (overhead lines, vegetation) and the history of past storms — forecasting at D-1, not at H+1.
- **Real repair durations** by cause and equipment instead of 2h/4h/8h; **real road network** and closures.
- **Network topology**: remote switching restores many customers before any crew arrives.
- **Crew skills and equipment** (cranes, generators), real shift plans.
- **Real field messages and real manager decisions** for bricks ② and ④.
- Critical infrastructure + personal data → small **local** models are not only frugal, they keep data in-house.

## Limits and assumptions (read before quoting numbers)

- Synthetic data; straight-line travel × 1.3; nominal durations; one crew per incident; no handover between crews.
- Fatigue slope and break effects are **assumptions** (sensitivity curves in the Workforce page).
- Under Monte Carlo uncertainty, shift overruns can appear (the rule uses nominal durations): a safety margin
  would be the next step.
- The learning loop is demonstrated with a **simulated** manager; with real decisions, the same code runs unchanged.

## Credits

Hackathon team (Enedis challenge, Hackathon IA for Impact 2026): Corinne Berges and teammates
<!-- TODO: add your teammates' names here -->.
This repository is a clean, English rewrite (V3) of my part of the hackathon work (V1), extended with workforce
realism (V2: breaks, fatigue, rotation) and the four frugal-AI bricks (V3). Thanks to Enedis for the challenge and
the data, and to La Mêlée for the event.

Licence: MIT.
