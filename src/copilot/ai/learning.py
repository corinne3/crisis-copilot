"""Brick ④ — Learn from overrides: the copilot learns the manager's priorities, an expert validates.

Loop:  decision log  →  learn the weights the manager actually uses  →  group the free-text reasons (LLM)
       →  replay the proposal on crises never used for learning (validation gate)  →  expert approves
       →  versioned scoring file (config/learned/scoring_vNNN.yaml)  →  used by the WSPT strategy.

What is learned is deliberately SMALL and READABLE: the two weights of the score
        score = customers × (1 + α·sensitive) × (1 + β·wait_h) / (travel + duration)
i.e. "how much a sensitive customer counts" (α) and "how much waiting counts" (β). No black box: the expert sees
two numbers, their confidence interval and their effect on every KPI before saying yes.

Statistical model: conditional logit (McFadden). At each decision the manager picks one incident among the
feasible candidates with probability ∝ exp(λ · log score_αβ). α, β, λ are fitted by maximum likelihood
(scipy, CPU, < 1 s); bootstrap gives the confidence intervals.

No real manager data exists yet, so a SimulatedManager with HIDDEN preferences plays the human. The test is
honest: can the loop recover the hidden α, β from decisions + noisy free-text reasons alone? With real
decisions (the crisis room logs every refusal / forced assignment) the same code runs unchanged.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from copilot.models import Assignment, Crew, Incident, SimState
from copilot.policies.base import Policy
from copilot.policies.wspt import WSPT
from copilot.travel import TravelMin

log = logging.getLogger("copilot.learning")

FEATURES = ["customers", "sensitive", "wait_min", "travel_min", "duration_min"]
LOG_COLUMNS = ["source", "crisis", "t", "crew_id", "recommended", "chosen", "accepted", "reason", "candidates"]


# --------------------------------------------------------------------------- features
def candidate_features(crew: Crew, incidents: list[Incident], t: int, params: dict,
                       travel_min: TravelMin) -> list[dict]:
    """Feasible candidates for one crew at time t, with the 5 features the score uses."""
    from copilot.simulator import is_feasible

    out = []
    for i in incidents:
        tr = travel_min(crew.lat, crew.lon, i.lat, i.lon)
        if is_feasible(crew, i, tr, t, params):
            out.append({"incident_id": i.incident_id, "customers": int(i.customers), "sensitive": bool(i.sensitive),
                        "wait_min": int(t - i.t_reported), "travel_min": round(float(tr), 1),
                        "duration_min": int(i.nominal_duration)})
    return out


def log_scores(c: pd.DataFrame | dict, alpha: float, beta: float) -> np.ndarray:
    """log(score) for candidates (vectorised)."""
    n = np.asarray(c["customers"], dtype=float)
    s = np.asarray(c["sensitive"], dtype=float)
    w = np.maximum(np.asarray(c["wait_min"], dtype=float), 0) / 60
    t = np.asarray(c["travel_min"], dtype=float) + np.asarray(c["duration_min"], dtype=float)
    return np.log(np.maximum(n, 1)) + np.log1p(alpha * s) + np.log1p(beta * w) - np.log(np.maximum(t, 1))


# --------------------------------------------------------------------------- simulated manager
REASONS = {
    "sensitive": ["There is a nursing home on {b}, it goes first.", "Hospital/clinic on {b} — medical priority.",
                  "{b} has a sensitive customer (home oxygen), can't wait.", "sensitive cust on {b}!! go there first",
                  "School + elderly residents on {b}, I prefer this one."],
    "wait": ["{b} has been waiting for ages, the mayor keeps calling.", "Fairness: {b} waits since this morning.",
             "Customers on {b} were forgotten for hours, not acceptable.", "too long wait on {b}, complaints rising",
             "Can't leave {b} in the dark any longer."],
    "distance": ["{b} is right next to the crew, quicker.", "Road to {a} is risky at night, {b} is closer.",
                 "Crew is already in the area of {b}.", "{a} too far for this crew"],
    "other": ["Field knowledge: {b} is easier to fix than it looks.", "Gut feeling, local experience.",
              "Called the field supervisor, he prefers {b}."],
}


@dataclass
class SimulatedManager:
    """Plays the human: hidden weights + decision noise + a short free-text reason when overriding."""
    alpha: float = 10.0          # cares MORE about sensitive customers than the default 5
    beta: float = 0.5            # cares MUCH more about long waits than the default 0.1 (fairness, mayors)
    temperature: float = 0.08    # decision noise (0 = always the best by its own weights)
    seed: int = 0
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def choose(self, cands: list[dict]) -> int:
        u = log_scores(pd.DataFrame(cands), self.alpha, self.beta)
        if self.temperature > 0:
            u = u / self.temperature + self.rng.gumbel(size=len(u))
        return int(np.argmax(u))

    def reason(self, rec: dict, chosen: dict) -> str:
        """Theme = the factor that pushed the manager's choice most, by its own (hidden) weights."""
        gain = {
            "sensitive": math.log1p(self.alpha * chosen["sensitive"]) - math.log1p(self.alpha * rec["sensitive"]),
            "wait": (math.log1p(self.beta * max(chosen["wait_min"], 0) / 60)
                     - math.log1p(self.beta * max(rec["wait_min"], 0) / 60)),
            "distance": (math.log(rec["travel_min"] + rec["duration_min"])
                         - math.log(chosen["travel_min"] + chosen["duration_min"])),
        }
        theme = max(gain, key=gain.get)
        if gain[theme] < 0.05:
            theme = "other"
        txt = self.rng.choice(REASONS[theme])
        return txt.format(a=rec["incident_id"], b=chosen["incident_id"])


class ManagerInTheLoop(Policy):
    """WSPT recommends, the (simulated) manager accepts or overrides; every decision is logged."""
    name = "WSPT+manager"

    def __init__(self, manager: SimulatedManager, recommender: Policy | None = None, crisis: str = "",
                 records: list | None = None):
        self.manager, self.recommender, self.crisis = manager, recommender or WSPT(), crisis
        self.records = records if records is not None else []

    def decide(self, state: SimState, travel_min: TravelMin) -> list[Assignment]:
        recs = self.recommender.decide(state, travel_min)
        crews = {c.crew_id: c for c in state.available_crews}
        taken: set[str] = set()
        out = []
        for a in recs:
            free_incs = [i for i in state.waiting_incidents if i.incident_id not in taken]
            cands = candidate_features(crews[a.crew_id], free_incs, state.t, state.params, travel_min)
            ids = [c["incident_id"] for c in cands]
            if a.incident_id not in ids:
                continue                                    # recommendation clashes with an earlier override
            k_rec = ids.index(a.incident_id)
            k = self.manager.choose(cands) if len(cands) > 1 else k_rec
            accepted = k == k_rec
            self.records.append({
                "source": "simulated", "crisis": self.crisis, "t": state.t, "crew_id": a.crew_id,
                "recommended": a.incident_id, "chosen": ids[k], "accepted": accepted,
                "reason": "" if accepted else self.manager.reason(cands[k_rec], cands[k]),
                "candidates": json.dumps(cands)})
            taken.add(ids[k])
            out.append(Assignment(a.crew_id, ids[k], {"rule": self.name, "override": not accepted}))
        return out


def collect_log(crises: list, manager: SimulatedManager | None = None, params: dict | None = None,
                max_per_crisis: int | None = None) -> pd.DataFrame:
    """Run the crises with the manager in the loop and return the decision log (one row per decision)."""
    from copilot.models import normalise_crisis_id
    from copilot.simulator import simulate_crisis, simulation_params

    manager = manager or SimulatedManager()
    p = {**(params or simulation_params()), "explain": False}
    rows: list[dict] = []
    for c in crises:
        recs: list[dict] = []
        simulate_crisis(c, ManagerInTheLoop(manager, crisis=normalise_crisis_id(c), records=recs), p)
        rows.extend(recs[:max_per_crisis] if max_per_crisis else recs)
    return pd.DataFrame(rows, columns=LOG_COLUMNS)


def decision_log_path() -> Path:
    from copilot.config import load_config, project_path

    return project_path(load_config()["paths"]["outputs"], "decision_log.csv")


def record_human_override(res_after, crisis: str, crew_id: str, t: int, recommended: str, reason: str = "",
                          travel_min: TravelMin | None = None, path: Path | None = None) -> dict | None:
    """Log a REAL override made in the crisis room, read from the re-simulation that applied it:
    candidates = incidents waiting at t (approximation: shift feasibility not re-checked), chosen = the incident
    the crew was actually sent to at t after the override. Appends to outputs/decision_log.csv."""
    from copilot.travel import make_travel

    travel_min = travel_min or make_travel()
    act = res_after.activities
    trip = act[(act["crew_id"] == crew_id) & (act["type"] == "travel") & (act["t_start"] >= t)].sort_values("t_start")
    if trip.empty:
        return None
    t = int(trip.iloc[0]["t_start"])                    # the crew's next decision after the refusal
    lat, lon, chosen = float(trip.iloc[0]["lat_from"]), float(trip.iloc[0]["lon_from"]), trip.iloc[0]["incident_id"]
    inc = res_after.incidents
    waiting = inc[(inc["t_reported"] <= t) & ((inc["t_assigned"] >= t) | inc["t_assigned"].isna())]
    cands = [{"incident_id": r.incident_id, "customers": int(r.customers), "sensitive": bool(r.sensitive),
              "wait_min": int(t - r.t_reported), "travel_min": round(float(travel_min(lat, lon, r.lat, r.lon)), 1),
              "duration_min": int(r.nominal_duration)} for r in waiting.itertuples()]
    if recommended not in {c["incident_id"] for c in cands}:
        rec = inc[inc["incident_id"] == recommended]
        if not rec.empty:
            r = rec.iloc[0]
            cands.append({"incident_id": recommended, "customers": int(r.customers), "sensitive": bool(r.sensitive),
                          "wait_min": int(t - r.t_reported),
                          "travel_min": round(float(travel_min(lat, lon, r.lat, r.lon)), 1),
                          "duration_min": int(r.nominal_duration)})
    row = {"source": "human", "crisis": crisis, "t": int(t), "crew_id": crew_id, "recommended": recommended,
           "chosen": chosen, "accepted": chosen == recommended, "reason": reason, "candidates": json.dumps(cands)}
    path = path or decision_log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row], columns=LOG_COLUMNS).to_csv(path, mode="a", header=not path.exists(), index=False)
    return row


def read_decision_log(path: Path | None = None) -> pd.DataFrame:
    path = path or decision_log_path()
    if not path.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    df = pd.read_csv(path, keep_default_na=False)
    df["accepted"] = df["accepted"].astype(str).str.lower().eq("true")
    return df


# --------------------------------------------------------------------------- learning
@dataclass
class Fit:
    alpha: float
    beta: float
    lam: float
    n_decisions: int
    n_overrides: int
    alpha_ci: tuple[float, float] = (math.nan, math.nan)
    beta_ci: tuple[float, float] = (math.nan, math.nan)
    log_likelihood: float = math.nan

    def as_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


def _choice_sets(df_log: pd.DataFrame) -> list[tuple[dict, int]]:
    """[(features as numpy arrays, index of the chosen candidate)] for decisions with at least 2 candidates."""
    sets = []
    for r in df_log.itertuples(index=False):
        cands = json.loads(r.candidates)
        if len(cands) < 2:
            continue                                         # no choice → no information
        ids = [c["incident_id"] for c in cands]
        if r.chosen not in ids:
            continue
        sets.append(({f: np.array([c[f] for c in cands], dtype=float) for f in FEATURES}, ids.index(r.chosen)))
    return sets


def _stack(sets):
    feats = {f: np.concatenate([c[f] for c, _ in sets]) for f in FEATURES}
    sizes = np.array([len(c["customers"]) for c, _ in sets])
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    chosen = starts + np.array([k for _, k in sets])
    return feats, sizes, starts, chosen


def _nll(theta, feats, sizes, starts, chosen):
    alpha, beta, lam = theta
    u = lam * log_scores(feats, alpha, beta)
    m = np.maximum.reduceat(u, starts)
    lse = np.log(np.add.reduceat(np.exp(u - np.repeat(m, sizes)), starts)) + m
    return -(u[chosen] - lse).sum()


def fit_preferences(df_log: pd.DataFrame, n_boot: int = 10, seed: int = 0) -> Fit:
    """Maximum-likelihood α, β (and decisiveness λ) from the decision log, with bootstrap 90 % intervals."""
    from scipy.optimize import minimize

    sets = _choice_sets(df_log)
    if len(sets) < 20:
        raise ValueError(f"only {len(sets)} decisions with a real choice: not enough to learn (need ≥ 20)")
    bounds = [(0.0, 30.0), (0.0, 3.0), (0.05, 50.0)]

    def solve(s, starts=([5.0, 0.1, 2.0], [10.0, 0.5, 4.0])):
        args = _stack(s)
        best = None
        for x0 in starts:
            r = minimize(_nll, x0, args=args, method="L-BFGS-B", bounds=bounds)
            if best is None or r.fun < best.fun:
                best = r
        return best

    r = solve(sets)
    fit = Fit(float(r.x[0]), float(r.x[1]), float(r.x[2]), len(sets),
              int((~df_log["accepted"].astype(bool)).sum()), log_likelihood=float(-r.fun))
    if n_boot:
        rng = np.random.default_rng(seed)
        boots = np.array([solve([sets[k] for k in rng.integers(0, len(sets), len(sets))], [r.x]).x for _ in range(n_boot)])
        fit.alpha_ci = tuple(round(float(q), 2) for q in np.quantile(boots[:, 0], [0.05, 0.95]))
        fit.beta_ci = tuple(round(float(q), 3) for q in np.quantile(boots[:, 1], [0.05, 0.95]))
    return fit


def agreement_rate(df_log: pd.DataFrame, alpha: float, beta: float) -> float:
    """Share of decisions where the score with (α, β) ranks first what the manager actually chose."""
    sets = _choice_sets(df_log)
    if not sets:
        return math.nan
    return float(np.mean([int(np.argmax(log_scores(c, alpha, beta))) == k for c, k in sets]))


# --------------------------------------------------------------------------- free-text reasons
THEME_KEYWORDS = {
    "sensitive customers": ["hospital", "clinic", "nursing", "sensitive", "medical", "oxygen", "school", "elderly"],
    "waiting time / fairness": ["wait", "since", "hours", "forgotten", "fair", "mayor", "complain", "longer", "ages"],
    "distance / access": ["far", "road", "close", "next to", "area", "quicker", "risky", "route"],
}


def themes_by_keywords(reasons: list[str]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for k, r in enumerate(reasons):
        low = r.lower()
        theme = next((t for t, kws in THEME_KEYWORDS.items() if any(w in low for w in kws)), "other")
        out.setdefault(theme, []).append(k)
    return out


THEME_SYSTEM = ("You group short free-text reasons written by a power-grid crisis manager when he overrode "
                "the copilot. Answer ONLY with JSON: {\"themes\": [{\"name\": str, \"ids\": [int, ...]}]}. "
                "Use 2 to 5 short theme names. Every id must appear exactly once.")


def group_reasons(reasons: list[str], client=None, max_items: int = 40) -> tuple[dict[str, list[int]], str]:
    """(theme → reason indexes, source). The LLM groups a sample of at most `max_items` reasons (spread over the
    whole log); if it is absent or its answer is invalid, keyword rules group ALL the reasons."""
    reasons = list(reasons)
    if client is not None and reasons:
        idx = np.unique(np.linspace(0, len(reasons) - 1, min(max_items, len(reasons))).astype(int)).tolist()
        prompt = "\n".join(f"{k}: {reasons[i]}" for k, i in enumerate(idx))
        data = client.generate_json(prompt, THEME_SYSTEM, task="group_reasons", max_tokens=600)
        try:
            groups, seen = {}, set()
            for t in data["themes"]:
                ids = []
                for i in map(int, t["ids"]):
                    if 0 <= i < len(idx) and i not in seen:
                        seen.add(i)
                        ids.append(i)
                if ids:
                    groups.setdefault(str(t["name"]), []).extend(idx[i] for i in ids)
            missing = [idx[i] for i in range(len(idx)) if i not in seen]
            if groups and len(missing) <= 0.2 * len(idx):        # small gaps are repaired, big ones rejected
                if missing:
                    groups.setdefault("other", []).extend(missing)
                return groups, "llm"
            log.info("LLM grouping rejected (%s of %s reasons missing)", len(missing), len(idx))
        except (TypeError, KeyError, ValueError, IndexError):
            pass
    return themes_by_keywords(reasons), "keywords"


# --------------------------------------------------------------------------- validation gate
@dataclass
class GateResult:
    passed: bool
    checks: list[dict]
    kpi_current: dict
    kpi_proposed: dict


def validation_gate(alpha: float, beta: float, test_crises: list, base_params: dict | None = None,
                    max_customer_hours_increase_pct: float = 3.0, n_jobs: int = -1) -> GateResult:
    """Replay current vs proposed weights on crises NEVER used for learning. Safety rules are hard limits."""
    from copilot.benchmark import run
    from copilot.simulator import simulation_params

    base = base_params or simulation_params()
    cur, _ = run(test_crises, {"current": WSPT}, params=base, n_jobs=n_jobs)
    new, _ = run(test_crises, {"proposed": WSPT}, params={**base, "alpha_sensitive": alpha, "beta_wait_per_h": beta},
                 n_jobs=n_jobs)
    a, b = cur.mean(numeric_only=True), new.mean(numeric_only=True)
    ch = 100 * (b["customer_hours"] / a["customer_hours"] - 1)
    checks = [
        {"check": "no shift overrun (safety)", "current": a["shift_overruns"], "proposed": b["shift_overruns"],
         "ok": b["shift_overruns"] == 0},
        {"check": "every incident restored", "current": a["unresolved"], "proposed": b["unresolved"],
         "ok": b["unresolved"] == 0},
        {"check": "sensitive customers: p90 wait not worse (h)", "current": round(a["wait_p90_sensitive_h"], 2),
         "proposed": round(b["wait_p90_sensitive_h"], 2), "ok": b["wait_p90_sensitive_h"] <= a["wait_p90_sensitive_h"] + 0.05},
        {"check": f"customer-hours: at most +{max_customer_hours_increase_pct:g} %",
         "current": round(a["customer_hours"]), "proposed": round(b["customer_hours"]),
         "change_pct": round(ch, 2), "ok": ch <= max_customer_hours_increase_pct},
    ]
    keep = ["customer_hours", "wait_p90_sensitive_h", "wait_max_sensitive_h", "wait_max_h", "pct_restored_4h",
            "gini_towns", "restored_95pct_h", "shift_overruns", "unresolved"]
    return GateResult(all(c["ok"] for c in checks), checks, {k: float(a[k]) for k in keep},
                      {k: float(b[k]) for k in keep})


# --------------------------------------------------------------------------- versioned scoring files
def learned_dir() -> Path:
    from copilot.config import project_path

    return project_path("config", "learned")


def save_version(fit: Fit, gate: GateResult | None, status: str, approved_by: str = "", note: str = "",
                 themes: dict | None = None, folder: Path | None = None) -> Path:
    """Write config/learned/scoring_vNNN.yaml (status: proposed / approved / rejected). Never overwrites."""
    import yaml

    folder = Path(folder or learned_dir())
    folder.mkdir(parents=True, exist_ok=True)
    n = 1 + max([int(p.stem.split("_v")[-1]) for p in folder.glob("scoring_v*.yaml")] or [0])
    path = folder / f"scoring_v{n:03d}.yaml"
    doc = {"version": n, "status": status, "date": dt.datetime.now().isoformat(timespec="seconds"),
           "approved_by": approved_by, "note": note,
           "score": {"alpha_sensitive": round(fit.alpha, 3), "beta_wait_per_h": round(fit.beta, 4)},
           "evidence": {"fit": fit.as_dict(), "themes": themes or {},
                        "gate": None if gate is None else {"passed": gate.passed, "checks": gate.checks,
                                                           "kpi_current": gate.kpi_current,
                                                           "kpi_proposed": gate.kpi_proposed}}}
    path.write_text(yaml.safe_dump(json.loads(json.dumps(doc, default=float)), sort_keys=False, allow_unicode=True),
                    encoding="utf-8")
    return path


def list_versions(folder: Path | None = None) -> list[dict]:
    import yaml

    folder = Path(folder or learned_dir())
    return [yaml.safe_load(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("scoring_v*.yaml"))]


def approved_score(folder: Path | None = None) -> dict | None:
    """{'alpha_sensitive', 'beta_wait_per_h', 'version'} of the latest APPROVED version, or None (= defaults)."""
    ok = [v for v in list_versions(folder) if v.get("status") == "approved"]
    return None if not ok else {**ok[-1]["score"], "version": ok[-1]["version"]}


def main(argv: list[str] | None = None) -> None:
    """python -m copilot.ai.learning  — full loop on simulated decisions, prints what the expert would see."""
    import argparse

    from copilot.benchmark import parse_crises

    ap = argparse.ArgumentParser(description="Brick 4 — learn from overrides (simulated manager)")
    ap.add_argument("--train", default="1-30", help="crises used to collect decisions (within 1-80)")
    ap.add_argument("--test", default="81-100", help="crises used by the validation gate (never learned)")
    ap.add_argument("--alpha", type=float, default=10.0, help="hidden alpha of the simulated manager")
    ap.add_argument("--beta", type=float, default=0.5, help="hidden beta of the simulated manager")
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("copilot.simulator").setLevel(logging.ERROR)

    df = collect_log(parse_crises(args.train), SimulatedManager(args.alpha, args.beta))
    print(f"{len(df)} decisions logged, {int((~df.accepted).sum())} overrides "
          f"({100 * (~df.accepted).mean():.0f} %)")
    fit = fit_preferences(df)
    print(f"learned α = {fit.alpha:.2f} {fit.alpha_ci}  (hidden {args.alpha})   "
          f"β = {fit.beta:.3f} {fit.beta_ci}  (hidden {args.beta})")
    print(f"agreement with the manager: default weights {agreement_rate(df, 5.0, 0.1):.0%} → "
          f"learned {agreement_rate(df, fit.alpha, fit.beta):.0%}")
    from copilot.ai.llm import get_client

    groups, src = group_reasons(df.loc[~df.accepted, "reason"].tolist(), get_client())
    print(f"override reasons ({src}):", {k: len(v) for k, v in groups.items()})
    gate = validation_gate(fit.alpha, fit.beta, parse_crises(args.test), n_jobs=args.jobs)
    for c in gate.checks:
        print(("  ✔ " if c["ok"] else "  ✘ ") + f"{c['check']}: {c['current']} → {c['proposed']}")
    print("gate:", "PASSED — ready for expert review" if gate.passed else "FAILED — proposal blocked")


if __name__ == "__main__":
    main()
