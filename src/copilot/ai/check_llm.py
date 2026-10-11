"""Check the local LLM end to end — every AI brick once, with timings and a rules-vs-LLM comparison.

    python -m copilot.ai.check_llm                 # model from config/default.yaml (phi4-mini)
    python -m copilot.ai.check_llm --model qwen2.5:1.5b

Writes outputs/llm_check.md (numbers you can quote in the README). No challenge data needed.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

from copilot.ai import field_reports as FR
from copilot.ai import learning as L
from copilot.ai import whatif as W
from copilot.ai.llm import LLMClient, rephrase
from copilot.templates import decision_card

EXPL = {"factors": {"customers": 412, "sensitive": True, "wait_min": 95, "travel_min": 23.4, "duration_min": 240},
        "alternative": {"incident_id": "INC_000318", "customers": 150, "delta_customer_hours": 37.0},
        "next_best_crew": {"crew_id": "SITE_07_C2", "delay_min": 18.0}, "confidence": 0.81}


def _step(title: str):
    print(f"\n── {title}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="End-to-end check of the local LLM")
    ap.add_argument("--model", default=None, help="Ollama model name (default: config llm.model)")
    args = ap.parse_args(argv)
    client = LLMClient.from_config()
    if args.model:
        client.model = args.model
    report = [f"# Local LLM check — `{client.model}`", ""]

    _step("1. Ollama and model")
    try:
        with urllib.request.urlopen(client.host + "/api/tags", timeout=3) as r:  # noqa: S310
            models = [m["name"] for m in json.loads(r.read().decode()).get("models", [])]
    except OSError:
        print(f"✘ Ollama does not answer on {client.host}. Install it (https://ollama.com) or start the Ollama app.")
        return 1
    print("models installed:", ", ".join(models) or "(none)")
    if not client.available():
        print(f"✘ model {client.model} is not installed. Run:  ollama pull {client.model}")
        return 1
    print(f"✔ {client.model} ready")

    _step("2. Warm-up (the first call loads the model in memory)")
    t = time.perf_counter()
    out = client.generate("Answer with one word: ready?", task="warmup", max_tokens=5)
    warm = time.perf_counter() - t
    print(f"{'✔' if out else '✘'} first answer in {warm:.1f} s: {out!r}")
    report += [f"- warm-up (model loading): {warm:.1f} s"]

    _step("3. Rephrase a decision card (number guard)")
    card = decision_card("SITE_04_C2", "INC_000123", EXPL)
    ok = 0
    for _ in range(3):
        txt, src = rephrase(card, client)
        ok += src == "llm"
    print(f"template : {card}\nLLM      : {txt if src == 'llm' else '(rejected by the number guard)'}")
    print(f"accepted by the number guard: {ok}/3")
    report += [f"- rephrasing accepted by the number guard: {ok}/3"]

    _step("4. Field reports — 12 real-life phrasings, rules vs LLM")
    rules = FR.evaluate(FR.HARD_MESSAGES, FR.extract_rules)
    t = time.perf_counter()
    llm = FR.evaluate(FR.HARD_MESSAGES, lambda m: FR.extract_llm(m, client) or FR.extract_rules(m))
    dt = (time.perf_counter() - t) / len(FR.HARD_MESSAGES)
    print(f"rules: {rules}\nLLM  : {llm}   ({dt:.1f} s per message)")
    for m in FR.HARD_MESSAGES[:4]:
        u = FR.understand(m["message"], client)
        print(f"  {m['message'][:60]:60s} → {u.type}, {u.incident_id}, {u.extra_minutes}, {u.status}")
    report += [f"- field reports, type accuracy: rules {rules['type']} % → LLM {llm['type']} % "
               f"(all fields: {rules['all']} % → {llm['all']} %), {dt:.1f} s/message"]

    _step("5. What-if — question → scenario (raw LLM reading, then after the cross-check with the rules)")
    n_raw = n_final = 0
    for q in EXAMPLES:
        raw = W.parse_llm(q, client)
        raw_desc = W.validate(W.Scenario(raw.crisis, [dict(c) for c in raw.changes]), q).describe() if raw else None
        final = W.reconcile(raw, q)
        expected = W.validate(W.parse_rules(q), q).describe()
        n_raw += raw_desc == expected
        n_final += final.describe() == expected
        print(f"{'✔' if final.describe() == expected else '✘'} {q}\n    LLM   → {raw_desc}\n    final → {final.describe()}"
              + (f"\n    issues: {final.issues}" if final.issues else ""))
    report += [f"- what-if: {n_raw}/{len(EXAMPLES)} questions right from the LLM alone, "
               f"{n_final}/{len(EXAMPLES)} after the cross-check with the rules"]

    _step("6. Learning loop — group free-text override reasons")
    reasons = [t.format(a="INC_000001", b=f"INC_{k:06d}") for k, t in
               enumerate(sum(L.REASONS.values(), []) * 2)][:30]
    groups, how = L.group_reasons(reasons, client)
    print(f"source: {how}")
    for th, ix in groups.items():
        print(f"  {th}: {len(ix)}  e.g. {reasons[ix[0]]!r}")
    report += [f"- reason grouping: {'LLM answer valid' if how == 'llm' else 'LLM answer invalid → keyword fallback'}"]

    u = client.usage()
    print(f"\n── Total: {u['calls']} calls, {u['failed']} failed, {u['seconds']} s, {u['tokens']} tokens")
    report += [f"- total: {u['calls']} calls, {u['seconds']} s, {u['tokens']} tokens, all on CPU, no cloud", ""]
    from copilot.config import load_config, project_path

    out = project_path(load_config()["paths"]["outputs"], "llm_check.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(report), encoding="utf-8")
    print(f"Report written to {out}")
    return 0


EXAMPLES = ["What if we lose 5 crews in Gers from H+6 to H+18?",
            "What if we get 10 extra crews in Ariège from H+4?",
            "Snow on the passes: mountain roads 3 times slower",
            "What if repairs take 20% longer and travel is 10% slower?",
            "Rotation with 2 groups 6h apart and smart breaks?"]

if __name__ == "__main__":
    raise SystemExit(main())
