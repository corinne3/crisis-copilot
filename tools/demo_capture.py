"""Capture the demo: screenshots of every page + a screen video, driven by Playwright.

Prerequisites (once):
    pip install playwright
    python -m playwright install chromium

Run (with the app already running in another terminal: streamlit run app/Home.py --server.port 8503):
    python tools/demo_capture.py                      # → demo_shots/*.png + demo_shots/video/*.webm
    python tools/demo_capture.py --only crisis,field,whatif   # redo some sections only

Every shot is independent: if one fails, the script logs it and goes on. LLM steps can take minutes on a CPU.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

OUT = Path("demo_shots")
LONG = 600_000          # ms — LLM calls on CPU can be slow


def idle(page: Page, extra_ms: int = 1500, timeout: int = LONG) -> None:
    """Wait until Streamlit has finished running the script (no 'Running…' widget, no spinner)."""
    page.wait_for_timeout(800)
    for sel in ('[data-testid="stStatusWidget"]', '[data-testid="stSpinner"]'):
        try:
            page.wait_for_selector(sel, state="hidden", timeout=timeout)
        except Exception:                                   # noqa: BLE001
            pass
    page.wait_for_timeout(extra_ms)                         # let charts and the map finish drawing


def goto(page: Page, base: str, path: str) -> None:
    page.goto(f"{base}/{path}", wait_until="networkidle", timeout=LONG)
    idle(page, 3000)


def scroll_to(page: Page, text: str) -> None:
    el = page.get_by_text(text, exact=False).first
    el.scroll_into_view_if_needed(timeout=30_000)
    page.mouse.wheel(0, -60)
    page.wait_for_timeout(700)


def shot(page: Page, name: str, anchor: str | None = None) -> None:
    try:
        if anchor:
            scroll_to(page, anchor)
        page.screenshot(path=str(OUT / f"{name}.png"))
        print("  ✔", name)
    except Exception as e:                                  # noqa: BLE001
        print("  ✘", name, "—", str(e).splitlines()[0])


def click(page: Page, button_text: str, wait_ms: int = 2000) -> bool:
    try:
        page.get_by_role("button", name=button_text, exact=False).first.click(timeout=30_000)
        idle(page, wait_ms)
        return True
    except Exception as e:                                  # noqa: BLE001
        print("  ✘ click", button_text, "—", str(e).splitlines()[0])
        return False


def type_in(page: Page, label: str, text: str) -> None:
    try:
        box = page.get_by_label(label, exact=False).first
        box.fill(text)
        box.press("Enter")
        idle(page, 2000)
    except Exception as e:                                  # noqa: BLE001
        print("  ✘ type", label, "—", str(e).splitlines()[0])


def s_home(page: Page, base: str) -> None:
    goto(page, base, "")
    shot(page, "01_home")


def s_crisis(page: Page, base: str) -> None:
    goto(page, base, "Crisis_room")
    shot(page, "02_crisis_kpis")
    shot(page, "03_crisis_curve", "Customers without power over time")
    page.wait_for_timeout(4000)                             # map tiles
    shot(page, "04_crisis_map", "Map of the crisis at time t")
    shot(page, "05_crisis_briefing", "Situation at H+")
    try:                                                    # a debatable decision: the refusal has a visible effect
        page.get_by_text("only debatable decisions").first.click()
        idle(page, 2000)
    except Exception as e:                                  # noqa: BLE001
        print("  ✘ debatable —", str(e).splitlines()[0])
    shot(page, "06_crisis_decision", "Explained decisions")
    if click(page, "Plain-language version", 2000):
        shot(page, "07_crisis_decision_llm", "Explained decisions")
    type_in(page, "Why do you refuse?", "Nursing home on the other incident, the mayor keeps calling.")
    if click(page, "Refuse this assignment", 3000):
        shot(page, "08_crisis_override_impact", "Impact of your decisions")


def s_workforce(page: Page, base: str) -> None:
    goto(page, base, "Workforce")
    shot(page, "09_workforce_steps")
    shot(page, "10_workforce_modes", "The 3 break modes")


def s_forecast(page: Page, base: str) -> None:
    goto(page, base, "Early_forecast")
    shot(page, "11_forecast")
    try:
        page.get_by_text("Reveal what really happened").first.click()
        idle(page, 2500)
    except Exception as e:                                  # noqa: BLE001
        print("  ✘ reveal —", str(e).splitlines()[0])
    shot(page, "12_forecast_revealed")
    shot(page, "13_forecast_recommendations", "What to prepare now")


def s_field(page: Page, base: str) -> None:
    goto(page, base, "Field_reports")
    shot(page, "14_field_reports_table", "Messages received")
    if click(page, "Re-plan the crisis", 3000):
        shot(page, "15_field_reports_replan", "Customer-hours outlook")


def s_whatif(page: Page, base: str) -> None:
    goto(page, base, "What_if")
    type_in(page, "Your question", "What if we lose 5 crews in Gers from H+6 to H+18?")
    shot(page, "16_whatif_answer", "Understood")
    shot(page, "17_whatif_curve", "Same strategy (WSPT)")
    type_in(page, "Your question", "What if repairs take 20% longer and travel is 10% slower?")
    shot(page, "18_whatif_crosscheck", "Understood")


def s_learning(page: Page, base: str) -> None:
    goto(page, base, "Learning_loop")
    shot(page, "19_learning_log")
    shot(page, "20_learning_reasons", "Why did the manager override")
    shot(page, "21_learning_weights", "What weights does the manager actually use")
    if click(page, "Run the validation gate", 3000):
        shot(page, "22_learning_gate", "Validation gate")
        shot(page, "23_learning_expert", "Expert decision")


def s_benchmark(page: Page, base: str) -> None:
    goto(page, base, "Benchmark")
    shot(page, "24_benchmark")
    shot(page, "25_benchmark_box", "Mean over the crises")


SECTIONS = {"home": s_home, "crisis": s_crisis, "workforce": s_workforce, "forecast": s_forecast,
            "field": s_field, "whatif": s_whatif, "learning": s_learning, "benchmark": s_benchmark}


def run(base: str, headless: bool, only: list[str]) -> None:
    OUT.mkdir(exist_ok=True)
    t0 = time.perf_counter()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless, args=["--enable-webgl", "--ignore-gpu-blocklist"])
        ctx = browser.new_context(viewport={"width": 1600, "height": 900}, device_scale_factor=1.25,
                                  record_video_dir=str(OUT / "video"), record_video_size={"width": 1600, "height": 900})
        page = ctx.new_page()
        for name in only or SECTIONS:
            print(name, "(LLM steps can take a few minutes)" if name in ("field", "learning") else "")
            SECTIONS[name](page, base)
        ctx.close()                                         # writes the video
        browser.close()
    print(f"\nDone in {time.perf_counter() - t0:.0f} s → {OUT.resolve()}")
    print("Zip the folder and send it:  Compress-Archive demo_shots demo_shots.zip -Force")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8503")
    ap.add_argument("--headless", action="store_true", help="no visible browser (the map may stay grey)")
    ap.add_argument("--only", default="", help=f"comma-separated sections among {','.join(SECTIONS)}")
    a = ap.parse_args()
    run(a.url.rstrip("/"), a.headless, [x.strip() for x in a.only.split(",") if x.strip()])
