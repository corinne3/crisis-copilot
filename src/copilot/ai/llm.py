"""One shared, small, LOCAL language model (Ollama) for every AI brick — with guards and a no-AI fallback.

Frugality rules applied here:
    - one small model for everything (default phi4-mini: Microsoft, 3.8B parameters, MIT licence, ~2.5 GB),
      on a laptop CPU, no cloud, no GPU;
    - the LLM never decides and never computes: it only rephrases, extracts or groups text;
    - every output is checked (JSON schema, number guard); if the check fails or Ollama is not running,
      the caller uses its deterministic fallback — the app works the same, only the wording is plainer;
    - every call is metered (duration, tokens) so the cost of the AI layer is visible in the app.

Setup (once):  install Ollama (https://ollama.com), then `ollama pull phi4-mini`.
"""
from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

log = logging.getLogger("copilot.llm")

_NUM = re.compile(r"(?<![\w.])-?\d+(?:[.,]\d+)?")


@dataclass
class CallStats:
    task: str
    seconds: float
    prompt_tokens: int = 0
    output_tokens: int = 0
    ok: bool = True


@dataclass
class LLMClient:
    host: str = "http://localhost:11434"
    model: str = "phi4-mini"
    timeout_s: float = 60.0
    temperature: float = 0.1
    keep_alive: str = "30m"
    stats: list[CallStats] = field(default_factory=list)
    _available: bool | None = None

    @classmethod
    def from_config(cls, cfg: dict | None = None) -> LLMClient:
        if cfg is None:
            from copilot.config import load_config

            cfg = load_config()
        c = cfg.get("llm", {})
        return cls(host=c.get("host", cls.host), model=c.get("model", cls.model),
                   timeout_s=float(c.get("timeout_s", 60)), temperature=float(c.get("temperature", 0.1)),
                   keep_alive=str(c.get("keep_alive", "30m")))

    # ------------------------------------------------------------------ transport
    def _post(self, path: str, payload: dict, timeout: float) -> dict:
        req = urllib.request.Request(self.host.rstrip("/") + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:   # noqa: S310 (local host from config)
            return json.loads(r.read().decode())

    def available(self, refresh: bool = False) -> bool:
        """True if Ollama answers AND the model is pulled. Cached (checked once)."""
        if self._available is not None and not refresh:
            return self._available
        try:
            with urllib.request.urlopen(self.host.rstrip("/") + "/api/tags", timeout=2) as r:  # noqa: S310
                models = [m.get("name", "") for m in json.loads(r.read().decode()).get("models", [])]
            base = self.model.split(":")[0]
            self._available = any(m == self.model or m.split(":")[0] == base for m in models)
            if not self._available:
                log.warning("Ollama is running but model %s is not pulled (ollama pull %s)", self.model, self.model)
        except (urllib.error.URLError, OSError, ValueError):
            self._available = False
        return self._available

    def generate(self, prompt: str, system: str = "", json_mode: bool = False, task: str = "generic",
                 max_tokens: int = 400) -> str | None:
        """Raw completion, or None if the model is unavailable / fails (the caller then falls back)."""
        if not self.available():
            return None
        payload = {"model": self.model, "prompt": prompt, "system": system, "stream": False,
                   "keep_alive": self.keep_alive,                     # avoid reloading the model at every pause
                   "options": {"temperature": self.temperature, "num_predict": max_tokens}}
        if json_mode:
            payload["format"] = "json"
        clock = time.perf_counter()
        try:
            out = self._post("/api/generate", payload, self.timeout_s)
        except (urllib.error.URLError, OSError, ValueError) as e:
            log.warning("LLM call failed (%s): %r", task, e)
            self.stats.append(CallStats(task, time.perf_counter() - clock, ok=False))
            return None
        self.stats.append(CallStats(task, time.perf_counter() - clock, int(out.get("prompt_eval_count", 0)),
                                    int(out.get("eval_count", 0))))
        return out.get("response", "")

    def generate_json(self, prompt: str, system: str = "", task: str = "json", max_tokens: int = 600) -> dict | None:
        txt = self.generate(prompt, system, json_mode=True, task=task, max_tokens=max_tokens)
        if txt is None:
            return None
        try:
            data = json.loads(txt)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", txt, re.S)
            try:
                data = json.loads(m.group(0)) if m else None
            except json.JSONDecodeError:
                data = None
        if not isinstance(data, dict):
            self.stats[-1].ok = False
            return None
        return data

    def usage(self) -> dict:
        """Frugality meter: what the AI layer has cost so far in this session."""
        return {"calls": len(self.stats), "failed": sum(not s.ok for s in self.stats),
                "seconds": round(sum(s.seconds for s in self.stats), 1),
                "tokens": sum(s.prompt_tokens + s.output_tokens for s in self.stats)}


# --------------------------------------------------------------------------- guards
def numbers_in(text: str) -> set[float]:
    return {float(n.replace(",", ".")) for n in _NUM.findall(text or "")}


def number_guard(source: str, output: str, allowed: set[float] | None = None) -> bool:
    """True if every number in `output` appears in `source` (or in `allowed`): the LLM invented nothing."""
    ok = numbers_in(source) | (allowed or set())
    return all(any(abs(n - m) < 1e-6 for m in ok) for n in numbers_in(output))


REPHRASE_SYSTEM = ("You rewrite operational messages for a power-grid crisis room. Keep the meaning, keep EVERY "
                   "number exactly as written, add no number, no fact and no advice. Be short and clear.")


def rephrase(text: str, client: LLMClient | None, audience: str = "a crisis manager") -> tuple[str, str]:
    """(text, source) — source is 'llm' if the rephrasing passed the number guard, else 'template'."""
    if client is None:
        return text, "template"
    out = client.generate(f"Rewrite this for {audience}, in at most 3 sentences:\n\n{text}",
                          REPHRASE_SYSTEM, task="rephrase", max_tokens=200)
    if out and out.strip() and number_guard(text, out):
        return out.strip(), "llm"
    if out:
        log.info("rephrasing rejected by the number guard")
    return text, "template"


_CLIENT: LLMClient | None = None


def get_client(cfg: dict | None = None) -> LLMClient | None:
    """Shared client, or None when no local model is reachable (degraded mode without AI)."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = LLMClient.from_config(cfg)
        if _CLIENT.available():                     # load the model in the background while the user navigates
            import threading

            threading.Thread(target=_CLIENT.generate, args=("ok",), kwargs={"task": "warmup", "max_tokens": 1},
                             daemon=True).start()
    return _CLIENT if _CLIENT.available() else None
