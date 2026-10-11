from copilot.ai.llm import LLMClient, number_guard, numbers_in, rephrase


class FakeClient(LLMClient):
    """Returns canned answers instead of calling Ollama."""

    def __init__(self, answers):
        super().__init__()
        self.answers = list(answers)
        self._available = True

    def generate(self, prompt, system="", json_mode=False, task="generic", max_tokens=400):
        return self.answers.pop(0) if self.answers else None


def test_numbers_and_guard():
    assert numbers_in("Crew 3 goes, 120 min, 4,5 h") == {3.0, 120.0, 4.5}
    assert number_guard("120 customers, 45 min", "45 minutes for 120 customers")
    assert not number_guard("120 customers", "about 150 customers")


def test_rephrase_falls_back_on_invented_numbers():
    text = "Crew SITE_01_C1 sent to INC_1: 120 customers off supply."
    assert rephrase(text, FakeClient(["Send crew to INC_1 now: 120 homes are dark."]))[1] == "llm"
    out, src = rephrase(text, FakeClient(["Send crew now: 999 homes are dark."]))
    assert src == "template" and out == text
    assert rephrase(text, None) == (text, "template")


def test_unreachable_ollama_is_degraded_mode():
    c = LLMClient(host="http://127.0.0.1:9")          # nothing listens there
    assert c.available() is False
    assert c.generate("hello") is None
    assert c.generate_json("hello") is None
