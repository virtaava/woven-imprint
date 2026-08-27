"""Tests for Character.ingest() routing through unified bookkeeping when enabled."""

from tests.helpers import FakeLLM, make_test_engine


class StructuredFactLLM(FakeLLM):
    """FakeLLM that distinguishes the unified turn-assessment call from the
    legacy per-subsystem calls (fact extraction / relationship assessment) by
    inspecting the system prompt, and counts calls to plain `generate()`
    separately so tests can assert ingest() never triggers real generation.
    """

    def __init__(self):
        super().__init__()
        self.unified_calls = 0
        self.legacy_calls = 0
        self.generate_calls = 0

    def generate(self, messages, **kw):
        self.generate_calls += 1
        return super().generate(messages, **kw)

    def generate_json(self, messages, **kw):
        head = messages[0].get("content", "") if messages else ""
        head = head.lower()
        if "bookkeeping assistant" in head:
            self.unified_calls += 1
            self.call_count += 1
            return {
                "emotion": {"mood": "content", "intensity": 0.3, "cause": ""},
                "relationship": {"trust": 0.01},
                "beat": None,
                "facts": [
                    {
                        "statement": "The visitor lives in Oulu.",
                        "subject": "user",
                        "predicate": "lives_in",
                        "object": "Oulu",
                        "event_time": None,
                    }
                ],
            }
        if "extract" in head or "fact" in head or "relationship" in head:
            self.legacy_calls += 1
        return super().generate_json(messages, **kw)

    def generate_json_robust(self, messages, temperature=0.3, **kw):
        return self.generate_json(messages, temperature=temperature, **kw)


def _char(unified: bool):
    engine = make_test_engine()
    engine.llm = StructuredFactLLM()
    char = engine.create_character("Ada")
    char.background = False
    char.parallel = False
    char.enforce_consistency = False
    char.unified_assessment = unified
    return engine, char


def test_ingest_uses_unified_call_and_creates_structured_facts():
    engine, char = _char(True)
    char.ingest("user", "I live in Oulu these days.", user_id="toni")
    assert engine.llm.unified_calls == 1
    assert engine.llm.legacy_calls == 0
    assert char.facts.find_active("user", "lives_in")["object"] == "Oulu"
    assert engine.llm.generate_calls == 0  # still no generation


def test_ingest_legacy_path_when_unified_off():
    engine, char = _char(False)
    char.ingest("user", "I live in Oulu these days.", user_id="toni")
    assert engine.llm.unified_calls == 0
    assert engine.llm.legacy_calls >= 1
    assert char.facts.count() == 0


def test_ingest_assistant_turn_is_character_speech():
    engine, char = _char(True)
    char.ingest("assistant", "I remember the harbor.", user_id="toni")
    mems = char.memory.get_all(tier="buffer")
    assert mems and mems[0]["content"].startswith("[Ada]") and mems[0]["role"] == "character"
