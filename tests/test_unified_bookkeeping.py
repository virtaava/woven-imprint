from tests.helpers import FakeLLM, make_test_engine


class CountingLLM(FakeLLM):
    def __init__(self):
        super().__init__()
        self.json_calls = []

    def generate_json(self, messages, temperature=0.3):
        self.json_calls.append(messages[0]["content"][:65])
        head = messages[0]["content"].lower()
        if "bookkeeping assistant" in head:
            return {
                "emotion": {"mood": "content", "intensity": 0.4, "cause": "nice chat"},
                "relationship": {"trust": 0.03, "familiarity": 0.05},
                "beat": None,
                "facts": ["The visitor is building a quiz app for phones."],
            }
        return super().generate_json(messages, temperature)

    def generate_json_robust(self, messages, temperature=0.3):
        return self.generate_json(messages, temperature)


def _engine(unified: bool):
    engine = make_test_engine()
    engine.llm = CountingLLM()
    orig = engine.create_character

    def create(*a, **kw):
        c = orig(*a, **kw)
        c.parallel = False
        c.background = False
        c.enforce_consistency = False
        c.unified_assessment = unified
        return c

    engine.create_character = create
    return engine


def test_unified_makes_one_json_call_per_turn_and_applies_all():
    engine = _engine(True)
    char = engine.create_character("Ada")
    char.chat("I am building a quiz app for phones.", user_id="toni")
    char.chat("It is going slowly.", user_id="toni")
    char.chat("Third turn triggers extraction.", user_id="toni")
    assert len(engine.llm.json_calls) == 3
    assert char.emotion.mood == "content"
    rel = char.relationships.get("toni")
    assert rel["dimensions"]["trust"] > 0
    facts = [m for m in char.memory.get_all(tier="core") if "quiz app" in m["content"]]
    assert facts and facts[0]["metadata"]["source"] == "extraction"
    assert char.health()["subsystems"]["assessment"]["success"] == 3


def test_legacy_path_unchanged():
    engine = _engine(False)
    char = engine.create_character("Ada")
    char.chat("hello", user_id="toni")
    heads = [h.lower() for h in engine.llm.json_calls]
    assert not any("bookkeeping assistant" in h for h in heads)
    assert any("emotion" in h for h in heads) and any("relationship" in h for h in heads)
