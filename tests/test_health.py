from tests.helpers import FakeEmbedder, FakeLLM, make_test_engine
from woven_imprint.engine import Engine


class BrokenJSONLLM(FakeLLM):
    def generate_json(self, messages, temperature=0.3, **kw):
        raise ValueError("small model garbage")

    def generate_json_robust(self, messages, temperature=0.3, **kw):
        raise ValueError("small model garbage")


def test_health_counts_subsystem_failures():
    engine = Engine(db_path=":memory:", llm=BrokenJSONLLM(), embedding=FakeEmbedder())
    char = engine.create_character("Sick")
    char.background = False
    char.parallel = False
    char.enforce_consistency = False
    # Asserts per-subsystem (emotion/relationship) failure keys, which only
    # the legacy multi-call path produces; unified mode reports one
    # "assessment" failure instead.
    char.unified_assessment = False
    char.chat("hello", user_id="u1")

    h = char.health()
    assert h["subsystems"]["emotion"]["failure"] >= 1
    assert h["subsystems"]["relationship"]["failure"] >= 1
    assert "small model garbage" in h["subsystems"]["emotion"]["last_error"]
    assert h["worker"] is None  # background off, never created


def test_health_counts_successes():
    engine = make_test_engine()
    char = engine.create_character("Well")
    char.chat("hello", user_id="u1")
    h = char.health()
    assert h["subsystems"]["emotion"]["success"] >= 1
    assert h["subsystems"]["emotion"]["failure"] == 0


def test_session_id_captured_at_submit():
    """Facts must be tagged with the session active when the turn happened."""
    engine = make_test_engine()
    char = engine.create_character("Tagger")
    sid1 = char.start_session()
    char.chat("I adopted a cat named Viima")  # extraction turn (turn 0 % 3 == 0)
    char.flush()
    facts = [m for m in char.memory.get_all(tier="core") if m.get("session_id")]
    assert all(f["session_id"] == sid1 for f in facts)
