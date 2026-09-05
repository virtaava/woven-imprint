"""Tier 3o: `entities` field in the unified turn assessment.

Spec: docs/superpowers/specs/2026-09-04-tier3o-entity-linking.md

The unified JSON gains an optional `entities` list (<=8 short canonical
names). Parsing is fault-tolerant like every other section: absent,
malformed, or non-list payloads yield [] and never raise.
"""

from __future__ import annotations

from woven_imprint.persona.assessment import TurnAssessor


def test_parse_entities_happy_path():
    out = TurnAssessor._parse_entities(["Rocket", "Caroline", "San Diego"])
    assert out == ["Rocket", "Caroline", "San Diego"]


def test_parse_entities_cleans_dedups_caps():
    raw = ["  Rocket ", "rocket", "", 42, None, "A", "B", "C", "D", "E", "F", "G", "H"]
    out = TurnAssessor._parse_entities(raw)
    assert out[0] == "Rocket"  # first-seen casing kept, stripped
    assert "rocket" not in out[1:]  # case-fold dedup
    assert 42 not in out and None not in out and "" not in out
    assert len(out) <= 8


def test_parse_entities_non_list_returns_empty():
    assert TurnAssessor._parse_entities("Rocket") == []
    assert TurnAssessor._parse_entities({"e": 1}) == []
    assert TurnAssessor._parse_entities(None) == []


def _mk_char(payload):
    """Character with a FakeLLM whose generate_json_robust returns `payload`."""
    from tests.helpers import make_test_engine

    engine = make_test_engine()
    char = engine.create_character("Entity Test")

    class _LLM:
        def generate(self, messages, **kw):
            return "ok"

        def generate_json_robust(self, messages, temperature=0.3, **kw):
            return payload

        def generate_json(self, messages, **kw):
            return payload

    char.llm = _LLM()
    char.assessor.llm = char.llm
    return char


def test_set_entities_merges_metadata():
    from tests.helpers import make_test_engine

    engine = make_test_engine()
    char = engine.create_character("Store Test")
    row = char.memory.add(content="[User] I adopted Rocket", metadata={"user_id": "u1"})
    updated = char.memory.set_entities(row["id"], ["Rocket"])
    assert updated["metadata"]["entities"] == ["Rocket"]
    assert updated["metadata"]["user_id"] == "u1"  # merge, not replace


def test_ingest_attaches_entities_to_turn_memory():
    char = _mk_char({"emotion": {}, "facts": [], "entities": ["Rocket", "Caroline"]})
    char.ingest("user", "Caroline adopted a beagle named Rocket", user_id="Melanie")
    mems = char.storage.get_memories(char.id, limit=5)
    turn = [m for m in mems if "Rocket" in m["content"]]
    assert turn and turn[0]["metadata"].get("entities") == ["Rocket", "Caroline"]


def test_background_chat_attaches_entities_after_flush():
    char = _mk_char({"emotion": {}, "facts": [], "entities": ["Rocket"]})
    char.background = True
    char.chat("tell me about Rocket", user_id="Melanie")
    char.flush()
    mems = char.storage.get_memories(char.id, limit=8)
    tagged = [m for m in mems if (m.get("metadata") or {}).get("entities") == ["Rocket"]]
    assert len(tagged) == 2  # user turn + response turn


def test_ingest_no_entities_leaves_metadata_untouched():
    char = _mk_char({"emotion": {}, "facts": [], "entities": []})
    char.ingest("user", "hello there", user_id="Melanie")
    mems = char.storage.get_memories(char.id, limit=5)
    turn = [m for m in mems if "hello there" in m["content"]]
    assert turn and "entities" not in (turn[0].get("metadata") or {})


def test_chat_stream_attaches_entities_to_both_turn_memories():
    from tests.helpers import make_test_engine

    engine = make_test_engine()
    char = engine.create_character("Entity Stream Test")

    payload = {"emotion": {}, "facts": [], "entities": ["Rocket"]}

    class _StreamLLM:
        def generate(self, messages, **kw):
            return "ok"

        def generate_stream(self, messages, **kw):
            yield "ok"

        def generate_json_robust(self, messages, temperature=0.3, **kw):
            return payload

        def generate_json(self, messages, **kw):
            return payload

    char.llm = _StreamLLM()
    char.assessor.llm = char.llm
    char.background = False
    list(char.chat_stream("tell me about Rocket", user_id="Melanie"))
    mems = char.storage.get_memories(char.id, limit=8)
    tagged = [m for m in mems if (m.get("metadata") or {}).get("entities") == ["Rocket"]]
    assert len(tagged) == 2  # user turn + response turn


class _BatchLLM:
    """Returns {"1": [...], "2": [...]} keyed by 1-based batch position."""

    def __init__(self, mapping=None, exc=None):
        self.mapping = mapping or {}
        self.exc = exc
        self.calls = 0

    def generate_json_robust(self, messages, temperature=0.3, **kw):
        self.calls += 1
        if self.exc:
            raise self.exc
        return self.mapping


def test_link_entities_backfills_only_unprocessed():
    from tests.helpers import make_test_engine

    engine = make_test_engine()
    char = engine.create_character("Backfill Test")
    a = char.memory.add(content="[User] Caroline adopted Rocket")
    b = char.memory.add(content="[User] dinner was pasta")
    char.memory.set_entities(b["id"], [])  # already processed (empty marker)
    llm = _BatchLLM({"1": ["Caroline", "Rocket"]})
    n = char.memory.link_entities(llm, batch_size=10)
    assert n == 1 and llm.calls == 1
    assert char.storage.get_memory(a["id"])["metadata"]["entities"] == ["Caroline", "Rocket"]
    # second run: nothing left to do, no LLM call
    llm2 = _BatchLLM({})
    assert char.memory.link_entities(llm2) == 0 and llm2.calls == 0


def test_link_entities_batch_failure_skips_not_raises():
    from tests.helpers import make_test_engine

    engine = make_test_engine()
    char = engine.create_character("Backfill Fail")
    char.memory.add(content="[User] one")
    n = char.memory.link_entities(_BatchLLM(exc=RuntimeError("down")), batch_size=10)
    assert n == 0  # skipped, no exception


def test_link_entities_marks_empty_result_as_processed():
    from tests.helpers import make_test_engine

    engine = make_test_engine()
    char = engine.create_character("Backfill Empty")
    a = char.memory.add(content="[User] nothing notable")
    n = char.memory.link_entities(_BatchLLM({}), batch_size=10)  # LLM returns {} -> no entities
    assert n == 1
    assert char.storage.get_memory(a["id"])["metadata"]["entities"] == []
