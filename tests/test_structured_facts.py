from datetime import datetime, timedelta, timezone

from tests.helpers import FakeLLM, make_test_engine
from woven_imprint import clock
from woven_imprint.character import Character

T0 = datetime(2026, 5, 3, 12, 0, tzinfo=timezone.utc)


class FactLLM(FakeLLM):
    def __init__(self):
        super().__init__()
        self.next_facts = []

    def generate_json(self, messages, temperature=0.3):
        head = messages[0]["content"].lower()
        if "bookkeeping assistant" in head:
            return {
                "emotion": {"mood": "content", "intensity": 0.3, "cause": ""},
                "relationship": {"trust": 0.01},
                "beat": None,
                "facts": self.next_facts,
            }
        return super().generate_json(messages, temperature)

    def generate_json_robust(self, messages, temperature=0.3):
        return self.generate_json(messages, temperature)


def _char():
    engine = make_test_engine()
    engine.llm = FactLLM()
    orig = engine.create_character

    def create(*a, **kw):
        c = orig(*a, **kw)
        c.parallel = False
        c.background = False
        c.enforce_consistency = False
        c.unified_assessment = True
        return c

    engine.create_character = create
    return engine, engine.create_character("Ada")


def test_parse_facts_accepts_strings_and_objects():
    out = Character._parse_facts(
        [
            "The visitor moved to Oulu last spring.",
            "short",
            {
                "statement": "The visitor's cat is named Pixel.",
                "subject": "user",
                "predicate": "has_cat_named",
                "object": "Pixel",
                "event_time": "2026-04-01",
            },
            {"statement": "x"},
        ],
        5,
    )
    assert out[0] == {
        "statement": "The visitor moved to Oulu last spring.",
        "subject": None,
        "predicate": None,
        "object": None,
        "event_time": None,
    }
    assert out[1]["predicate"] == "has_cat_named" and out[1]["event_time"] == "2026-04-01"
    assert len(out) == 2


def test_structured_supersession_across_whole_store():
    engine, char = _char()
    llm = engine.llm
    with clock.override(T0):
        llm.next_facts = [
            {
                "statement": "The visitor's cat is named Pixel.",
                "subject": "user",
                "predicate": "has_cat_named",
                "object": "Pixel",
            }
        ]
        for _ in range(3):
            char.chat("hello", user_id="toni")  # extraction fires on turn 3
    pixel = char.facts.find_active("user", "has_cat_named")
    assert pixel and pixel["object"] == "Pixel" and pixel["memory_id"]
    # bury it under >60 unrelated core memories — beyond the old 50-row heuristic window
    llm.next_facts = []
    for i in range(70):
        char.memory.add(f"Routine note {i} about weather.", tier="core")
    with clock.override(T0 + timedelta(days=40)):
        llm.next_facts = [
            {
                "statement": "The visitor's cat is named Moss.",
                "subject": "user",
                "predicate": "has_cat_named",
                "object": "Moss",
                "event_time": "2026-06-10",
            }
        ]
        for _ in range(3):
            char.chat("news", user_id="toni")
    cur = char.facts.current("user", "has_cat_named")
    assert [f["object"] for f in cur] == ["Moss"]
    hist = char.facts.history("user", "has_cat_named")
    assert (
        hist[0]["object"] == "Pixel"
        and hist[0]["valid_to"] == "2026-06-10 00:00:00"
        and hist[0]["superseded_by"] == cur[0]["id"]
    )
    old_mem = engine.storage.get_memory(pixel["memory_id"])
    assert old_mem["status"] == "contradicted"
    new_mem = engine.storage.get_memory(cur[0]["memory_id"])
    assert (
        new_mem["metadata"]["fact_id"] == cur[0]["id"]
        and new_mem["metadata"]["contradicts"] == pixel["memory_id"]
    )
    assert new_mem["metadata"]["user_id"] == "toni"


def test_same_object_reinforces_instead_of_duplicating():
    engine, char = _char()
    llm = engine.llm
    llm.next_facts = [
        {
            "statement": "The visitor lives in Oulu.",
            "subject": "user",
            "predicate": "lives_in",
            "object": "Oulu",
        }
    ]
    for _ in range(6):
        char.chat("hi", user_id="toni")  # extraction on turns 3 and 6
    assert len(char.facts.history("user", "lives_in")) == 1
    f = char.facts.find_active("user", "lives_in")
    mem = engine.storage.get_memory(f["memory_id"])
    assert mem["certainty"] >= 1.0  # reinforced (clamped)


def test_backdated_correction_does_not_supersede():
    """A structured fact whose event_time predates the currently-active fact's
    valid_from is a backdated correction, not a supersession: it must land in
    history without touching the active fact or its memory."""
    engine, char = _char()
    with clock.override(T0):
        char._store_structured_fact(
            {
                "statement": "The visitor's cat is named Moss.",
                "subject": "user",
                "predicate": "has_cat_named",
                "object": "Moss",
                "event_time": "2026-06-10",
            },
            "toni",
            None,
            0.75,
        )
    old = char.facts.find_active("user", "has_cat_named")
    assert old and old["object"] == "Moss"
    old_mem = engine.storage.get_memory(old["memory_id"])

    with clock.override(T0 + timedelta(days=1)):
        char._store_structured_fact(
            {
                "statement": "Actually, the visitor's cat used to be named Pixel.",
                "subject": "user",
                "predicate": "has_cat_named",
                "object": "Pixel",
                "event_time": "2026-01-15",
            },
            "toni",
            None,
            0.75,
        )

    current = char.facts.find_active("user", "has_cat_named")
    assert current is not None and current["id"] == old["id"]
    assert old_mem["status"] != "contradicted"
    assert engine.storage.get_memory(old["memory_id"])["status"] == old_mem["status"]

    hist = char.facts.history("user", "has_cat_named")
    assert [h["object"] for h in hist] == ["Pixel", "Moss"]
    new = hist[0]
    assert new["valid_to"] == old["valid_from"]
    assert new["superseded_by"] == old["id"]
    new_mem = engine.storage.get_memory(new["memory_id"])
    assert "contradicts" not in new_mem["metadata"]

    as_of_rows = char.facts.as_of("2026-03-01 00:00:00", "user", "has_cat_named")
    assert [f["id"] for f in as_of_rows] == [new["id"]]


def test_unstructured_fact_keeps_legacy_behavior():
    engine, char = _char()
    engine.llm.next_facts = ["The visitor likes tea very much."]
    for _ in range(3):
        char.chat("hi", user_id="toni")
    assert char.facts.count() == 0
    assert any("likes tea" in m["content"] for m in char.memory.get_all(tier="core"))
