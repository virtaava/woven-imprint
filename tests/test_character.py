"""Tests for Character prompt assembly — cache-friendly prefix/volatile split."""


def test_prompt_prefix_stable_across_turns():
    from tests.helpers import make_test_engine

    engine = make_test_engine()
    char = engine.create_character(
        "Stable", persona={"backstory": "A lighthouse keeper.", "personality": "calm"}
    )
    char.chat("Hello, I'm Toni")
    first = [dict(m) for m in char.last_chat_messages]
    char.chat("I found a key with a raven on it")
    second = [dict(m) for m in char.last_chat_messages]

    # Message 0 = stable persona prefix, identical across turns
    assert first[0]["role"] == "system"
    assert first[0]["content"] == second[0]["content"]
    # Volatile state (memories etc.) must NOT be inside message 0
    assert "memories" not in first[0]["content"].lower()


def test_update_relationship_zero_fills_when_llm_omits_all_keys():
    """Regression: legacy behavior — an LLM response missing every relationship
    key still produces a zero-filled deltas dict (not an empty one), so
    relationships.update is still called and the relationship record advances
    (trajectory settles to "stable" rather than the update being skipped)."""
    from tests.helpers import make_test_engine

    class EmptyRelationshipLLM:
        def generate(self, messages, **kw):
            return "ok"

        def generate_json(self, messages, **kw):
            return {}

        def generate_json_robust(self, messages, **kw):
            return {}

    engine = make_test_engine()
    char = engine.create_character("Ada", persona={})
    char.llm = EmptyRelationshipLLM()

    char._update_relationship("hi", "hello", "player1")

    rel = char.relationships.get("player1")
    assert rel is not None
    assert rel["trajectory"] == "stable"
