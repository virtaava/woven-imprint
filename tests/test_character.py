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
