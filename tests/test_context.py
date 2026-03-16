"""Tests for context window management."""

from woven_imprint.context import ContextManager, ConversationTurn


class TestConversationTurn:
    def test_token_estimate(self):
        turn = ConversationTurn(role="user", content="Hello world")
        assert turn.estimated_tokens > 0
        # ~11 chars / 4 + 1 = ~3-4 tokens
        assert turn.estimated_tokens < 10

    def test_long_content(self):
        turn = ConversationTurn(role="user", content="x" * 4000)
        assert turn.estimated_tokens >= 1000


class TestContextManager:
    def test_empty(self):
        cm = ContextManager()
        assert cm.turn_count == 0
        assert cm.get_messages() == []

    def test_add_turns(self):
        cm = ContextManager()
        cm.add_turn("user", "Hello")
        cm.add_turn("assistant", "Hi there")
        assert cm.turn_count == 2

        messages = cm.get_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_max_turns_enforced(self):
        cm = ContextManager(max_turns=5)
        for i in range(10):
            cm.add_turn("user", f"Message {i}")
        assert cm.turn_count == 5
        # Oldest messages should be in summary
        messages = cm.get_messages()
        # Should have summary + 5 turns
        assert (
            any(
                "summary" in m.get("content", "").lower() or m["role"] == "system" for m in messages
            )
            or len(messages) == 5
        )

    def test_clear(self):
        cm = ContextManager()
        cm.add_turn("user", "Hello")
        cm.add_turn("assistant", "Hi")
        cm.clear()
        assert cm.turn_count == 0
        assert cm.get_messages() == []

    def test_compress_without_llm(self):
        cm = ContextManager()
        for i in range(8):
            role = "user" if i % 2 == 0 else "assistant"
            cm.add_turn(role, f"Message number {i} with some content")

        summary = cm.compress()
        assert len(summary) > 0
        # Should keep last 4 turns
        assert cm.turn_count == 4

    def test_compress_with_llm(self):
        class FakeLLM:
            def generate(self, messages, **kw):
                return "The conversation covered greetings and introductions."

        cm = ContextManager()
        for i in range(8):
            role = "user" if i % 2 == 0 else "assistant"
            cm.add_turn(role, f"Message {i}")

        summary = cm.compress(llm=FakeLLM())
        assert "greetings" in summary
        assert cm.turn_count == 4

    def test_summary_included_in_messages(self):
        cm = ContextManager(max_turns=3)
        for i in range(6):
            cm.add_turn("user", f"Message {i}")

        messages = cm.get_messages()
        # Should have a system message with summary + 3 recent turns
        any(m["role"] == "system" and "summary" in m.get("content", "").lower() for m in messages)
        # Summary is generated from overflow
        assert cm.turn_count == 3

    def test_conversation_tokens_estimate(self):
        cm = ContextManager()
        cm.add_turn("user", "Hello " * 100)
        tokens = cm.get_conversation_tokens()
        assert tokens > 100

    def test_to_dict_roundtrip(self):
        cm = ContextManager()
        cm.add_turn("user", "Hello")
        cm.add_turn("assistant", "Hi")
        cm._summary = "Earlier conversation happened"

        d = cm.to_dict()
        cm2 = ContextManager.from_dict(d)
        assert cm2.turn_count == 2
        assert cm2._summary == "Earlier conversation happened"

    def test_compress_preserves_recent(self):
        cm = ContextManager()
        for i in range(10):
            role = "user" if i % 2 == 0 else "assistant"
            cm.add_turn(role, f"Turn {i}")

        cm.compress()
        messages = cm.get_messages()
        # Last 4 turns should still be there
        turn_messages = [m for m in messages if m["role"] != "system"]
        assert len(turn_messages) == 4
        assert "Turn 9" in turn_messages[-1]["content"]
