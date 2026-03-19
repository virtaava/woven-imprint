"""Tests for Character.ingest() — external message recording without LLM generation."""

import pytest

from woven_imprint import Engine


class FakeEmbedder:
    """Deterministic embedder for testing."""

    def __init__(self):
        self._vocab = {}
        self._next = 0

    def embed(self, text):
        vec = [0.0] * 50
        for word in text.lower().split()[:10]:
            if word not in self._vocab:
                self._vocab[word] = self._next % 50
                self._next += 1
            vec[self._vocab[word]] += 1.0
        mag = sum(x * x for x in vec) ** 0.5
        if mag > 0:
            vec = [x / mag for x in vec]
        return vec

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]

    def dimensions(self):
        return 50


class FakeLLM:
    """Predictable LLM that tracks calls for assertion."""

    def __init__(self):
        self.call_count = 0
        self.generate_call_count = 0

    def generate(self, messages, **kw):
        self.call_count += 1
        self.generate_call_count += 1
        return "I hear you."

    def generate_json(self, messages, **kw):
        self.call_count += 1
        user = messages[-1].get("content", "") if messages else ""
        system = messages[0].get("content", "") if messages else ""

        # Fact extraction
        if "extract" in system.lower() or "facts" in user.lower():
            return ["The user mentioned something notable"]

        # Relationship assessment
        if "relationship" in system.lower() or "trust" in user.lower():
            return {
                "trust": 0.03,
                "affection": 0.01,
                "respect": 0.02,
                "familiarity": 0.05,
                "tension": 0.0,
            }

        return {}


def _setup():
    llm = FakeLLM()
    embedder = FakeEmbedder()
    engine = Engine(db_path=":memory:", llm=llm, embedding=embedder)
    _orig_create = engine.create_character

    def _create_sequential(*args, **kwargs):
        char = _orig_create(*args, **kwargs)
        char.parallel = False
        return char

    engine.create_character = _create_sequential
    return engine, llm, embedder


class TestIngest:
    def test_ingest_stores_user_message_in_memory(self):
        engine, _, _ = _setup()
        char = engine.create_character("Alice", persona={})
        char.ingest("user", "Hello Alice, I'm back")
        buf = char.memory.count(tier="buffer")
        assert buf >= 1
        memories = char.memory.get_all(tier="buffer")
        content = " ".join(m["content"] for m in memories)
        assert "Hello Alice" in content
        engine.close()

    def test_ingest_stores_assistant_message_in_memory(self):
        engine, _, _ = _setup()
        char = engine.create_character("Alice", persona={})
        char.ingest("assistant", "Welcome back! How are you?")
        buf = char.memory.count(tier="buffer")
        assert buf >= 1
        memories = char.memory.get_all(tier="buffer")
        content = " ".join(m["content"] for m in memories)
        assert "Welcome back" in content
        engine.close()

    def test_ingest_adds_to_conversation_buffer(self):
        engine, _, _ = _setup()
        char = engine.create_character("Alice", persona={})
        char.ingest("user", "Message 1")
        char.ingest("assistant", "Reply 1")
        char.ingest("user", "Message 2")
        assert char._context.turn_count == 3
        engine.close()

    def test_ingest_does_not_call_llm_generate(self):
        """ingest() must never call llm.generate() — only generate_json for extraction."""
        engine, llm, _ = _setup()
        char = engine.create_character("Alice", persona={})
        llm.generate_call_count = 0
        char.ingest("user", "Hello Alice")
        char.ingest("assistant", "Hello there!")
        assert llm.generate_call_count == 0
        engine.close()

    def test_ingest_with_user_id_creates_relationship(self):
        engine, _, _ = _setup()
        char = engine.create_character("Alice", persona={})
        char.ingest("user", "Hello Alice", user_id="player1")
        rel = char.relationships.get("player1")
        assert rel is not None
        engine.close()

    def test_ingest_relationship_evolves(self):
        engine, _, _ = _setup()
        char = engine.create_character("Alice", persona={})
        char.ingest("user", "I trust you completely", user_id="player1")
        char.ingest("assistant", "That means a lot to me", user_id="player1")
        char.ingest("user", "Let's work together", user_id="player1")

        rel = char.relationships.get("player1")
        assert rel is not None
        assert rel["dimensions"]["familiarity"] > 0
        engine.close()

    def test_ingest_returns_none(self):
        engine, _, _ = _setup()
        char = engine.create_character("Alice", persona={})
        result = char.ingest("user", "Hello")
        assert result is None
        engine.close()

    def test_ingest_auto_starts_session(self):
        engine, _, _ = _setup()
        char = engine.create_character("Alice", persona={})
        assert char._session_id is None
        char.ingest("user", "Hello")
        assert char._session_id is not None
        engine.close()

    def test_ingest_increments_turn_count(self):
        engine, _, _ = _setup()
        char = engine.create_character("Alice", persona={})
        assert char._turn_count == 0
        char.ingest("user", "Hello")
        assert char._turn_count == 1
        char.ingest("assistant", "Hi there")
        assert char._turn_count == 2
        engine.close()

    def test_ingest_rejects_invalid_role(self):
        engine, _, _ = _setup()
        char = engine.create_character("Alice", persona={})
        with pytest.raises(ValueError, match="role must be"):
            char.ingest("system", "Hello")
        engine.close()

    def test_ingest_truncates_long_content(self):
        engine, _, _ = _setup()
        char = engine.create_character("Alice", persona={})
        huge = "x" * 100_000
        char.ingest("user", huge)  # should not crash
        buf = char.memory.count(tier="buffer")
        assert buf >= 1
        engine.close()

    def test_ingest_fact_extraction_at_interval(self):
        """Fact extraction should trigger at the configured interval (turn 0)."""
        engine, _, _ = _setup()
        char = engine.create_character("Alice", persona={})
        # Turn 0 triggers extraction
        char.ingest("user", "My brother Marcus is missing")
        core = char.memory.count(tier="core")
        assert core >= 1
        engine.close()

    def test_ingest_mixed_with_chat(self):
        """ingest and chat should share the same session and memory."""
        engine, llm, _ = _setup()
        char = engine.create_character("Alice", persona={})
        # Ingest external messages first
        char.ingest("user", "Context from SillyTavern")
        char.ingest("assistant", "Character replied in SillyTavern")
        # Then use chat normally
        response = char.chat("Now talking through woven-imprint")
        assert isinstance(response, str)
        # Should have memories from both ingest and chat
        buf = char.memory.count(tier="buffer")
        assert buf >= 4  # 2 from ingest + 2 from chat
        engine.close()
