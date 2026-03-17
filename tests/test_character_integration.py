"""Integration tests for the core chat loop — the main feature.

Tests the full path: chat() → memory storage → retrieval → response →
fact extraction → relationship update → emotional state → narrative arc.
"""

from woven_imprint import Engine


class FakeEmbedder:
    """Deterministic embedder — words map to consistent vector positions."""

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
    """Predictable LLM for testing the full pipeline."""

    def __init__(self):
        self.call_count = 0

    def generate(self, messages, **kw):
        self.call_count += 1
        system = messages[0]["content"] if messages else ""
        if "summariz" in system.lower():
            return "Session covered a discussion about finding a missing person."
        if "reflect" in (messages[-1].get("content", "") if messages else "").lower():
            return "I notice I've been getting more involved in this case than usual."
        if "consolidat" in system.lower():
            return "Multiple conversations about a missing person investigation."
        return "I hear you. Let me look into that."

    def generate_json(self, messages, **kw):
        self.call_count += 1
        user = messages[-1].get("content", "") if messages else ""

        # Fact extraction
        if (
            "facts" in user.lower()
            or "extract" in (messages[0].get("content", "") if messages else "").lower()
        ):
            return ["The user is looking for help with a case"]

        # Emotion assessment
        if "emotional" in (messages[0].get("content", "") if messages else "").lower():
            return {"mood": "contemplative", "intensity": 0.4, "cause": "thinking about the case"}

        # Relationship assessment
        if (
            "relationship" in (messages[0].get("content", "") if messages else "").lower()
            or "trust" in user.lower()
        ):
            return {
                "trust": 0.03,
                "affection": 0.01,
                "respect": 0.02,
                "familiarity": 0.05,
                "tension": 0.0,
            }

        # Narrative beat
        if (
            "beat" in (messages[0].get("content", "") if messages else "").lower()
            or "narrative" in (messages[0].get("content", "") if messages else "").lower()
        ):
            return {"is_beat": False}

        # Consistency check
        if (
            "consistency" in (messages[0].get("content", "") if messages else "").lower()
            or "contradiction" in (messages[0].get("content", "") if messages else "").lower()
        ):
            return {"hard_violations": [], "soft_flags": [], "score": 0.95}

        return {}


def _setup():
    llm = FakeLLM()
    embedder = FakeEmbedder()
    engine = Engine(db_path=":memory:", llm=llm, embedding=embedder)
    # Disable parallel mode in tests (threading + in-memory SQLite + fake LLMs = flaky)
    _orig_create = engine.create_character

    def _create_sequential(*args, **kwargs):
        char = _orig_create(*args, **kwargs)
        char.parallel = False
        return char

    engine.create_character = _create_sequential
    return engine, llm, embedder


class TestChatLoop:
    def test_basic_chat_returns_response(self):
        engine, llm, _ = _setup()
        char = engine.create_character("Alice", persona={"personality": "witty"})
        response = char.chat("Hello Alice")
        assert isinstance(response, str)
        assert len(response) > 0
        engine.close()

    def test_chat_stores_buffer_memories(self):
        engine, _, _ = _setup()
        char = engine.create_character("Alice", persona={})
        char.chat("Hello Alice")
        buf = char.memory.count(tier="buffer")
        assert buf >= 2  # user message + character response

        engine.close()

    def test_chat_with_user_id_creates_relationship(self):
        engine, _, _ = _setup()
        char = engine.create_character("Alice", persona={})
        char.chat("Hello", user_id="player1")
        rel = char.relationships.get("player1")
        assert rel is not None
        engine.close()

    def test_relationship_evolves_over_turns(self):
        engine, _, _ = _setup()
        char = engine.create_character("Alice", persona={})
        char.chat("Hello", user_id="player1")
        char.chat("I need your help", user_id="player1")
        char.chat("Thank you for everything", user_id="player1")

        rel = char.relationships.get("player1")
        assert rel is not None
        assert rel["dimensions"]["familiarity"] > 0
        engine.close()

    def test_emotion_updates_when_not_lightweight(self):
        engine, _, _ = _setup()
        char = engine.create_character("Alice", persona={})
        char.lightweight = False
        char.chat("Something dramatic happened!")
        # FakeLLM returns "contemplative" for emotion
        assert char.emotion.mood == "contemplative"
        engine.close()

    def test_lightweight_skips_emotion(self):
        engine, _, _ = _setup()
        char = engine.create_character("Alice", persona={})
        char.lightweight = True
        char.chat("Something happened")
        assert char.emotion.mood == "neutral"  # Not updated
        engine.close()

    def test_fact_extraction_every_3rd_turn(self):
        engine, _, _ = _setup()
        char = engine.create_character("Alice", persona={})
        # Turn 0: extraction runs
        char.chat("My brother Marcus is missing")
        core_after_1 = char.memory.count(tier="core")

        # Turn 1, 2: extraction skipped
        char.chat("He was last seen near the Thames")
        char.chat("Please help me find him")

        # Turn 3: extraction runs again
        char.chat("He works at a pub in Southwark")
        core_after_4 = char.memory.count(tier="core")

        assert core_after_4 > core_after_1
        engine.close()

    def test_conversation_buffer_accumulates(self):
        engine, _, _ = _setup()
        char = engine.create_character("Alice", persona={})
        char.chat("Message 1")
        char.chat("Message 2")
        char.chat("Message 3")
        assert char._context.turn_count == 6  # 3 user + 3 assistant
        engine.close()

    def test_session_lifecycle(self):
        engine, _, _ = _setup()
        char = engine.create_character("Alice", persona={})
        session_id = char.start_session()
        assert session_id is not None

        char.chat("Hello")
        char.chat("Goodbye")
        summary = char.end_session()

        assert summary is not None
        assert char._session_id is None
        # Summary stored as core memory
        core = char.memory.get_all(tier="core")
        summaries = [m for m in core if "Session Summary" in m["content"]]
        assert len(summaries) >= 1
        engine.close()

    def test_cross_session_memory_persistence(self):
        engine, _, _ = _setup()
        char = engine.create_character("Alice", persona={})

        # Session 1
        char.chat("My name is Marcus and I need help")
        char.end_session()

        # Session 2 — recall should find memories from session 1
        memories = char.recall("Marcus help", limit=5)
        assert len(memories) > 0
        assert any("Marcus" in m["content"] for m in memories)
        engine.close()

    def test_reflect_generates_core_memory(self):
        engine, _, _ = _setup()
        char = engine.create_character("Alice", persona={})
        # Need enough buffer memories for reflection
        for i in range(6):
            char.memory.add(f"Event {i} happened today", tier="buffer")

        reflection = char.reflect()
        assert isinstance(reflection, str)
        assert len(reflection) > 0

        core = char.memory.get_all(tier="core")
        reflections = [m for m in core if "Reflection" in m["content"]]
        assert len(reflections) >= 1
        engine.close()

    def test_consolidate_compresses_buffer(self):
        engine, _, _ = _setup()
        char = engine.create_character("Alice", persona={})
        # Fill buffer past threshold
        for i in range(15):
            char.memory.add(f"Similar memory about topic {i}", tier="buffer")

        stats = char.consolidate()
        assert stats["created"] >= 1
        assert stats["archived"] >= 1
        engine.close()

    def test_input_size_limit(self):
        engine, _, _ = _setup()
        char = engine.create_character("Alice", persona={})
        huge_message = "x" * 100_000
        response = char.chat(huge_message)
        # Should not crash, message gets truncated
        assert isinstance(response, str)
        engine.close()

    def test_get_relationship_convenience(self):
        engine, _, _ = _setup()
        char = engine.create_character("Alice", persona={})
        char.chat("Hello", user_id="player1")
        rel = char.get_relationship("player1")
        assert rel is not None
        assert "dimensions" in rel
        engine.close()

    def test_multiple_characters_isolated(self):
        engine, _, _ = _setup()
        alice = engine.create_character("Alice", persona={"personality": "serious"})
        bob = engine.create_character("Bob", persona={"personality": "cheerful"})

        alice.memory.add("Alice's secret", tier="core")
        bob.memory.add("Bob's secret", tier="core")

        alice_mems = alice.recall("secret", limit=5)
        bob_mems = bob.recall("secret", limit=5)

        # Each should only see their own memories
        alice_content = " ".join(m["content"] for m in alice_mems)
        bob_content = " ".join(m["content"] for m in bob_mems)

        assert "Alice" in alice_content
        assert "Bob" not in alice_content
        assert "Bob" in bob_content
        assert "Alice" not in bob_content
        engine.close()
