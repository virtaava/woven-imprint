"""Tests for multi-character interaction."""

from woven_imprint.storage.sqlite import SQLiteStorage
from woven_imprint.character import Character
from woven_imprint.persona.model import PersonaModel
from woven_imprint.interaction import interact, group_interaction


class FakeEmbedder:
    def embed(self, text):
        h = hash(text) % 1000
        return [h / 1000, (h * 7) % 1000 / 1000, (h * 13) % 1000 / 1000]

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]

    def dimensions(self):
        return 3


class FakeLLM:
    def __init__(self):
        self._call_count = 0

    def generate(self, messages, temperature=0.7, max_tokens=2048):
        self._call_count += 1
        # Extract character name from system prompt
        system = messages[0]["content"] if messages else ""
        if "Alice" in system:
            return f"Alice says something interesting. (call {self._call_count})"
        elif "Bob" in system:
            return f"Bob responds thoughtfully. (call {self._call_count})"
        elif "Carol" in system:
            return f"Carol observes the situation. (call {self._call_count})"
        return f"Character speaks. (call {self._call_count})"

    def generate_json(self, messages, temperature=0.3):
        return {
            "trust": 0.02,
            "affection": 0.01,
            "respect": 0.01,
            "familiarity": 0.03,
            "tension": 0.0,
        }


def _make_character(storage, name, char_id):
    persona = PersonaModel(
        {"name": name, "hard": {"name": name}, "soft": {"personality": "friendly"}},
    )
    storage.save_character(char_id, name, persona.to_dict())
    llm = FakeLLM()
    embedder = FakeEmbedder()
    char = Character(char_id, storage, llm, embedder, persona)
    char.enforce_consistency = False
    return char


class TestInteract:
    def test_basic_interaction(self):
        storage = SQLiteStorage(":memory:")
        alice = _make_character(storage, "Alice", "c1")
        bob = _make_character(storage, "Bob", "c2")

        result = interact(alice, bob, "Meeting at a café", rounds=1)

        assert len(result.turns) == 2
        assert result.turns[0].speaker == "Alice"
        assert result.turns[1].speaker == "Bob"
        assert result.situation == "Meeting at a café"
        storage.close()

    def test_multiple_rounds(self):
        storage = SQLiteStorage(":memory:")
        alice = _make_character(storage, "Alice", "c1")
        bob = _make_character(storage, "Bob", "c2")

        result = interact(alice, bob, "A debate", rounds=3)

        # 3 rounds × 2 turns each = 6 turns
        assert len(result.turns) == 6
        # Speakers alternate
        speakers = [t.speaker for t in result.turns]
        assert speakers[0] == "Alice"
        assert speakers[1] == "Bob"
        # After round 1, they swap who leads
        assert speakers[2] == "Bob"
        assert speakers[3] == "Alice"
        storage.close()

    def test_b_opens(self):
        storage = SQLiteStorage(":memory:")
        alice = _make_character(storage, "Alice", "c1")
        bob = _make_character(storage, "Bob", "c2")

        result = interact(alice, bob, "Bob approaches", rounds=1, a_opens=False)

        assert result.turns[0].speaker == "Bob"
        assert result.turns[1].speaker == "Alice"
        storage.close()

    def test_memories_created(self):
        storage = SQLiteStorage(":memory:")
        alice = _make_character(storage, "Alice", "c1")
        bob = _make_character(storage, "Bob", "c2")

        interact(alice, bob, "First meeting", rounds=1)

        # Both characters should have buffer memories
        assert alice.memory.count(tier="buffer") > 0
        assert bob.memory.count(tier="buffer") > 0
        storage.close()

    def test_relationships_created(self):
        storage = SQLiteStorage(":memory:")
        alice = _make_character(storage, "Alice", "c1")
        bob = _make_character(storage, "Bob", "c2")

        interact(alice, bob, "First meeting", rounds=1)

        # Both should have relationships with each other
        rel_a = alice.relationships.get("c2")
        rel_b = bob.relationships.get("c1")
        assert rel_a is not None
        assert rel_b is not None
        storage.close()


class TestGroupInteraction:
    def test_basic_group(self):
        storage = SQLiteStorage(":memory:")
        alice = _make_character(storage, "Alice", "c1")
        bob = _make_character(storage, "Bob", "c2")
        carol = _make_character(storage, "Carol", "c3")

        results = group_interaction([alice, bob, carol], "A meeting", rounds=1)

        assert len(results) == 1
        assert len(results[0].turns) == 3  # one per character
        speakers = [t.speaker for t in results[0].turns]
        assert speakers == ["Alice", "Bob", "Carol"]
        storage.close()

    def test_multiple_rounds(self):
        storage = SQLiteStorage(":memory:")
        alice = _make_character(storage, "Alice", "c1")
        bob = _make_character(storage, "Bob", "c2")

        results = group_interaction([alice, bob], "Ongoing scene", rounds=2)

        assert len(results) == 2
        assert len(results[0].turns) == 2
        assert len(results[1].turns) == 2
        storage.close()
