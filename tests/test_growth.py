"""Tests for character growth engine."""

from woven_imprint.storage.sqlite import SQLiteStorage
from woven_imprint.persona.model import PersonaModel
from woven_imprint.persona.growth import GrowthEngine, GrowthEvent


class FakeLLM:
    def __init__(self, growth_events=None):
        self._events = growth_events or []

    def generate(self, messages, temperature=0.7, max_tokens=2048):
        return "Response"

    def generate_json(self, messages, temperature=0.3):
        return self._events


class TestGrowthEngine:
    def _setup(self, growth_events=None):
        storage = SQLiteStorage(":memory:")
        storage.save_character("c1", "Alice", {"soft": {"personality": "shy and reserved"}})
        persona = PersonaModel(
            {"name": "Alice", "soft": {"personality": "shy and reserved"}},
        )
        llm = FakeLLM(growth_events or [])
        engine = GrowthEngine(storage, llm, "c1", persona)
        return storage, persona, engine

    def test_no_growth_without_enough_memories(self):
        storage, persona, engine = self._setup()
        events = engine.detect_growth(min_memories=20)
        assert events == []
        storage.close()

    def test_detect_growth(self):
        storage, persona, engine = self._setup(
            [
                {
                    "trait": "personality",
                    "old_value": "shy and reserved",
                    "new_value": "shy but gradually opening up to trusted friends",
                    "reason": "Multiple positive interactions with Bob built confidence",
                    "confidence": 0.75,
                }
            ]
        )

        # Add enough memories
        for i in range(25):
            storage.save_memory(
                {
                    "id": f"m{i}",
                    "character_id": "c1",
                    "tier": "core",
                    "content": f"Had a positive interaction {i}",
                }
            )

        events = engine.detect_growth(min_memories=20)
        assert len(events) == 1
        assert events[0].trait == "personality"
        assert events[0].confidence == 0.75
        storage.close()

    def test_apply_growth_updates_persona(self):
        storage, persona, engine = self._setup()

        events = [
            GrowthEvent(
                trait="personality",
                old_value="shy and reserved",
                new_value="shy but more confident",
                reason="Built trust through interactions",
                confidence=0.8,
            )
        ]

        applied = engine.apply_growth(events, threshold=0.6)
        assert len(applied) == 1
        assert persona.soft["personality"] == "shy but more confident"

        # Check persisted
        char = storage.load_character("c1")
        assert char["persona"]["soft"]["personality"] == "shy but more confident"

        # Check growth memory was created
        memories = storage.get_memories("c1", tier="core")
        growth_mems = [m for m in memories if "[Growth]" in m["content"]]
        assert len(growth_mems) == 1
        storage.close()

    def test_below_threshold_not_applied(self):
        storage, persona, engine = self._setup()

        events = [
            GrowthEvent(
                trait="personality",
                old_value="shy",
                new_value="outgoing",
                reason="One conversation",
                confidence=0.3,
            )
        ]

        applied = engine.apply_growth(events, threshold=0.6)
        assert len(applied) == 0
        assert persona.soft["personality"] == "shy and reserved"  # Unchanged
        storage.close()

    def test_grow_convenience_method(self):
        storage, persona, engine = self._setup(
            [
                {
                    "trait": "personality",
                    "old_value": "shy",
                    "new_value": "more open",
                    "reason": "Repeated positive interactions",
                    "confidence": 0.7,
                }
            ]
        )

        for i in range(25):
            storage.save_memory(
                {
                    "id": f"m{i}",
                    "character_id": "c1",
                    "tier": "core",
                    "content": f"Interaction {i}",
                }
            )

        applied = engine.grow(min_memories=20, threshold=0.6)
        assert len(applied) == 1
        assert persona.soft["personality"] == "more open"
        storage.close()
