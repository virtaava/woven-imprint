"""Tests for persona model."""

from datetime import datetime, timezone

from woven_imprint import clock
from woven_imprint.persona.model import PersonaModel

# Fixed reference date for all birthday/age assertions below, so these tests
# are deterministic regardless of wall-clock time or timezone (PersonaModel
# derives age/birthday from woven_imprint.clock, which is UTC).
FIXED_NOW = datetime(2026, 6, 21, 12, 0, tzinfo=timezone.utc)


class TestPersonaModel:
    def test_basic_construction(self):
        p = PersonaModel(
            {
                "name": "Alice",
                "hard": {"name": "Alice", "species": "human"},
                "soft": {"personality": "witty and sharp"},
                "backstory": "Former detective",
            }
        )
        assert p.name == "Alice"
        assert p.backstory == "Former detective"
        assert p.soft["personality"] == "witty and sharp"

    def test_age_from_birthdate(self):
        with clock.override(FIXED_NOW):
            p = PersonaModel({"name": "Alice"}, birthdate="2000-06-15")
            assert p.age == 26  # FIXED_NOW is 2026-06-21, birthday already passed this year

    def test_age_without_birthdate(self):
        p = PersonaModel({"name": "Alice", "hard": {"age": 30}})
        assert p.age == 30

    def test_birthday_detection(self):
        with clock.override(FIXED_NOW):
            today = clock.today()
            bday = f"2000-{today.month:02d}-{today.day:02d}"
            p = PersonaModel({"name": "Alice"}, birthdate=bday)
            assert p.is_birthday is True
            assert p.days_until_birthday == 0

    def test_not_birthday(self):
        # FIXED_NOW is 2026-06-21, so 2000-01-01 is unconditionally not today.
        with clock.override(FIXED_NOW):
            p = PersonaModel({"name": "Alice"}, birthdate="2000-01-01")
            assert p.is_birthday is False

    def test_system_prompt_contains_persona(self):
        p = PersonaModel(
            {
                "name": "Alice",
                "hard": {"name": "Alice"},
                "soft": {"personality": "witty", "speaking_style": "clipped sentences"},
                "backstory": "A detective in London",
            },
            birthdate="2000-03-15",
        )

        prompt = p.build_system_prompt()
        assert "Alice" in prompt
        assert "witty" in prompt
        assert "clipped sentences" in prompt
        assert "A detective in London" in prompt

    def test_hard_facts(self):
        p = PersonaModel(
            {
                "name": "Alice",
                "hard": {"name": "Alice", "species": "human", "birthplace": "London"},
                "backstory": "Born in London",
            }
        )
        facts = p.get_hard_facts()
        assert any("Alice" in f for f in facts)
        assert any("London" in f for f in facts)

    def test_prompt_excludes_creator_notes_greetings_tags_but_keeps_scenario(self):
        p = PersonaModel(
            {
                "name": "Alice",
                "hard": {"name": "Alice", "creator_notes": "SEKRET_CREATOR_NOTE"},
                "soft": {
                    "personality": "witty",
                    "greetings": ["SEKRET_GREETING_TEXT"],
                    "tags": ["SEKRET_TAG"],
                    "scenario": "A quiet cafe in autumn",
                },
            }
        )

        prompt = p.build_system_prompt()
        assert "SEKRET_CREATOR_NOTE" not in prompt
        assert "SEKRET_GREETING_TEXT" not in prompt
        assert "SEKRET_TAG" not in prompt
        assert "A quiet cafe in autumn" in prompt  # scenario still renders

        facts = p.get_hard_facts()
        assert not any("SEKRET_CREATOR_NOTE" in f for f in facts)

        # Excluded keys are still stored, so export_card can round-trip them.
        d = p.to_dict()
        assert d["hard"]["creator_notes"] == "SEKRET_CREATOR_NOTE"
        assert d["soft"]["greetings"] == ["SEKRET_GREETING_TEXT"]
        assert d["soft"]["tags"] == ["SEKRET_TAG"]

    def test_update_soft(self):
        p = PersonaModel({"name": "Alice", "soft": {"mood": "happy"}})
        p.update_soft("mood", "contemplative")
        assert p.soft["mood"] == "contemplative"

    def test_to_dict_roundtrip(self):
        original = {
            "name": "Alice",
            "hard": {"name": "Alice"},
            "soft": {"personality": "witty"},
            "temporal": {"location": "London"},
            "backstory": "A detective",
        }
        p = PersonaModel(original)
        d = p.to_dict()
        assert d["name"] == "Alice"
        assert d["soft"]["personality"] == "witty"
        assert d["temporal"]["location"] == "London"
