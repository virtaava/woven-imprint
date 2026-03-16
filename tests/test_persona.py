"""Tests for persona model."""

from datetime import date
from unittest.mock import patch

from woven_imprint.persona.model import PersonaModel


class TestPersonaModel:
    def test_basic_construction(self):
        p = PersonaModel({
            "name": "Alice",
            "hard": {"name": "Alice", "species": "human"},
            "soft": {"personality": "witty and sharp"},
            "backstory": "Former detective",
        })
        assert p.name == "Alice"
        assert p.backstory == "Former detective"
        assert p.soft["personality"] == "witty and sharp"

    def test_age_from_birthdate(self):
        # Use a fixed date to avoid test flakiness
        p = PersonaModel({"name": "Alice"}, birthdate="2000-06-15")
        age = p.age
        assert age is not None
        assert age >= 25  # Will be true from 2025 onwards

    def test_age_without_birthdate(self):
        p = PersonaModel({"name": "Alice", "hard": {"age": 30}})
        assert p.age == 30

    def test_birthday_detection(self):
        today = date.today()
        bday = f"2000-{today.month:02d}-{today.day:02d}"
        p = PersonaModel({"name": "Alice"}, birthdate=bday)
        assert p.is_birthday is True
        assert p.days_until_birthday == 0

    def test_not_birthday(self):
        # Use a date that's definitely not today
        p = PersonaModel({"name": "Alice"}, birthdate="2000-01-01")
        if date.today().month != 1 or date.today().day != 1:
            assert p.is_birthday is False

    def test_system_prompt_contains_persona(self):
        p = PersonaModel({
            "name": "Alice",
            "hard": {"name": "Alice"},
            "soft": {"personality": "witty", "speaking_style": "clipped sentences"},
            "backstory": "A detective in London",
        }, birthdate="2000-03-15")

        prompt = p.build_system_prompt()
        assert "Alice" in prompt
        assert "witty" in prompt
        assert "clipped sentences" in prompt
        assert "A detective in London" in prompt

    def test_hard_facts(self):
        p = PersonaModel({
            "name": "Alice",
            "hard": {"name": "Alice", "species": "human", "birthplace": "London"},
            "backstory": "Born in London",
        })
        facts = p.get_hard_facts()
        assert any("Alice" in f for f in facts)
        assert any("London" in f for f in facts)

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
