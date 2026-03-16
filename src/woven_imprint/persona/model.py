"""Persona model — manages character identity and constraint levels."""

from __future__ import annotations

from datetime import date


class PersonaModel:
    """Manages a character's persona with four constraint levels.

    Levels:
        hard:     Immutable identity (name, core backstory)
        temporal: Time/event-driven facts (age from birthdate, location)
        soft:     Personality traits that evolve slowly
        emergent: Formed through interaction (not stored here)
    """

    def __init__(self, persona: dict, birthdate: str | None = None):
        self.hard: dict = persona.get("hard", {})
        self.temporal: dict = persona.get("temporal", {})
        self.soft: dict = persona.get("soft", {})
        self.birthdate: date | None = None

        if birthdate:
            self.birthdate = date.fromisoformat(birthdate)

        # Core identity fields (always in hard)
        self.name: str = persona.get("name", self.hard.get("name", "Unknown"))
        self.backstory: str = persona.get("backstory", self.hard.get("backstory", ""))

    @property
    def age(self) -> int | None:
        """Derive age from birthdate. Returns None if no birthdate set."""
        if not self.birthdate:
            return self.hard.get("age") or self.temporal.get("age")
        today = date.today()
        age = today.year - self.birthdate.year
        if (today.month, today.day) < (self.birthdate.month, self.birthdate.day):
            age -= 1
        return age

    @property
    def is_birthday(self) -> bool:
        """Check if today is the character's birthday."""
        if not self.birthdate:
            return False
        today = date.today()
        return today.month == self.birthdate.month and today.day == self.birthdate.day

    @property
    def days_until_birthday(self) -> int | None:
        """Days until next birthday. 0 if today."""
        if not self.birthdate:
            return None
        today = date.today()
        next_bday = self.birthdate.replace(year=today.year)
        if next_bday < today:
            next_bday = next_bday.replace(year=today.year + 1)
        return (next_bday - today).days

    def build_system_prompt(self) -> str:
        """Build the character's system prompt from all constraint levels."""
        parts = []

        parts.append(f"You ARE {self.name}. Stay in character at all times.")

        # Hard constraints
        if self.backstory:
            parts.append(f"Backstory: {self.backstory}")
        for key, val in self.hard.items():
            if key not in ("name", "backstory"):
                parts.append(f"{key.replace('_', ' ').title()}: {val}")

        # Temporal facts
        if self.age is not None:
            parts.append(f"Age: {self.age}")
            if self.is_birthday:
                parts.append("Today is your birthday!")
        for key, val in self.temporal.items():
            if key != "age":
                parts.append(f"{key.replace('_', ' ').title()}: {val}")

        # Soft constraints
        if self.soft:
            personality = self.soft.get("personality", "")
            if personality:
                parts.append(f"Personality: {personality}")
            speaking_style = self.soft.get("speaking_style", "")
            if speaking_style:
                parts.append(f"Speaking style: {speaking_style}")
            for key, val in self.soft.items():
                if key not in ("personality", "speaking_style"):
                    parts.append(f"{key.replace('_', ' ').title()}: {val}")

        return "\n".join(parts)

    def get_hard_facts(self) -> list[str]:
        """Return all hard constraint facts as natural language strings."""
        facts = [f"Name is {self.name}"]
        if self.backstory:
            facts.append(f"Backstory: {self.backstory}")
        for key, val in self.hard.items():
            if key not in ("name", "backstory"):
                facts.append(f"{key}: {val}")
        return facts

    def update_soft(self, key: str, value: str) -> None:
        """Update a soft constraint (character growth)."""
        self.soft[key] = value

    def update_temporal(self, key: str, value: str) -> None:
        """Update a temporal fact (event-driven change)."""
        self.temporal[key] = value

    def to_dict(self) -> dict:
        """Serialize persona to dict for storage."""
        return {
            "name": self.name,
            "backstory": self.backstory,
            "hard": self.hard,
            "temporal": self.temporal,
            "soft": self.soft,
        }
