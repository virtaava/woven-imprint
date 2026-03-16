"""Engine — entry point for creating and managing characters."""

from __future__ import annotations

import json
from pathlib import Path

from .character import Character
from .embedding.base import EmbeddingProvider
from .embedding.ollama import OllamaEmbedding
from .llm.base import LLMProvider
from .llm.ollama import OllamaLLM
from .persona.model import PersonaModel
from .storage.sqlite import SQLiteStorage
from .utils.text import generate_id


class Engine:
    """Woven Imprint engine — manages characters and their persistence.

    Usage:
        engine = Engine("characters.db")
        alice = engine.create_character("Alice", persona={...})
        response = alice.chat("Hello!")
    """

    def __init__(
        self,
        db_path: str | Path = "characters.db",
        llm: LLMProvider | None = None,
        embedding: EmbeddingProvider | None = None,
    ):
        self.storage = SQLiteStorage(db_path)
        self.llm = llm or OllamaLLM()
        self.embedder = embedding or OllamaEmbedding()

    def create_character(
        self,
        name: str,
        persona: dict | None = None,
        birthdate: str | None = None,
        character_id: str | None = None,
    ) -> Character:
        """Create a new persistent character.

        Args:
            name: Character name.
            persona: Persona definition with optional keys:
                - hard: dict of immutable facts
                - soft: dict of personality traits
                - temporal: dict of time-dependent facts
                - backstory: str
                - personality: str (shorthand for soft.personality)
                - speaking_style: str (shorthand for soft.speaking_style)
            birthdate: ISO date string (YYYY-MM-DD) for age derivation.
            character_id: Optional custom ID (auto-generated if not provided).

        Returns:
            Character instance ready for chat.
        """
        char_id = character_id or generate_id("char-")
        persona = persona or {}

        # Normalize shorthand fields into constraint levels
        normalized = {
            "name": name,
            "hard": persona.get("hard", {}),
            "soft": persona.get("soft", {}),
            "temporal": persona.get("temporal", {}),
            "backstory": persona.get("backstory", ""),
        }
        normalized["hard"]["name"] = name

        # Move shorthand persona fields to soft constraints
        for key in ("personality", "speaking_style", "occupation", "appearance"):
            if key in persona and key not in normalized["soft"]:
                normalized["soft"][key] = persona[key]

        # Move backstory to hard
        if "backstory" in persona:
            normalized["hard"]["backstory"] = persona["backstory"]
            normalized["backstory"] = persona["backstory"]

        persona_model = PersonaModel(normalized, birthdate=birthdate)

        # Persist to storage
        self.storage.save_character(
            char_id,
            name,
            normalized,
            birthdate=birthdate,
        )

        # Seed bedrock memories from persona definition
        char = Character(char_id, self.storage, self.llm, self.embedder, persona_model)
        self._seed_bedrock(char, normalized)

        return char

    def load_character(self, character_id: str) -> Character | None:
        """Load an existing character from storage.

        Returns:
            Character instance, or None if not found.
        """
        data = self.storage.load_character(character_id)
        if not data:
            return None

        persona_model = PersonaModel(data["persona"], birthdate=data.get("birthdate"))
        return Character(
            data["id"],
            self.storage,
            self.llm,
            self.embedder,
            persona_model,
        )

    def get_character(self, character_id: str) -> Character:
        """Load a character, raising if not found.

        Raises:
            KeyError: If no character with this ID exists.
        """
        char = self.load_character(character_id)
        if char is None:
            raise KeyError(f"Character not found: {character_id}")
        return char

    def list_characters(self) -> list[dict]:
        """List all characters in the database."""
        return self.storage.list_characters()

    def delete_character(self, character_id: str) -> None:
        """Permanently delete a character and all their data."""
        self.storage.delete_character(character_id)

    def import_character(self, path: str | Path) -> Character:
        """Import a character from an exported JSON file.

        Returns:
            The imported Character instance.
        """
        with open(path) as f:
            data = json.load(f)

        char_id = data.get("id", generate_id("char-"))
        persona = data.get("persona", {})
        birthdate = data.get("birthdate")

        # Save character
        self.storage.save_character(
            char_id,
            persona.get("name", "Unknown"),
            persona,
            birthdate=birthdate,
        )

        persona_model = PersonaModel(persona, birthdate=birthdate)
        char = Character(char_id, self.storage, self.llm, self.embedder, persona_model)

        # Import memories (re-embed them)
        for tier_name in ("bedrock", "core", "buffer"):
            for mem_data in data.get("memories", {}).get(tier_name, []):
                char.memory.add(
                    content=mem_data["content"],
                    tier=tier_name,
                    role=mem_data.get("role"),
                    importance=mem_data.get("importance", 0.5),
                    metadata=mem_data.get("metadata", {}),
                )

        # Import relationships
        for rel_data in data.get("relationships", []):
            self.storage.save_relationship(rel_data)

        return char

    def close(self) -> None:
        """Close the database connection."""
        self.storage.close()

    def _seed_bedrock(self, char: Character, persona: dict) -> None:
        """Seed bedrock memories from the persona definition."""
        seeds = []

        name = persona.get("name", "")
        backstory = persona.get("backstory", "")
        if backstory:
            seeds.append(f"My name is {name}. {backstory}")

        # Hard facts
        for key, val in persona.get("hard", {}).items():
            if key not in ("name", "backstory") and isinstance(val, str):
                seeds.append(f"[Fact] {key}: {val}")

        # Soft traits as self-knowledge
        soft = persona.get("soft", {})
        if soft.get("personality"):
            seeds.append(f"[Self] My personality: {soft['personality']}")
        if soft.get("speaking_style"):
            seeds.append(f"[Self] How I speak: {soft['speaking_style']}")

        for seed in seeds:
            char.memory.add(
                content=seed,
                tier="bedrock",
                role="observation",
                importance=0.9,
            )
