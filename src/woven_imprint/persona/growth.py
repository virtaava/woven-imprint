"""Character growth — soft constraints evolve through accumulated experience."""

from __future__ import annotations

from dataclasses import dataclass

from ..llm.base import LLMProvider
from ..storage.sqlite import SQLiteStorage
from .model import PersonaModel


@dataclass
class GrowthEvent:
    """A detected change in a character's soft constraints."""

    trait: str
    old_value: str
    new_value: str
    reason: str
    confidence: float = 0.0


class GrowthEngine:
    """Detect and apply character growth from accumulated memories.

    Periodically analyzes a character's recent experiences and reflections
    to determine if soft constraints (personality, opinions, habits) should
    evolve. Hard constraints never change. Growth is gradual and justified.
    """

    def __init__(
        self,
        storage: SQLiteStorage,
        llm: LLMProvider,
        character_id: str,
        persona: PersonaModel,
    ):
        self.storage = storage
        self.llm = llm
        self.character_id = character_id
        self.persona = persona

    def detect_growth(self, min_memories: int = 20) -> list[GrowthEvent]:
        """Analyze recent experiences for potential character growth.

        Only runs if enough core memories have accumulated.

        Returns:
            List of detected growth events (not yet applied).
        """
        core = self.storage.get_memories(self.character_id, tier="core", limit=50)
        if len(core) < min_memories:
            return []

        # Build context from recent core memories
        memory_text = "\n".join(f"- {m['content'][:200]}" for m in core[:30])

        # Current soft constraints
        soft_text = ""
        for key, val in self.persona.soft.items():
            soft_text += f"- {key}: {val}\n"
        if not soft_text:
            soft_text = "No soft traits defined."

        messages = [
            {
                "role": "system",
                "content": (
                    "You analyze a character's accumulated experiences to detect "
                    "genuine personality growth or change. Characters evolve slowly "
                    "through meaningful experiences.\n\n"
                    "Rules:\n"
                    "- Only detect changes supported by multiple memories\n"
                    "- Changes must be gradual, not sudden reversals\n"
                    "- A character becoming 'slightly more trusting' is realistic; "
                    "  'completely changed personality' is not\n"
                    "- Return JSON array of growth events, or empty array []\n"
                    "- Each event: {trait, old_value, new_value, reason, confidence}\n"
                    "- confidence: 0.0-1.0 (how strongly the evidence supports this change)"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"CHARACTER'S CURRENT TRAITS:\n{soft_text}\n\n"
                    f"RECENT EXPERIENCES:\n{memory_text}\n\n"
                    f"Based on these experiences, has this character grown or changed "
                    f"in any way? Return JSON array of growth events."
                ),
            },
        ]

        try:
            result = self.llm.generate_json(messages)
            events_raw = result if isinstance(result, list) else result.get("events", [])

            events = []
            for e in events_raw:
                if not isinstance(e, dict):
                    continue
                events.append(
                    GrowthEvent(
                        trait=e.get("trait", ""),
                        old_value=e.get("old_value", ""),
                        new_value=e.get("new_value", ""),
                        reason=e.get("reason", ""),
                        confidence=float(e.get("confidence", 0.0)),
                    )
                )
            return events
        except (ValueError, KeyError, TypeError):
            return []

    def apply_growth(self, events: list[GrowthEvent], threshold: float = 0.6) -> list[GrowthEvent]:
        """Apply detected growth events that meet the confidence threshold.

        Updates the persona's soft constraints and records the changes
        as core memories.

        Args:
            events: Growth events from detect_growth().
            threshold: Minimum confidence to apply (default 0.6).

        Returns:
            List of events that were actually applied.
        """
        applied = []

        for event in events:
            if event.confidence < threshold:
                continue
            if not event.trait or not event.new_value:
                continue

            # Update soft constraint
            self.persona.update_soft(event.trait, event.new_value)

            # Record growth as a core memory
            from ..utils.text import generate_id

            growth_memory = {
                "id": generate_id("mem-"),
                "character_id": self.character_id,
                "tier": "core",
                "content": (
                    f"[Growth] My {event.trait} has shifted: "
                    f"'{event.old_value}' → '{event.new_value}'. "
                    f"Reason: {event.reason}"
                ),
                "importance": 0.8,
                "certainty": event.confidence,
                "status": "active",
                "source_refs": [],
                "role": "observation",
                "metadata": {
                    "type": "growth",
                    "trait": event.trait,
                    "old": event.old_value,
                    "new": event.new_value,
                },
            }
            self.storage.save_memory(growth_memory)

            # Update persisted character persona
            char_data = self.storage.load_character(self.character_id)
            if char_data:
                persona_dict = char_data["persona"]
                if "soft" not in persona_dict:
                    persona_dict["soft"] = {}
                persona_dict["soft"][event.trait] = event.new_value
                self.storage.save_character(
                    self.character_id,
                    char_data["name"],
                    persona_dict,
                    birthdate=char_data.get("birthdate"),
                )

            applied.append(event)

        return applied

    def grow(self, min_memories: int = 20, threshold: float = 0.6) -> list[GrowthEvent]:
        """Detect and apply growth in one step.

        Convenience method that calls detect_growth() then apply_growth().
        """
        events = self.detect_growth(min_memories=min_memories)
        if not events:
            return []
        return self.apply_growth(events, threshold=threshold)
