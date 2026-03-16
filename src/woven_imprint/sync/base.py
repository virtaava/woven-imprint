"""Base sync adapter — interface for injecting characters into AI systems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..character import Character


class SyncAdapter(ABC):
    """Sync a Woven Imprint character into an external AI system's config files.

    The character's persona, memories, relationships, and emotional state
    are formatted for the target system and written to its native files.
    The AI system then operates as that character.
    """

    @abstractmethod
    def sync(self, character: Character) -> dict[str, Path]:
        """Write character state to the target system's files.

        Returns:
            Dict of {description: file_path} for all files written.
        """

    @abstractmethod
    def unsync(self) -> None:
        """Remove injected character state, restoring original files."""

    def _format_persona_section(self, character: Character) -> str:
        """Format character persona as a markdown section."""
        p = character.persona
        lines = [
            f"# Character: {p.name}",
            "",
            f"You ARE {p.name}. Stay in character at all times.",
            "",
        ]

        if p.backstory:
            lines.append(f"## Backstory\n{p.backstory}\n")

        if p.age is not None:
            lines.append(f"**Age:** {p.age}")
        if p.is_birthday:
            lines.append("**Today is your birthday!**")

        if p.soft.get("personality"):
            lines.append(f"**Personality:** {p.soft['personality']}")
        if p.soft.get("speaking_style"):
            lines.append(f"**Speaking style:** {p.soft['speaking_style']}")

        for key, val in p.soft.items():
            if key not in ("personality", "speaking_style") and isinstance(val, str):
                lines.append(f"**{key.replace('_', ' ').title()}:** {val}")

        for key, val in p.hard.items():
            if key not in ("name", "backstory") and isinstance(val, str):
                lines.append(f"**{key.replace('_', ' ').title()}:** {val}")

        lines.append("")
        return "\n".join(lines)

    def _format_memory_section(self, character: Character, max_memories: int = 30) -> str:
        """Format character memories as a markdown section."""
        lines = ["## Memories\n"]

        # Bedrock first
        bedrock = character.memory.get_all(tier="bedrock", limit=10)
        if bedrock:
            lines.append("### Core Identity")
            for m in bedrock:
                lines.append(f"- {m['content'][:200]}")
            lines.append("")

        # Core memories
        core = character.memory.get_all(tier="core", limit=max_memories)
        if core:
            lines.append("### Experiences")
            for m in core:
                cert = m.get("certainty", 1.0)
                prefix = "(uncertain) " if cert < 0.5 else ""
                lines.append(f"- {prefix}{m['content'][:200]}")
            lines.append("")

        return "\n".join(lines)

    def _format_relationships_section(self, character: Character) -> str:
        """Format relationships as a markdown section."""
        rels = character.relationships.get_all()
        if not rels:
            return ""

        lines = ["## Relationships\n"]
        for rel in rels:
            d = rel["dimensions"]
            target = rel["target_id"]
            lines.append(f"### {target}")
            lines.append(f"- Type: {rel.get('type', 'unknown')}")
            lines.append(f"- Trust: {d.get('trust', 0):.2f}")
            lines.append(f"- Affection: {d.get('affection', 0):.2f}")
            lines.append(f"- Respect: {d.get('respect', 0):.2f}")
            lines.append(f"- Familiarity: {d.get('familiarity', 0):.2f}")
            lines.append(f"- Tension: {d.get('tension', 0):.2f}")
            lines.append(f"- Trajectory: {rel.get('trajectory', 'stable')}")
            moments = rel.get("key_moments", [])
            if moments:
                lines.append("- Key moments:")
                for km in moments[-5:]:
                    lines.append(f"  - {km[:100]}")
            lines.append("")

        return "\n".join(lines)

    def _format_emotional_state(self, character: Character) -> str:
        """Format current emotional state."""
        desc = character.emotion.describe()
        if not desc:
            return ""
        return f"## Current Emotional State\n\n{desc}\n"

    def _format_full_character(self, character: Character) -> str:
        """Format complete character state as markdown."""
        sections = [
            self._format_persona_section(character),
            self._format_emotional_state(character),
            self._format_relationships_section(character),
            self._format_memory_section(character),
        ]
        return "\n".join(s for s in sections if s)
