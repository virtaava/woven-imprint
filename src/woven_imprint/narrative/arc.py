"""Narrative arc awareness — track story beats and tension curves.

Characters exist within stories. This module tracks where the character
is in a narrative arc (setup, rising action, climax, resolution) and
adjusts their behavior accordingly. A character at a story's climax
should feel different urgency than one in the quiet aftermath.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..llm.base import LLMProvider
from ..prompts import ARC_BEAT_NONE_TMPL, ARC_BEAT_RECENT_TMPL, SOMEONE_FALLBACK, render


class ArcPhase(str, Enum):
    """Phases of a narrative arc."""

    SETUP = "setup"
    RISING = "rising_action"
    CLIMAX = "climax"
    FALLING = "falling_action"
    RESOLUTION = "resolution"
    EPILOGUE = "epilogue"


@dataclass
class StoryBeat:
    """A significant moment in the narrative."""

    description: str
    phase: ArcPhase
    tension: float  # 0.0 - 1.0
    turn_number: int
    characters_involved: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "phase": self.phase.value,
            "tension": self.tension,
            "turn_number": self.turn_number,
            "characters_involved": self.characters_involved,
            "tags": self.tags,
        }


@dataclass
class NarrativeArc:
    """Tracks the overall narrative arc across interactions."""

    title: str = ""
    current_phase: ArcPhase = ArcPhase.SETUP
    tension: float = 0.2  # Current tension level
    beats: list[StoryBeat] = field(default_factory=list)
    turn_count: int = 0

    @property
    def tension_trend(self) -> str:
        """Is tension rising, falling, or stable?"""
        if len(self.beats) < 2:
            return "stable"
        recent = [b.tension for b in self.beats[-3:]]
        if len(recent) < 2:
            return "stable"
        delta = recent[-1] - recent[0]
        if delta > 0.1:
            return "rising"
        elif delta < -0.1:
            return "falling"
        return "stable"

    def describe(self) -> str:
        """Natural language description for prompt injection."""
        parts = []

        phase_descriptions = {
            ArcPhase.SETUP: "The story is in its early stages. Things are being established.",
            ArcPhase.RISING: "Events are escalating. Stakes are rising. Conflict is building.",
            ArcPhase.CLIMAX: "This is a critical moment. Everything is coming to a head.",
            ArcPhase.FALLING: "The major conflict has passed. Consequences are unfolding.",
            ArcPhase.RESOLUTION: "Things are settling. Relationships and situations are finding their new normal.",
            ArcPhase.EPILOGUE: "The story has concluded. This is the aftermath.",
        }

        parts.append(phase_descriptions.get(self.current_phase, ""))

        if self.tension > 0.7:
            parts.append("Tension is very high right now.")
        elif self.tension > 0.4:
            parts.append("There is moderate tension in the air.")
        elif self.tension < 0.15:
            parts.append("The atmosphere is calm and relaxed.")

        trend = self.tension_trend
        if trend == "rising":
            parts.append("Things have been getting more intense.")
        elif trend == "falling":
            parts.append("Things have been calming down.")

        return " ".join(p for p in parts if p)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "current_phase": self.current_phase.value,
            "tension": self.tension,
            "tension_trend": self.tension_trend,
            "turn_count": self.turn_count,
            "beats": [b.to_dict() for b in self.beats],
        }

    @classmethod
    def from_dict(cls, data: dict) -> NarrativeArc:
        arc = cls(
            title=data.get("title", ""),
            current_phase=ArcPhase(data.get("current_phase", "setup")),
            tension=data.get("tension", 0.2),
            turn_count=data.get("turn_count", 0),
        )
        for b in data.get("beats", []):
            arc.beats.append(
                StoryBeat(
                    description=b["description"],
                    phase=ArcPhase(b["phase"]),
                    tension=b["tension"],
                    turn_number=b["turn_number"],
                    characters_involved=b.get("characters_involved", []),
                    tags=b.get("tags", []),
                )
            )
        return arc


class ArcTracker:
    """Analyze interactions to detect story beats and update the narrative arc."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    @staticmethod
    def should_analyze(arc: NarrativeArc) -> bool:
        """Increment the turn counter and apply the every-other-turn rule."""
        arc.turn_count += 1
        return not (arc.turn_count % 2 != 0 and arc.turn_count > 1)

    @staticmethod
    def parse_beat(
        result: dict, arc: NarrativeArc, character_name: str, other_name: str = ""
    ) -> StoryBeat | None:
        if not isinstance(result, dict) or not result.get("is_beat", False):
            return None
        phase_str = result.get("phase", arc.current_phase.value)
        try:
            phase = ArcPhase(phase_str)
        except ValueError:
            phase = arc.current_phase
        tension = max(0.0, min(1.0, float(result.get("tension", arc.tension))))
        tags = result.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        beat = StoryBeat(
            description=str(result.get("description", ""))[:300],
            phase=phase,
            tension=tension,
            turn_number=arc.turn_count,
            characters_involved=[character_name] + ([other_name] if other_name else []),
            tags=[str(t) for t in tags[:5]],
        )
        arc.current_phase = phase
        arc.tension = tension
        arc.beats.append(beat)
        return beat

    def analyze_beat(
        self,
        message: str,
        response: str,
        arc: NarrativeArc,
        character_name: str,
        other_name: str = "",
    ) -> StoryBeat | None:
        """Analyze an exchange for narrative significance.

        Not every exchange is a story beat. Returns None if the exchange
        is routine conversation without narrative significance.
        """
        if not self.should_analyze(arc):
            return None

        recent_beats = ""
        if arc.beats:
            recent_beats = "\n".join(
                f"- [{b.phase.value}] {b.description} (tension: {b.tension:.1f})"
                for b in arc.beats[-5:]
            )

        beats_line = (
            ARC_BEAT_RECENT_TMPL.format(recent=recent_beats) if recent_beats else ARC_BEAT_NONE_TMPL
        )
        other_display = other_name or SOMEONE_FALLBACK
        messages = render(
            "arc_beat",
            phase=arc.current_phase.value,
            tension=arc.tension,
            beats_line=beats_line,
            other_display=other_display,
            message=message[:300],
            character_name=character_name,
            response=response[:300],
        )

        try:
            result = self.llm.generate_json_robust(messages)
            return self.parse_beat(result, arc, character_name, other_name)
        except (ValueError, KeyError, TypeError):
            # Propagate so the caller can track subsystem health
            # (Character.health()) instead of a failure going silently
            # unnoticed as "not a beat."
            raise
