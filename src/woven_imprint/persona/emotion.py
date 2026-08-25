"""Emotional state — mood that persists and influences character responses."""

from __future__ import annotations

from dataclasses import dataclass

from ..llm.base import LLMProvider
from ..prompts import render


# Emotion dimensions based on the PAD model (Pleasure-Arousal-Dominance)
# Simplified to named emotional states for readability
EMOTION_LABELS = {
    # (pleasure, arousal, dominance) ranges → label
    "joyful": (0.6, 0.3, 0.3),
    "content": (0.4, -0.2, 0.2),
    "excited": (0.5, 0.7, 0.3),
    "anxious": (-0.2, 0.6, -0.3),
    "angry": (-0.5, 0.6, 0.5),
    "sad": (-0.5, -0.3, -0.4),
    "fearful": (-0.4, 0.5, -0.6),
    "disgusted": (-0.6, 0.2, 0.3),
    "surprised": (0.1, 0.7, -0.1),
    "neutral": (0.0, 0.0, 0.0),
    "contemplative": (0.1, -0.3, 0.1),
    "melancholic": (-0.3, -0.4, -0.2),
    "determined": (0.2, 0.4, 0.6),
    "vulnerable": (-0.1, 0.2, -0.5),
    "amused": (0.5, 0.3, 0.2),
}


@dataclass
class EmotionalState:
    """Current emotional state of a character."""

    mood: str = "neutral"
    intensity: float = 0.5  # 0.0 = barely noticeable, 1.0 = overwhelming
    cause: str = ""  # what triggered this mood
    turns_held: int = 0  # how many turns this mood has persisted

    def decay(self, rate: float | None = None) -> None:
        """Emotions naturally decay toward neutral over time."""
        if rate is None:
            from ..config import get_config

            rate = get_config().persona.emotion_decay_rate
        self.intensity = max(0.0, self.intensity - rate)
        self.turns_held += 1
        if self.intensity < 0.1:
            self.mood = "neutral"
            from ..config import get_config

            self.intensity = get_config().persona.emotion_neutral_intensity
            self.cause = ""
            self.turns_held = 0

    def to_dict(self) -> dict:
        return {
            "mood": self.mood,
            "intensity": self.intensity,
            "cause": self.cause,
            "turns_held": self.turns_held,
        }

    @classmethod
    def from_dict(cls, data: dict) -> EmotionalState:
        return cls(
            mood=data.get("mood", "neutral"),
            intensity=data.get("intensity", 0.5),
            cause=data.get("cause", ""),
            turns_held=data.get("turns_held", 0),
        )

    def describe(self) -> str:
        """Natural language description for prompt injection."""
        if self.mood == "neutral" or self.intensity < 0.2:
            return ""

        intensity_word = (
            "slightly"
            if self.intensity < 0.4
            else ("quite" if self.intensity < 0.7 else "intensely")
        )
        desc = f"You are currently feeling {intensity_word} {self.mood}."
        if self.cause:
            desc += f" This is because: {self.cause}."
        if self.turns_held > 3:
            desc += " This feeling has been lingering for a while."
        return desc


class EmotionEngine:
    """Assess and update emotional state from conversation content."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    @staticmethod
    def parse_assessment(result: dict, current: EmotionalState) -> EmotionalState:
        if not isinstance(result, dict):
            result = {}
        mood = str(result.get("mood", "neutral")).lower().strip()
        if mood not in EMOTION_LABELS:
            mood = "neutral"
        intensity = max(0.0, min(1.0, float(result.get("intensity", 0.3))))
        cause = str(result.get("cause", ""))[:200]
        return EmotionalState(mood=mood, intensity=intensity, cause=cause, turns_held=0)

    def assess(
        self, message: str, response: str, current: EmotionalState, character_name: str
    ) -> EmotionalState:
        """Assess how an exchange affects the character's emotional state."""
        messages = render(
            "emotion_turn",
            character_name=character_name,
            mood=current.mood,
            intensity=current.intensity,
            message=message[:300],
            response=response[:300],
        )

        try:
            result = self.llm.generate_json_robust(messages)
            return self.parse_assessment(result, current)
        except (ValueError, KeyError, TypeError):
            # On failure, still decay current state (graceful degradation
            # of the emotion itself), but propagate the error so the
            # caller can track subsystem health (Character.health())
            # instead of the failure going silently unnoticed.
            current.decay()
            raise

    def assess_event(
        self, event: str, current: EmotionalState, character_name: str
    ) -> EmotionalState:
        """Assess emotional impact of a world event (no dialogue pair)."""
        messages = render(
            "emotion_event",
            character_name=character_name,
            valid_moods=", ".join(EMOTION_LABELS),
            mood=current.mood,
            intensity=current.intensity,
            event=event,
        )
        try:
            result = self.llm.generate_json_robust(messages)
            if not isinstance(result, dict):
                result = {}
            mood = result.get("mood", current.mood)
            if mood not in EMOTION_LABELS:
                mood = "neutral"
            intensity = result.get("intensity", current.intensity)
            new = EmotionalState(
                mood=mood,
                intensity=max(0.0, min(1.0, float(intensity))),
                cause=str(result.get("cause", ""))[:200],
                turns_held=0,
            )
            return new
        except (ValueError, KeyError, TypeError):
            # Same degrade-then-raise pattern as assess(): decay current
            # state for graceful degradation, but propagate the error so
            # Character.health() can see the failure instead of it being
            # silently swallowed.
            current.decay()
            raise
