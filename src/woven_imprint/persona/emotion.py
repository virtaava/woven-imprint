"""Emotional state — mood that persists and influences character responses."""

from __future__ import annotations

from dataclasses import dataclass

from ..llm.base import LLMProvider


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

    def decay(self, rate: float = 0.15) -> None:
        """Emotions naturally decay toward neutral over time."""
        self.intensity = max(0.0, self.intensity - rate)
        self.turns_held += 1
        if self.intensity < 0.1:
            self.mood = "neutral"
            self.intensity = 0.3
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

    def assess(
        self, message: str, response: str, current: EmotionalState, character_name: str
    ) -> EmotionalState:
        """Assess how an exchange affects the character's emotional state."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You assess how a conversation affects a character's emotional state. "
                    "Return JSON with:\n"
                    "- mood: one of: joyful, content, excited, anxious, angry, sad, fearful, "
                    "  disgusted, surprised, neutral, contemplative, melancholic, determined, "
                    "  vulnerable, amused\n"
                    "- intensity: float 0.0-1.0 (how strongly they feel this)\n"
                    "- cause: brief reason for the mood (1 sentence)\n\n"
                    "Be realistic. Most conversations produce mild emotions (0.2-0.5). "
                    "Only dramatic events warrant high intensity."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Character: {character_name}\n"
                    f"Current mood: {current.mood} (intensity {current.intensity:.1f})\n\n"
                    f"Someone said: {message[:300]}\n"
                    f"{character_name} responded: {response[:300]}\n\n"
                    f"What is {character_name}'s emotional state now? Return JSON."
                ),
            },
        ]

        try:
            result = self.llm.generate_json(messages)
            mood = str(result.get("mood", "neutral")).lower().strip()
            if mood not in EMOTION_LABELS:
                mood = "neutral"
            intensity = max(0.0, min(1.0, float(result.get("intensity", 0.3))))
            cause = str(result.get("cause", ""))[:200]

            return EmotionalState(
                mood=mood,
                intensity=intensity,
                cause=cause,
                turns_held=0,
            )
        except (ValueError, KeyError, TypeError):
            # On failure, decay current state
            current.decay()
            return current
