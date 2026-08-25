"""One LLM call per turn for all bookkeeping: emotion, relationship deltas, story beat, facts."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..llm.base import LLMProvider
from ..narrative.arc import ArcTracker, NarrativeArc, StoryBeat
from .emotion import EMOTION_LABELS, EmotionEngine, EmotionalState


@dataclass
class TurnAssessment:
    emotion: EmotionalState | None
    relationship: dict[str, float] | None
    beat: StoryBeat | None
    facts: list[str]
    raw: dict = field(default_factory=dict)


class TurnAssessor:
    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def build_messages(
        self,
        *,
        message: str,
        response: str,
        character_name: str,
        other_name: str,
        current_emotion: EmotionalState,
        arc: NarrativeArc,
        relationship: dict | None,
        want_facts: bool,
        want_beat: bool,
        want_relationship: bool,
        max_facts: int,
        context_hint: str,
    ) -> list[dict]:
        sections = [
            '"emotion": {"mood": one of '
            + ", ".join(EMOTION_LABELS)
            + ', "intensity": float 0.0-1.0, '
            '"cause": one sentence}. Be realistic: most exchanges produce mild emotions (0.2-0.5).'
        ]
        if want_relationship:
            sections.append(
                '"relationship": {"trust", "affection", "respect", "familiarity", "tension"} — each a float '
                "between -0.15 and 0.15 (familiarity 0.0 to 0.15) describing how THIS exchange shifts the "
                "relationship; 0.0 for no change. Be conservative: most exchanges are 0.01-0.05."
            )
        if want_beat:
            sections.append(
                '"beat": {"is_beat": bool, "description": one sentence, "phase": setup|rising_action|climax|'
                'falling_action|resolution|epilogue, "tension": float 0.0-1.0, "tags": [strings]} or null. '
                "Only genuine story beats (revelations, confrontations, betrayals, decisions) are beats."
            )
        if want_facts:
            sections.append(
                f'"facts": up to {max_facts} strings — specific NEW facts, opinions, preferences, biographical '
                "details or commitments worth remembering long-term, one per string; [] if nothing notable."
            )
        system = (
            f"You are the bookkeeping assistant for the character {character_name}. Assess the exchange and "
            "return ONE JSON object with exactly these keys:\n- " + "\n- ".join(sections)
        )
        rel_line = ""
        if want_relationship and relationship:
            dims = relationship.get("dimensions", {})
            rel_line = (
                f"Current relationship: {relationship.get('type', 'acquaintance')}, "
                f"trust={dims.get('trust', 0):.2f}, affection={dims.get('affection', 0):.2f}, "
                f"familiarity={dims.get('familiarity', 0):.2f}\n"
            )
        arc_line = ""
        if want_beat:
            recent = "\n".join(
                f"- [{b.phase.value}] {b.description} (tension: {b.tension:.1f})"
                for b in arc.beats[-5:]
            )
            arc_line = (
                f"Current arc phase: {arc.current_phase.value}; tension {arc.tension:.1f}\n"
                + (f"Recent beats:\n{recent}\n" if recent else "No prior beats.\n")
            )
        user = (
            f"Character: {character_name}\n"
            f"Current mood: {current_emotion.mood} (intensity {current_emotion.intensity:.1f})\n"
            f"{rel_line}{arc_line}\n"
            f"[{other_name or 'Someone'}]: {message[:300]}\n"
            f"[{character_name}]: {response[:300]}\n"
            f"{context_hint}\n\nReturn the JSON object."
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def assess(self, **kwargs) -> TurnAssessment:
        messages = self.build_messages(**kwargs)
        data = self.llm.generate_json_robust(messages)
        if not isinstance(data, dict):
            data = {}
        current = kwargs["current_emotion"]
        emotion = (
            EmotionEngine.parse_assessment(data.get("emotion") or {}, current)
            if isinstance(data.get("emotion"), dict)
            else None
        )
        rel = None
        if kwargs["want_relationship"] and isinstance(data.get("relationship"), dict):
            from ..character import Character  # local import to avoid a cycle

            parsed = Character._parse_relationship_deltas(data["relationship"])
            rel = parsed or None
        beat = None
        if kwargs["want_beat"] and isinstance(data.get("beat"), dict):
            beat = ArcTracker.parse_beat(
                data["beat"], kwargs["arc"], kwargs["character_name"], kwargs["other_name"]
            )
        facts: list[str] = []
        if kwargs["want_facts"]:
            from ..character import Character

            facts = Character._parse_facts(data.get("facts", []), kwargs["max_facts"])
        return TurnAssessment(emotion=emotion, relationship=rel, beat=beat, facts=facts, raw=data)
