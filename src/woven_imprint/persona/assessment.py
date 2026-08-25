"""One LLM call per turn for all bookkeeping: emotion, relationship deltas, story beat, facts."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..llm.base import LLMProvider
from ..narrative.arc import ArcTracker, NarrativeArc, StoryBeat
from ..prompts import (
    SOMEONE_FALLBACK,
    TURN_ASSESSMENT_BEAT_SECTION,
    TURN_ASSESSMENT_EMOTION_SECTION,
    TURN_ASSESSMENT_FACTS_SECTION,
    TURN_ASSESSMENT_NONE_TMPL,
    TURN_ASSESSMENT_RECENT_TMPL,
    TURN_ASSESSMENT_RELATIONSHIP_SECTION,
    render,
)
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
        sections = [TURN_ASSESSMENT_EMOTION_SECTION.format(moods=", ".join(EMOTION_LABELS))]
        if want_relationship:
            sections.append(TURN_ASSESSMENT_RELATIONSHIP_SECTION.format())
        if want_beat:
            sections.append(TURN_ASSESSMENT_BEAT_SECTION.format())
        if want_facts:
            sections.append(TURN_ASSESSMENT_FACTS_SECTION.format(max_facts=max_facts))
        sections_text = "\n- ".join(sections)

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
            beats_part = (
                TURN_ASSESSMENT_RECENT_TMPL.format(recent=recent)
                if recent
                else TURN_ASSESSMENT_NONE_TMPL
            )
            arc_line = (
                f"Current arc phase: {arc.current_phase.value}; tension {arc.tension:.1f}\n"
                + beats_part
            )

        return render(
            "turn_assessment",
            character_name=character_name,
            sections=sections_text,
            mood=current_emotion.mood,
            intensity=current_emotion.intensity,
            rel_line=rel_line,
            arc_line=arc_line,
            other_display=other_name or SOMEONE_FALLBACK,
            message=message[:300],
            response=response[:300],
            context_hint=context_hint,
        )

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
