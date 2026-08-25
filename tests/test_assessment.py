"""Tests for TurnAssessor — single-call bookkeeping, and the pure parsers it delegates to."""

from woven_imprint.llm.base import LLMProvider
from woven_imprint.narrative.arc import ArcPhase, NarrativeArc
from woven_imprint.persona.assessment import TurnAssessor
from woven_imprint.persona.emotion import EmotionalState


class ScriptedLLM(LLMProvider):
    def __init__(self, payload):
        self.payload, self.calls = payload, 0

    def generate(self, messages, temperature=0.7, max_tokens=2048):
        return "x"

    def generate_json(self, messages, temperature=0.3):
        self.calls += 1
        return self.payload

    def generate_json_robust(self, messages, temperature=0.3):
        return self.generate_json(messages, temperature)


FULL = {
    "emotion": {"mood": "amused", "intensity": 0.4, "cause": "a joke"},
    "relationship": {
        "trust": 0.05,
        "affection": 0.02,
        "respect": 0.0,
        "familiarity": 0.03,
        "tension": -0.01,
    },
    "beat": {
        "is_beat": True,
        "description": "A confession.",
        "phase": "rising_action",
        "tension": 0.6,
        "tags": ["revelation"],
    },
    "facts": ["The visitor's sister is called Aino.", "short"],
}


def _kwargs(llm_arc):
    return dict(
        message="hi",
        response="hello",
        character_name="Ada",
        other_name="Toni",
        current_emotion=EmotionalState(),
        arc=llm_arc,
        relationship={"type": "acquaintance", "dimensions": {}},
        want_facts=True,
        want_beat=True,
        want_relationship=True,
        max_facts=5,
        context_hint="",
    )


def test_single_call_parses_all_sections():
    llm = ScriptedLLM(FULL)
    arc = NarrativeArc()
    arc.turn_count = 1  # so the every-2nd-turn rule allows a beat on the next turn
    out = TurnAssessor(llm).assess(**_kwargs(arc))
    assert llm.calls == 1
    assert out.emotion.mood == "amused" and abs(out.emotion.intensity - 0.4) < 1e-9
    assert out.relationship == FULL["relationship"]
    assert out.beat is not None and out.beat.phase == ArcPhase.RISING and arc.tension == 0.6
    assert out.facts == ["The visitor's sister is called Aino."]  # len > 10 filter


def test_missing_sections_are_none_or_empty():
    llm = ScriptedLLM({"emotion": {"mood": "weird-label", "intensity": 3}})
    out = TurnAssessor(llm).assess(**_kwargs(NarrativeArc()))
    assert out.emotion.mood == "neutral" and out.emotion.intensity == 1.0
    assert out.relationship is None and out.beat is None and out.facts == []


def test_prompt_omits_unwanted_sections():
    llm = ScriptedLLM(FULL)
    kw = _kwargs(NarrativeArc())
    kw.update(want_facts=False, want_beat=False, want_relationship=False)
    msgs = TurnAssessor(llm).build_messages(**kw)
    system = msgs[0]["content"]
    assert '"facts"' not in system and '"beat"' not in system and '"relationship"' not in system
    assert '"emotion"' in system


def test_engine_parsers_match_legacy_behavior():
    from woven_imprint.character import Character
    from woven_imprint.persona.emotion import EmotionEngine

    result = EmotionEngine.parse_assessment(
        {"mood": "ANGRY ", "intensity": "0.9", "cause": "x" * 300}, EmotionalState()
    )
    assert len(result.cause) == 200
    assert result.mood == "angry"

    assert Character._parse_relationship_deltas({"trust": 0.2, "affection": "bad"}) == {
        "trust": 0.2,
        "respect": 0.0,
        "familiarity": 0.0,
        "tension": 0.0,
    }
    assert Character._parse_relationship_deltas("not a dict") == {}

    assert Character._parse_facts(["ok fact that is long enough", 5, "tiny"], 5) == [
        "ok fact that is long enough"
    ]
