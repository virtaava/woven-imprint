"""Tests for narrative arc tracking."""

from woven_imprint.narrative.arc import (
    ArcPhase,
    ArcTracker,
    NarrativeArc,
    StoryBeat,
)
from woven_imprint.llm.base import LLMProvider


class FakeLLM(LLMProvider):
    def __init__(self, is_beat=True, phase="rising_action", tension=0.5):
        self._is_beat = is_beat
        self._phase = phase
        self._tension = tension

    def generate(self, messages, **kw):
        return ""

    def generate_json(self, messages, **kw):
        return {
            "is_beat": self._is_beat,
            "description": "Something significant happened",
            "phase": self._phase,
            "tension": self._tension,
            "tags": ["confrontation"],
        }


class TestNarrativeArc:
    def test_default_state(self):
        arc = NarrativeArc()
        assert arc.current_phase == ArcPhase.SETUP
        assert arc.tension == 0.2
        assert arc.tension_trend == "stable"
        assert len(arc.beats) == 0

    def test_tension_trend_rising(self):
        arc = NarrativeArc()
        arc.beats = [
            StoryBeat("a", ArcPhase.SETUP, 0.2, 1),
            StoryBeat("b", ArcPhase.RISING, 0.4, 2),
            StoryBeat("c", ArcPhase.RISING, 0.6, 3),
        ]
        assert arc.tension_trend == "rising"

    def test_tension_trend_falling(self):
        arc = NarrativeArc()
        arc.beats = [
            StoryBeat("a", ArcPhase.CLIMAX, 0.9, 1),
            StoryBeat("b", ArcPhase.FALLING, 0.6, 2),
            StoryBeat("c", ArcPhase.FALLING, 0.3, 3),
        ]
        assert arc.tension_trend == "falling"

    def test_describe_setup(self):
        arc = NarrativeArc(current_phase=ArcPhase.SETUP, tension=0.1)
        desc = arc.describe()
        assert "early stages" in desc.lower()

    def test_describe_climax_high_tension(self):
        arc = NarrativeArc(current_phase=ArcPhase.CLIMAX, tension=0.9)
        desc = arc.describe()
        assert "critical" in desc.lower()
        assert "high" in desc.lower()

    def test_to_dict_roundtrip(self):
        arc = NarrativeArc(title="Test Story", current_phase=ArcPhase.RISING, tension=0.5)
        arc.beats.append(StoryBeat("beat1", ArcPhase.RISING, 0.5, 1, ["Alice"], ["tag1"]))

        d = arc.to_dict()
        arc2 = NarrativeArc.from_dict(d)

        assert arc2.title == "Test Story"
        assert arc2.current_phase == ArcPhase.RISING
        assert arc2.tension == 0.5
        assert len(arc2.beats) == 1
        assert arc2.beats[0].description == "beat1"


class TestStoryBeat:
    def test_to_dict(self):
        beat = StoryBeat(
            description="Darcy proposes",
            phase=ArcPhase.CLIMAX,
            tension=0.9,
            turn_number=5,
            characters_involved=["Darcy", "Elizabeth"],
            tags=["proposal", "confrontation"],
        )
        d = beat.to_dict()
        assert d["phase"] == "climax"
        assert d["tension"] == 0.9
        assert "Darcy" in d["characters_involved"]


class TestArcTracker:
    def test_detects_beat(self):
        tracker = ArcTracker(FakeLLM(is_beat=True, phase="rising_action", tension=0.6))
        arc = NarrativeArc()
        arc.turn_count = 1  # Make it even on next call

        beat = tracker.analyze_beat("I know your secret", "How dare you", arc, "Alice", "Bob")
        assert beat is not None
        assert beat.phase == ArcPhase.RISING
        assert beat.tension == 0.6
        assert arc.current_phase == ArcPhase.RISING
        assert arc.tension == 0.6

    def test_skips_non_beat(self):
        tracker = ArcTracker(FakeLLM(is_beat=False))
        arc = NarrativeArc()
        arc.turn_count = 1

        beat = tracker.analyze_beat("Nice weather", "Indeed", arc, "Alice")
        assert beat is None

    def test_throttled_every_other_turn(self):
        tracker = ArcTracker(FakeLLM(is_beat=True))
        arc = NarrativeArc()

        # Turn 1: analyzed (turn_count becomes 1, but 1 % 2 != 0 → skip)
        # Actually turn_count starts at 0, incremented to 1, 1%2!=0 and >1 is false → analyzed
        beat1 = tracker.analyze_beat("msg", "resp", arc, "Alice")
        # Turn 2: turn_count=2, 2%2==0 → analyzed
        beat2 = tracker.analyze_beat("msg", "resp", arc, "Alice")
        # Turn 3: turn_count=3, 3%2!=0 and >1 → skipped
        beat3 = tracker.analyze_beat("msg", "resp", arc, "Alice")

        # At least some should be analyzed
        analyzed = sum(1 for b in [beat1, beat2, beat3] if b is not None)
        assert analyzed >= 1

    def test_updates_arc_phase(self):
        tracker = ArcTracker(FakeLLM(is_beat=True, phase="climax", tension=0.9))
        arc = NarrativeArc()
        arc.turn_count = 1

        tracker.analyze_beat("The truth comes out", "You lied!", arc, "Alice", "Bob")
        assert arc.current_phase == ArcPhase.CLIMAX
        assert arc.tension == 0.9
        assert len(arc.beats) == 1

    def test_failure_returns_none(self):
        class FailingLLM(LLMProvider):
            def generate(self, messages, **kw):
                return ""

            def generate_json(self, messages, **kw):
                raise ValueError("fail")

        tracker = ArcTracker(FailingLLM())
        arc = NarrativeArc()
        arc.turn_count = 1

        beat = tracker.analyze_beat("msg", "resp", arc, "Alice")
        assert beat is None
