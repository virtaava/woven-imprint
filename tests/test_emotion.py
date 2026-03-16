"""Tests for emotional state tracking."""

from woven_imprint.persona.emotion import EmotionalState, EmotionEngine, EMOTION_LABELS
from woven_imprint.llm.base import LLMProvider


class FakeLLM(LLMProvider):
    def __init__(self, mood="content", intensity=0.5, cause="a nice chat"):
        self._mood = mood
        self._intensity = intensity
        self._cause = cause

    def generate(self, messages, **kw):
        return ""

    def generate_json(self, messages, **kw):
        return {"mood": self._mood, "intensity": self._intensity, "cause": self._cause}


class TestEmotionalState:
    def test_default_neutral(self):
        e = EmotionalState()
        assert e.mood == "neutral"
        assert e.intensity == 0.5

    def test_describe_neutral(self):
        e = EmotionalState()
        # Neutral at low intensity = no description
        e.intensity = 0.1
        assert e.describe() == ""

    def test_describe_active_mood(self):
        e = EmotionalState(mood="anxious", intensity=0.6, cause="upcoming exam")
        desc = e.describe()
        assert "anxious" in desc
        assert "upcoming exam" in desc

    def test_describe_intensity_words(self):
        e = EmotionalState(mood="joyful")

        e.intensity = 0.3
        assert "slightly" in e.describe()

        e.intensity = 0.5
        assert "quite" in e.describe()

        e.intensity = 0.8
        assert "intensely" in e.describe()

    def test_describe_lingering(self):
        e = EmotionalState(mood="sad", intensity=0.5, turns_held=5)
        assert "lingering" in e.describe()

    def test_decay(self):
        e = EmotionalState(mood="angry", intensity=0.6)
        e.decay(rate=0.15)
        assert abs(e.intensity - 0.45) < 0.001
        assert e.turns_held == 1
        assert e.mood == "angry"  # Still angry

    def test_decay_to_neutral(self):
        e = EmotionalState(mood="angry", intensity=0.1)
        e.decay(rate=0.15)
        assert e.mood == "neutral"
        assert e.turns_held == 0

    def test_to_dict_roundtrip(self):
        e = EmotionalState(mood="excited", intensity=0.7, cause="good news", turns_held=2)
        d = e.to_dict()
        e2 = EmotionalState.from_dict(d)
        assert e2.mood == "excited"
        assert e2.intensity == 0.7
        assert e2.cause == "good news"
        assert e2.turns_held == 2


class TestEmotionEngine:
    def test_assess_updates_mood(self):
        engine = EmotionEngine(FakeLLM(mood="joyful", intensity=0.6, cause="positive exchange"))
        current = EmotionalState()
        new = engine.assess("Hello!", "Hi there!", current, "Alice")
        assert new.mood == "joyful"
        assert new.intensity == 0.6
        assert new.turns_held == 0

    def test_assess_clamps_intensity(self):
        engine = EmotionEngine(FakeLLM(mood="angry", intensity=5.0))
        current = EmotionalState()
        new = engine.assess("msg", "resp", current, "Alice")
        assert new.intensity <= 1.0

    def test_assess_invalid_mood_defaults_neutral(self):
        engine = EmotionEngine(FakeLLM(mood="invented_mood"))
        current = EmotionalState()
        new = engine.assess("msg", "resp", current, "Alice")
        assert new.mood == "neutral"

    def test_assess_failure_decays(self):
        class FailingLLM(LLMProvider):
            def generate(self, messages, **kw):
                return ""

            def generate_json(self, messages, **kw):
                raise ValueError("fail")

        engine = EmotionEngine(FailingLLM())
        current = EmotionalState(mood="happy", intensity=0.6)
        result = engine.assess("msg", "resp", current, "Alice")
        # Should decay rather than crash
        assert result.intensity < 0.6

    def test_all_labels_valid(self):
        for label in EMOTION_LABELS:
            assert isinstance(label, str)
            assert len(EMOTION_LABELS[label]) == 3
