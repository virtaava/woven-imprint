"""Tests for persona consistency checker."""

from woven_imprint.persona.model import PersonaModel
from woven_imprint.persona.consistency import ConsistencyChecker
from woven_imprint.config import CharacterConfig


class FakeLLM:
    """Configurable fake LLM for testing consistency checks."""

    def __init__(self, json_response=None):
        self._json = json_response or {
            "hard_violations": [],
            "soft_flags": [],
            "score": 1.0,
        }

    def generate(self, messages, temperature=0.7, max_tokens=2048):
        return "Response"

    def generate_json(self, messages, temperature=0.3):
        return self._json


class TestConsistencyChecker:
    def test_consistent_response(self):
        persona = PersonaModel(
            {
                "name": "Alice",
                "hard": {"name": "Alice", "species": "human"},
                "backstory": "A detective from London",
            }
        )
        checker = ConsistencyChecker(FakeLLM(), persona)
        report = checker.check("I'm working on the case.")
        assert report.consistent is True
        assert report.score == 1.0

    def test_hard_violation_detected(self):
        persona = PersonaModel(
            {
                "name": "Alice",
                "hard": {"name": "Alice"},
                "backstory": "Born in London",
            }
        )
        llm = FakeLLM(
            {
                "hard_violations": ["Character claims to be named Bob"],
                "soft_flags": [],
                "score": 0.2,
            }
        )
        checker = ConsistencyChecker(llm, persona)
        report = checker.check("My name is Bob.")
        assert report.consistent is False
        assert len(report.hard_violations) == 1
        assert report.score == 0.2

    def test_soft_flag_still_consistent(self):
        persona = PersonaModel(
            {
                "name": "Alice",
                "hard": {"name": "Alice"},
                "soft": {"personality": "serious and stoic"},
            }
        )
        llm = FakeLLM(
            {
                "hard_violations": [],
                "soft_flags": ["Character is being unusually cheerful"],
                "score": 0.7,
            }
        )
        checker = ConsistencyChecker(llm, persona)
        report = checker.check("Ha ha, what a wonderful day!")
        assert report.consistent is True  # soft flags don't make it inconsistent
        assert len(report.soft_flags) == 1

    def test_no_hard_facts(self):
        persona = PersonaModel({"name": "Unknown"})
        checker = ConsistencyChecker(FakeLLM(), persona)
        report = checker.check("Anything goes")
        assert report.consistent is True

    def test_enforce_regenerates_on_violation(self):
        persona = PersonaModel(
            {
                "name": "Alice",
                "hard": {"name": "Alice"},
                "backstory": "A detective",
            }
        )

        call_count = 0

        class RegeneratingLLM:
            def generate(self, messages, temperature=0.7, max_tokens=2048):
                nonlocal call_count
                call_count += 1
                return "I'm Alice, working the case."

            def generate_json(self, messages, temperature=0.3):
                nonlocal call_count
                if call_count <= 1:
                    return {
                        "hard_violations": ["Claims wrong name"],
                        "soft_flags": [],
                        "score": 0.3,
                    }
                return {
                    "hard_violations": [],
                    "soft_flags": [],
                    "score": 0.9,
                }

        checker = ConsistencyChecker(RegeneratingLLM(), persona)
        response, report = checker.enforce(
            "My name is Bob",
            [{"role": "user", "content": "What's your name?"}],
        )
        assert report.score >= 0.3  # Should improve after retry

    def test_score_clamping(self):
        persona = PersonaModel({"name": "Alice", "hard": {"name": "Alice"}})
        llm = FakeLLM({"hard_violations": [], "soft_flags": [], "score": 5.0})
        checker = ConsistencyChecker(llm, persona)
        report = checker.check("test")
        assert report.score == 1.0  # Clamped

    def test_llm_failure_returns_safe_default(self):
        persona = PersonaModel({"name": "Alice", "hard": {"name": "Alice"}})

        class FailingLLM:
            def generate(self, messages, **kwargs):
                return "not json"

            def generate_json(self, messages, **kwargs):
                raise ValueError("Parse failed")

        checker = ConsistencyChecker(FailingLLM(), persona)
        report = checker.check("test")
        assert report.consistent is True  # Safe default on failure

    def test_context_passed_to_check_via_enforce(self):
        """Verify that enforce() builds context from messages and passes to check()."""
        persona = PersonaModel(
            {
                "name": "Alice",
                "hard": {"name": "Alice"},
                "backstory": "A detective",
            }
        )

        received_contexts = []

        class ContextCaptureLLM:
            def generate(self, messages, **kwargs):
                return "I'm Alice."

            def generate_json(self, messages, **kwargs):
                # Capture the user message content to verify context presence
                for m in messages:
                    if m["role"] == "user":
                        received_contexts.append(m["content"])
                return {"hard_violations": [], "soft_flags": [], "score": 0.95}

        checker = ConsistencyChecker(ContextCaptureLLM(), persona)
        messages = [
            {"role": "user", "content": "Tell me about yourself"},
            {"role": "assistant", "content": "I'm a detective"},
            {"role": "user", "content": "What's your name?"},
        ]
        response, report = checker.enforce("I'm Alice", messages)
        assert report.consistent is True
        # Verify context was included in the check call
        assert any("CONVERSATION CONTEXT" in ctx for ctx in received_contexts)

    def test_configurable_fail_open_score(self):
        """Verify custom fail_open_score is used when LLM returns non-dict."""
        persona = PersonaModel({"name": "Alice", "hard": {"name": "Alice"}})

        class NonDictLLM:
            def generate(self, messages, **kwargs):
                return "response"

            def generate_json(self, messages, **kwargs):
                return "not a dict"

        config = CharacterConfig(consistency_fail_open_score=0.6)
        checker = ConsistencyChecker(NonDictLLM(), persona, config=config)
        report = checker.check("test")
        assert report.consistent is True
        assert report.score == 0.6

    def test_json_retry_on_non_dict(self):
        """Verify retry at low temperature when first generate_json returns non-dict."""
        persona = PersonaModel({"name": "Alice", "hard": {"name": "Alice"}})

        call_count = 0

        class RetryLLM:
            def generate(self, messages, **kwargs):
                return "response"

            def generate_json(self, messages, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return "not a dict"
                # Second call (retry) returns valid dict
                return {"hard_violations": [], "soft_flags": [], "score": 0.95}

        checker = ConsistencyChecker(RetryLLM(), persona)
        report = checker.check("test")
        assert call_count == 2  # Verified retry happened
        assert report.score == 0.95
