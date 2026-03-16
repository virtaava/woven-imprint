"""Persona benchmarks — consistency checking, constraint enforcement, growth."""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from woven_imprint.persona.model import PersonaModel
from woven_imprint.persona.consistency import ConsistencyChecker
from woven_imprint.persona.growth import GrowthEngine, GrowthEvent
from woven_imprint.llm.base import LLMProvider
from woven_imprint.embedding.base import EmbeddingProvider
from eval.framework import BenchmarkResult, SuiteResult, timed


class EvalEmbedder(EmbeddingProvider):
    def embed(self, text):
        h = hash(text) % 1000
        return [h / 1000, (h * 7) % 1000 / 1000, (h * 13) % 1000 / 1000]

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]

    def dimensions(self):
        return 3


class EvalLLM(LLMProvider):
    def generate(self, messages, temperature=0.7, max_tokens=2048):
        return "Response."

    def generate_json(self, messages, temperature=0.3):
        return []


# ── Benchmarks ──────────────────────────────────────────────


@timed
def bench_hard_constraint_detection() -> BenchmarkResult:
    """Consistency checker should flag hard constraint violations."""
    persona = PersonaModel(
        {
            "name": "Alice",
            "hard": {"name": "Alice", "species": "human", "birthplace": "London"},
            "backstory": "Born and raised in London",
            "soft": {"personality": "witty"},
        }
    )

    # LLM that reports violations when they exist
    class ViolationDetector(LLMProvider):
        def generate(self, messages, **kw):
            return ""

        def generate_json(self, messages, **kw):
            user_msg = messages[-1]["content"] if messages else ""
            if "My name is Bob" in user_msg:
                return {
                    "hard_violations": ["Character claims name is Bob, but name is Alice"],
                    "soft_flags": [],
                    "score": 0.1,
                }
            if "born in Tokyo" in user_msg:
                return {
                    "hard_violations": ["Character claims born in Tokyo, but born in London"],
                    "soft_flags": [],
                    "score": 0.2,
                }
            return {"hard_violations": [], "soft_flags": [], "score": 0.95}

    checker = ConsistencyChecker(ViolationDetector(), persona)

    # Should detect violation
    r1 = checker.check("My name is Bob and I'm from here.")
    # Should detect violation
    r2 = checker.check("I was born in Tokyo, lovely city.")
    # Should pass
    r3 = checker.check("I'm working on the case in my London office.")

    checks = {
        "name_violation_caught": not r1.consistent,
        "birthplace_violation_caught": not r2.consistent,
        "clean_response_passes": r3.consistent,
        "violation_score_low": r1.score < 0.5,
        "clean_score_high": r3.score > 0.5,
    }
    score = sum(checks.values()) / len(checks)

    return BenchmarkResult(
        name="hard_constraint_detection",
        passed=score >= 0.8,
        score=score,
        details=checks,
    )


@timed
def bench_soft_flag_not_blocking() -> BenchmarkResult:
    """Soft constraint flags should not mark response as inconsistent."""
    persona = PersonaModel(
        {
            "name": "Alice",
            "hard": {"name": "Alice"},
            "soft": {"personality": "serious and stoic"},
        }
    )

    class SoftFlagger(LLMProvider):
        def generate(self, messages, **kw):
            return ""

        def generate_json(self, messages, **kw):
            return {
                "hard_violations": [],
                "soft_flags": ["Character is unusually cheerful"],
                "score": 0.7,
            }

    checker = ConsistencyChecker(SoftFlagger(), persona)
    report = checker.check("What a wonderful day! Ha ha!")

    checks = {
        "still_consistent": report.consistent,
        "has_soft_flags": len(report.soft_flags) > 0,
        "score_not_zero": report.score > 0.0,
    }
    score = sum(checks.values()) / len(checks)

    return BenchmarkResult(
        name="soft_flag_not_blocking",
        passed=score >= 1.0,
        score=score,
        details=checks,
    )


@timed
def bench_temporal_age_derivation() -> BenchmarkResult:
    """Temporal facts: age derived from birthdate, birthday detection."""
    from datetime import date

    today = date.today()

    # Character whose birthday is today
    bday_today = f"2000-{today.month:02d}-{today.day:02d}"
    p1 = PersonaModel({"name": "A"}, birthdate=bday_today)

    # Character whose birthday is not today
    other_month = (today.month % 12) + 1
    bday_other = f"2000-{other_month:02d}-15"
    p2 = PersonaModel({"name": "B"}, birthdate=bday_other)

    # Character with no birthdate but static age
    p3 = PersonaModel({"name": "C", "hard": {"age": 30}})

    checks = {
        "birthday_today_detected": p1.is_birthday,
        "birthday_today_age_correct": p1.age == today.year - 2000,
        "other_not_birthday": not p2.is_birthday,
        "other_age_not_none": p2.age is not None,
        "static_age_works": p3.age == 30,
        "days_until_birthday_zero": p1.days_until_birthday == 0,
    }
    score = sum(checks.values()) / len(checks)

    return BenchmarkResult(
        name="temporal_age_derivation",
        passed=score >= 0.8,
        score=score,
        details=checks,
    )


@timed
def bench_growth_threshold_gating() -> BenchmarkResult:
    """Growth events below threshold should not be applied."""
    from woven_imprint.storage.sqlite import SQLiteStorage

    storage = SQLiteStorage(":memory:")
    storage.save_character("c1", "Test", {"soft": {"mood": "neutral"}})
    persona = PersonaModel({"name": "Test", "soft": {"mood": "neutral"}})
    engine = GrowthEngine(storage, EvalLLM(), "c1", persona)

    low_confidence = GrowthEvent("mood", "neutral", "happy", "one good day", 0.3)
    high_confidence = GrowthEvent("mood", "neutral", "content", "sustained positive", 0.8)

    applied_low = engine.apply_growth([low_confidence], threshold=0.6)
    applied_high = engine.apply_growth([high_confidence], threshold=0.6)

    checks = {
        "low_not_applied": len(applied_low) == 0,
        "high_applied": len(applied_high) == 1,
        "persona_updated": persona.soft["mood"] == "content",
    }
    score = sum(checks.values()) / len(checks)

    storage.close()
    return BenchmarkResult(
        name="growth_threshold_gating",
        passed=score >= 1.0,
        score=score,
        details=checks,
    )


@timed
def bench_growth_persisted() -> BenchmarkResult:
    """Growth should be persisted to storage and recorded as memory."""
    from woven_imprint.storage.sqlite import SQLiteStorage

    storage = SQLiteStorage(":memory:")
    storage.save_character("c1", "Test", {"soft": {"personality": "shy"}})
    persona = PersonaModel({"name": "Test", "soft": {"personality": "shy"}})
    engine = GrowthEngine(storage, EvalLLM(), "c1", persona)

    event = GrowthEvent("personality", "shy", "shy but more confident", "built trust", 0.75)
    engine.apply_growth([event], threshold=0.6)

    # Check storage
    char = storage.load_character("c1")
    stored_persona = char["persona"]["soft"]["personality"] if char else ""

    # Check growth memory
    memories = storage.get_memories("c1", tier="core")
    growth_mems = [m for m in memories if "[Growth]" in m["content"]]

    checks = {
        "storage_updated": stored_persona == "shy but more confident",
        "growth_memory_created": len(growth_mems) == 1,
        "growth_memory_has_reason": "built trust" in growth_mems[0]["content"]
        if growth_mems
        else False,
    }
    score = sum(checks.values()) / len(checks)

    storage.close()
    return BenchmarkResult(
        name="growth_persisted",
        passed=score >= 1.0,
        score=score,
        details=checks,
    )


@timed
def bench_system_prompt_completeness() -> BenchmarkResult:
    """System prompt should include all persona components."""
    persona = PersonaModel(
        {
            "name": "Alice",
            "hard": {"name": "Alice", "occupation": "detective"},
            "soft": {"personality": "witty and sharp", "speaking_style": "dry humor"},
            "temporal": {"location": "London"},
            "backstory": "Former police officer turned private investigator",
        },
        birthdate="1998-03-15",
    )

    prompt = persona.build_system_prompt()

    checks = {
        "has_name": "Alice" in prompt,
        "has_backstory": "police officer" in prompt.lower() or "investigator" in prompt.lower(),
        "has_personality": "witty" in prompt.lower(),
        "has_speaking_style": "dry humor" in prompt.lower(),
        "has_age": str(persona.age) in prompt,
        "has_location": "London" in prompt,
        "has_stay_in_character": "character" in prompt.lower(),
    }
    score = sum(checks.values()) / len(checks)

    return BenchmarkResult(
        name="system_prompt_completeness",
        passed=score >= 0.85,
        score=score,
        details=checks,
    )


# ── Suite runner ────────────────────────────────────────────


def run_persona_suite() -> SuiteResult:
    import time

    start = time.time()
    suite = SuiteResult(suite_name="Persona & Consistency")

    benchmarks = [
        bench_hard_constraint_detection,
        bench_soft_flag_not_blocking,
        bench_temporal_age_derivation,
        bench_growth_threshold_gating,
        bench_growth_persisted,
        bench_system_prompt_completeness,
    ]

    for bench in benchmarks:
        result = bench()
        suite.results.append(result)

    suite.total_duration_ms = (time.time() - start) * 1000
    return suite


if __name__ == "__main__":
    suite = run_persona_suite()
    print(suite.summary())
