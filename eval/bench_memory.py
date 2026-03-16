"""Memory benchmarks — recall accuracy, cross-session persistence, consolidation."""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from woven_imprint import Engine
from woven_imprint.llm.base import LLMProvider
from woven_imprint.embedding.base import EmbeddingProvider
from eval.framework import BenchmarkResult, SuiteResult, timed

# ── Deterministic test providers ────────────────────────────


class EvalEmbedder(EmbeddingProvider):
    """Embedding that produces consistent vectors based on word overlap."""

    def __init__(self, vocab_size: int = 100):
        self._vocab: dict[str, int] = {}
        self._next_idx = 0
        self._dims = vocab_size

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dims
        for word in self._tokenize(text):
            if word not in self._vocab:
                self._vocab[word] = self._next_idx % self._dims
                self._next_idx += 1
            vec[self._vocab[word]] += 1.0
        # Normalize
        magnitude = sum(x * x for x in vec) ** 0.5
        if magnitude > 0:
            vec = [x / magnitude for x in vec]
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def dimensions(self) -> int:
        return self._dims


class EvalLLM(LLMProvider):
    """Deterministic LLM that returns predictable responses for eval."""

    def generate(self, messages, temperature=0.7, max_tokens=2048):
        return "I acknowledge the situation and respond in character."

    def generate_json(self, messages, temperature=0.3):
        # For fact extraction, return empty (we inject facts manually)
        return []


# ── Benchmarks ──────────────────────────────────────────────


@timed
def bench_recall_precision(engine: Engine) -> BenchmarkResult:
    """Store specific facts, then query for them. Measure precision@5."""
    char = engine.create_character("RecallTest", persona={"personality": "test"})

    # Store 10 distinct facts as core memories
    facts = [
        "Alice's favorite color is blue",
        "Bob lives in Manchester",
        "The project deadline is March 30th",
        "Charlie plays the violin every Tuesday",
        "Diana allergic to shellfish",
        "The office is on the 4th floor of Riverside Building",
        "Edward drives a red Honda Civic",
        "Fiona graduated from Edinburgh University in 2019",
        "The team budget is 50,000 pounds",
        "George has two cats named Salt and Pepper",
    ]
    for fact in facts:
        char.memory.add(content=fact, tier="core", importance=0.8)

    # Query for specific facts and check if they're in top-5
    queries = [
        ("favorite color", "blue"),
        ("lives in Manchester", "Bob"),
        ("deadline", "March"),
        ("violin", "Charlie"),
        ("allergic", "shellfish"),
        ("office floor", "4th"),
        ("drives", "Honda"),
        ("graduated Edinburgh", "Fiona"),
        ("budget", "50,000"),
        ("cats", "Salt"),
    ]

    hits = 0
    for query, expected_keyword in queries:
        results = char.recall(query, limit=5)
        top_contents = " ".join(r["content"] for r in results)
        if expected_keyword.lower() in top_contents.lower():
            hits += 1

    precision = hits / len(queries)
    return BenchmarkResult(
        name="recall_precision_at_5",
        passed=precision >= 0.7,
        score=precision,
        details={"hits": hits, "queries": len(queries), "facts_stored": len(facts)},
    )


@timed
def bench_cross_session_persistence(engine: Engine) -> BenchmarkResult:
    """Store facts in session 1, end session, recall in session 2."""
    char = engine.create_character("PersistTest", persona={"personality": "test"})

    # Session 1: store facts
    char.start_session()
    char.memory.add(
        content="User mentioned they are moving to Tokyo next month",
        tier="core",
        session_id=char._session_id,
        importance=0.9,
    )
    char.memory.add(
        content="User's daughter Emma just started school",
        tier="core",
        session_id=char._session_id,
        importance=0.8,
    )
    char.memory.add(
        content="User prefers morning meetings before 10am",
        tier="core",
        session_id=char._session_id,
        importance=0.6,
    )
    char._session_id = None  # End session without LLM summary

    # Session 2: recall facts
    char.start_session()
    results_tokyo = char.recall("moving to Tokyo", limit=3)
    results_emma = char.recall("daughter school", limit=3)
    results_meeting = char.recall("meeting preference", limit=3)

    found_tokyo = any("Tokyo" in r["content"] for r in results_tokyo)
    found_emma = any("Emma" in r["content"] for r in results_emma)
    found_meeting = any("morning" in r["content"].lower() for r in results_meeting)

    score = sum([found_tokyo, found_emma, found_meeting]) / 3.0
    return BenchmarkResult(
        name="cross_session_persistence",
        passed=score >= 0.66,
        score=score,
        details={"tokyo": found_tokyo, "emma": found_emma, "meeting": found_meeting},
    )


@timed
def bench_memory_tier_separation(engine: Engine) -> BenchmarkResult:
    """Verify that bedrock > core > buffer in retrieval importance."""
    char = engine.create_character("TierTest", persona={"backstory": "A detective in London"})

    # Bedrock: high importance, core: medium, buffer: low
    char.memory.add(
        content="My fundamental identity as a detective in London defines who I am",
        tier="bedrock",
        importance=0.95,
    )
    char.memory.add(
        content="Currently working on a detective case in London investigating fraud",
        tier="core",
        importance=0.7,
    )
    char.memory.add(
        content="Someone mentioned rain in London today at lunch briefly",
        tier="buffer",
        importance=0.2,
    )

    results = char.recall("detective London identity", limit=3)
    if len(results) < 3:
        return BenchmarkResult(
            name="memory_tier_separation",
            passed=False,
            score=0.0,
            details={"error": f"Only {len(results)} results returned"},
        )

    # Bedrock should score highest due to importance weighting in RRF
    # We check that bedrock appears in top 2 (importance + semantic boost it)
    tiers = [r["tier"] for r in results]
    bedrock_in_top2 = "bedrock" in tiers[:2]
    all_tiers_present = set(tiers) == {"bedrock", "core", "buffer"}
    score = (1.0 if bedrock_in_top2 else 0.0) * 0.6 + (1.0 if all_tiers_present else 0.0) * 0.4

    return BenchmarkResult(
        name="memory_tier_separation",
        passed=bedrock_in_top2,
        score=score,
        details={
            "tier_order": tiers,
            "bedrock_in_top2": bedrock_in_top2,
            "all_tiers_present": all_tiers_present,
        },
    )


@timed
def bench_belief_revision(engine: Engine) -> BenchmarkResult:
    """Test that contradicted memories lose certainty and new ones take over."""
    char = engine.create_character("BeliefTest", persona={"personality": "test"})

    # Store original belief
    mem = char.memory.add(content="Bob's favorite food is pizza", tier="core", importance=0.8)
    original_id = mem["id"]

    # Contradict it
    new_mem = char.belief.contradict(original_id, "Bob's favorite food is sushi, not pizza")

    # Check original is contradicted
    old = char.memory.get(original_id)
    old_status = old["status"] if old else "missing"
    old_certainty = old["certainty"] if old else -1

    # Check new memory is active
    new = char.memory.get(new_mem["id"])
    new_status = new["status"] if new else "missing"

    # Recall should return sushi, not pizza
    results = char.recall("Bob's favorite food", limit=3)
    top_content = results[0]["content"] if results else ""
    sushi_top = "sushi" in top_content.lower()

    checks = {
        "old_contradicted": old_status == "contradicted",
        "old_certainty_zero": old_certainty == 0.0,
        "new_active": new_status == "active",
        "sushi_recalled_first": sushi_top,
    }
    score = sum(checks.values()) / len(checks)

    return BenchmarkResult(
        name="belief_revision",
        passed=score >= 0.75,
        score=score,
        details=checks,
    )


@timed
def bench_relationship_bounded_change(engine: Engine) -> BenchmarkResult:
    """Verify relationship deltas are bounded to MAX_DELTA per interaction."""
    char = engine.create_character("RelBoundTest", persona={"personality": "test"})

    # Try to spike trust by 1.0 in one update
    char.relationships.get_or_create("target1")
    rel = char.relationships.update("target1", {"trust": 1.0, "affection": -1.0})

    trust = rel["dimensions"]["trust"]
    affection = rel["dimensions"]["affection"]

    # Should be clamped to ±0.15
    trust_bounded = abs(trust) <= 0.15 + 0.001
    affection_bounded = abs(affection) <= 0.15 + 0.001

    score = (1.0 if trust_bounded else 0.0) * 0.5 + (1.0 if affection_bounded else 0.0) * 0.5

    return BenchmarkResult(
        name="relationship_bounded_change",
        passed=trust_bounded and affection_bounded,
        score=score,
        details={
            "trust": trust,
            "affection": affection,
            "trust_bounded": trust_bounded,
            "affection_bounded": affection_bounded,
        },
    )


@timed
def bench_familiarity_monotonic(engine: Engine) -> BenchmarkResult:
    """Familiarity should only increase — you can't un-know someone."""
    char = engine.create_character("FamTest", persona={"personality": "test"})

    char.relationships.get_or_create("target1")
    char.relationships.update("target1", {"familiarity": 0.1})
    rel1 = char.relationships.get("target1")
    fam1 = rel1["dimensions"]["familiarity"]

    # Try to decrease
    char.relationships.update("target1", {"familiarity": -0.1})
    rel2 = char.relationships.get("target1")
    fam2 = rel2["dimensions"]["familiarity"]

    monotonic = fam2 >= fam1
    return BenchmarkResult(
        name="familiarity_monotonic",
        passed=monotonic,
        score=1.0 if monotonic else 0.0,
        details={"fam_after_increase": fam1, "fam_after_decrease_attempt": fam2},
    )


@timed
def bench_birthday_age(engine: Engine) -> BenchmarkResult:
    """Age should derive from birthdate correctly."""
    char = engine.create_character(
        "AgeTest",
        birthdate="2000-06-15",
        persona={"personality": "test"},
    )

    age = char.persona.age
    from datetime import date

    today = date.today()
    expected = today.year - 2000
    if (today.month, today.day) < (6, 15):
        expected -= 1

    correct = age == expected
    return BenchmarkResult(
        name="birthday_derived_age",
        passed=correct,
        score=1.0 if correct else 0.0,
        details={"computed_age": age, "expected_age": expected},
    )


# ── Suite runner ────────────────────────────────────────────


def run_memory_suite() -> SuiteResult:
    """Run all memory benchmarks."""
    import time

    start = time.time()
    suite = SuiteResult(suite_name="Memory & Core Mechanics")

    embedder = EvalEmbedder()
    llm = EvalLLM()
    engine = Engine(db_path=":memory:", llm=llm, embedding=embedder)

    benchmarks = [
        bench_recall_precision,
        bench_cross_session_persistence,
        bench_memory_tier_separation,
        bench_belief_revision,
        bench_relationship_bounded_change,
        bench_familiarity_monotonic,
        bench_birthday_age,
    ]

    for bench in benchmarks:
        # Fresh engine per benchmark to avoid cross-contamination
        engine = Engine(db_path=":memory:", llm=llm, embedding=embedder)
        result = bench(engine)
        suite.results.append(result)
        engine.close()

    suite.total_duration_ms = (time.time() - start) * 1000
    return suite


if __name__ == "__main__":
    suite = run_memory_suite()
    print(suite.summary())
