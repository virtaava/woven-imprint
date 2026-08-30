"""Tests for memory consolidation engine."""

import pytest

from woven_imprint.config import get_config
from woven_imprint.storage.sqlite import SQLiteStorage
from woven_imprint.memory.consolidation import ConsolidationEngine, _cluster_memories

from tests.helpers import make_test_engine


class FakeEmbedder:
    """Fake embedder that returns deterministic vectors based on content hash."""

    def embed(self, text: str) -> list[float]:
        h = hash(text) % 1000
        return [h / 1000, (h * 7) % 1000 / 1000, (h * 13) % 1000 / 1000]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def dimensions(self) -> int:
        return 3


class FakeLLM:
    """Fake LLM that returns predictable summaries."""

    def generate(self, messages, temperature=0.7, max_tokens=2048):
        return "Consolidated summary of related memories."

    def generate_json(self, messages, temperature=0.3):
        return {"facts": []}


@pytest.fixture
def storage():
    s = SQLiteStorage(":memory:")
    s.save_character("c1", "Alice", {})
    yield s
    s.close()


class TestClustering:
    def test_empty(self):
        assert _cluster_memories([]) == []

    def test_singleton(self):
        mems = [{"id": "m1", "content": "hello", "embedding": [1.0, 0.0, 0.0]}]
        clusters = _cluster_memories(mems)
        assert len(clusters) == 1
        assert len(clusters[0]) == 1

    def test_identical_vectors_cluster(self):
        vec = [0.5, 0.5, 0.5]
        mems = [
            {"id": "m1", "content": "a", "embedding": vec},
            {"id": "m2", "content": "b", "embedding": vec},
            {"id": "m3", "content": "c", "embedding": vec},
        ]
        clusters = _cluster_memories(mems, similarity_threshold=0.99)
        assert len(clusters) == 1
        assert len(clusters[0]) == 3

    def test_distant_vectors_separate(self):
        mems = [
            {"id": "m1", "content": "a", "embedding": [1.0, 0.0, 0.0]},
            {"id": "m2", "content": "b", "embedding": [0.0, 1.0, 0.0]},
        ]
        clusters = _cluster_memories(mems, similarity_threshold=0.9)
        assert len(clusters) == 2

    def test_no_embedding(self):
        mems = [{"id": "m1", "content": "a"}]
        clusters = _cluster_memories(mems)
        assert len(clusters) == 1


class TestConsolidationEngine:
    def test_needs_consolidation(self, storage):
        engine = ConsolidationEngine(storage, FakeLLM(), FakeEmbedder(), "c1", threshold=5)
        assert not engine.needs_consolidation()

        for i in range(6):
            storage.save_memory(
                {
                    "id": f"m{i}",
                    "character_id": "c1",
                    "tier": "buffer",
                    "content": f"memory {i}",
                }
            )
        assert engine.needs_consolidation()

    def test_consolidate_too_few(self, storage):
        engine = ConsolidationEngine(storage, FakeLLM(), FakeEmbedder(), "c1")
        stats = engine.consolidate()
        assert stats["clusters"] == 0

    def test_consolidate_archives_originals(self, storage):
        # Create 15 buffer memories with identical embeddings so they cluster
        vec = [0.5, 0.5, 0.5]
        for i in range(15):
            storage.save_memory(
                {
                    "id": f"m{i}",
                    "character_id": "c1",
                    "tier": "buffer",
                    "content": f"similar memory {i}",
                    "embedding": vec,
                }
            )

        # Tier 3c: default consolidation_keep_sources=True keeps cluster
        # sources active instead of archiving them — force the legacy
        # archive-everything path to preserve this test's original intent
        # (see tests/test_consolidation_keep_sources.py for the new default).
        engine = ConsolidationEngine(
            storage, FakeLLM(), FakeEmbedder(), "c1", threshold=10, keep_sources=False
        )
        stats = engine.consolidate()

        assert stats["created"] >= 1
        assert stats["archived"] >= 1

        # Check originals are archived
        active_buffer = storage.get_memories("c1", tier="buffer", status="active")
        assert len(active_buffer) == 0

        # Check core memory was created
        core = storage.get_memories("c1", tier="core", status="active")
        assert len(core) >= 1
        assert "[Consolidated]" in core[0]["content"]

    def test_dry_run(self, storage):
        vec = [0.5, 0.5, 0.5]
        for i in range(15):
            storage.save_memory(
                {
                    "id": f"m{i}",
                    "character_id": "c1",
                    "tier": "buffer",
                    "content": f"memory {i}",
                    "embedding": vec,
                }
            )

        engine = ConsolidationEngine(storage, FakeLLM(), FakeEmbedder(), "c1")
        stats = engine.consolidate(dry_run=True)

        assert stats["summarized"] > 0
        # Nothing should be written in dry run
        active = storage.get_memories("c1", tier="buffer", status="active")
        assert len(active) == 15


@pytest.fixture
def consolidation_setup():
    """A live Character (with call-counting FakeLLM) wired through Engine,
    so drain()/dry_run behavior can be exercised end-to-end."""
    engine = make_test_engine()
    char = engine.create_character("Watcher")
    char.consolidator.threshold = 10
    return engine, char


def test_dry_run_makes_no_llm_calls(consolidation_setup):
    engine, char = consolidation_setup
    for i in range(30):
        char.memory.add(f"the lake was calm on day {i}", tier="buffer")
    before = char.llm.call_count
    result = char.consolidator.consolidate(dry_run=True)
    assert char.llm.call_count == before
    assert result["clusters"] >= 1
    # nothing archived under dry_run
    assert engine.storage.count_memories(char.id, tier="buffer") == 30


class _ExhaustedBudget:
    """Fake budget whose take() always denies — for asserting zero LLM calls."""

    def take(self, n: int = 1) -> bool:
        return False


def test_consolidate_budget_exhausted_makes_no_llm_calls(consolidation_setup):
    engine, char = consolidation_setup
    for i in range(15):
        char.memory.add(f"the lake was calm on day {i}", tier="buffer")
    before = char.llm.call_count
    result = char.consolidator.consolidate(budget=_ExhaustedBudget())
    assert char.llm.call_count == before
    assert result["llm_calls"] == 0
    assert result.get("budget_exhausted") is True


class _MismatchedEmbedder:
    """Returns a different dimensionality than the DB already has on file —
    simulates a swapped embedder mid-run."""

    def embed(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4, 0.5]  # 5-d, vs helpers.FakeEmbedder's 50-d

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def dimensions(self) -> int:
        return 5


def test_consolidate_dimension_mismatch_raises_and_writes_nothing(consolidation_setup):
    """Regression for M1: consolidation writes the cluster summary via
    storage.save_memory directly, bypassing MemoryStore.add's dimension
    guard. A swapped embedder must not be able to silently write a
    mixed-dimension vector on this path — it should raise, matching
    MemoryStore.add's behavior."""
    engine, char = consolidation_setup
    for i in range(15):
        char.memory.add(f"the lake was calm on day {i}", tier="buffer")
    # DB now has embedding_dimensions=50 on file (from helpers.FakeEmbedder via memory.add).
    assert engine.storage.meta_get("embedding_dimensions") == "50"

    char.consolidator.embedder = _MismatchedEmbedder()

    with pytest.raises(ValueError, match="[Ee]mbedding dimension mismatch"):
        char.consolidator.consolidate()

    # No mixed-dimension (5-d) vector should have landed in core.
    core = engine.storage.get_memories(char.id, tier="core", status="active")
    for m in core:
        if m.get("embedding"):
            assert len(m["embedding"]) == 50


def test_maintenance_job_consolidate_reports_failure_on_dimension_mismatch(consolidation_setup):
    """The maintenance runner catches per-job exceptions — verify a
    dimension-mismatch ValueError raised deep inside drain() surfaces as a
    failed job entry rather than crashing the whole maintenance run."""
    from woven_imprint.maintenance import MaintenanceRunner

    engine, char = consolidation_setup
    for i in range(15):
        char.memory.add(f"the lake was calm on day {i}", tier="buffer")
    char.consolidator.embedder = _MismatchedEmbedder()

    runner = MaintenanceRunner(char)
    report = runner.run(jobs=["consolidate"])

    assert report["jobs"]["consolidate"]["status"] == "failed"
    assert "dimension mismatch" in report["jobs"]["consolidate"]["error"].lower()
    core = engine.storage.get_memories(char.id, tier="core", status="active")
    for m in core:
        if m.get("embedding"):
            assert len(m["embedding"]) == 50


class _EmptySummaryLLM(FakeLLM):
    """Every summarization call 'fails' by returning an empty string —
    simulates a misbehaving/failing summarizer."""

    def __init__(self):
        self.call_count = 0

    def generate(self, messages, temperature=0.7, max_tokens=2048, **kw):
        self.call_count += 1
        return ""


def test_drain_stops_on_no_progress_pass(consolidation_setup, monkeypatch):
    """Regression for M5: when a pass archives nothing (e.g. every summary
    comes back empty), drain() used to re-cluster the same buffer rows on
    the next pass, re-spending budget for zero gain, up to max_chunks times.
    It should instead stop after the first no-progress pass."""
    engine, char = consolidation_setup
    empty_llm = _EmptySummaryLLM()
    monkeypatch.setattr(char.consolidator, "llm", empty_llm)
    for i in range(15):
        char.memory.add(f"the lake was calm on day {i}", tier="buffer")

    class _UnlimitedBudget:
        def __init__(self):
            self.used = 0

        def take(self, n: int = 1) -> bool:
            self.used += n
            return True

    budget = _UnlimitedBudget()
    result = char.consolidator.drain(max_chunks=10, budget=budget)

    assert result["passes"] == 1
    assert result["archived"] == 0
    # buffer still above threshold — proves the loop stopped early, not
    # because the buffer got drained
    assert char.consolidator.needs_consolidation()


def test_drain_processes_beyond_single_chunk(consolidation_setup, monkeypatch):
    engine, char = consolidation_setup
    # Force a small chunk_size so 60 buffer rows require multiple consolidate()
    # passes to fully drain — this is the behavior drain() adds over the old
    # hardcoded single-pass 500-row cap.
    monkeypatch.setattr(get_config().maintenance, "consolidate_chunk_size", 20)
    for i in range(60):
        char.memory.add(f"note {i} about the {'lake' if i % 2 else 'forest'}", tier="buffer")
    result = char.consolidator.drain(max_chunks=10)
    assert result["passes"] >= 1
    assert not char.consolidator.needs_consolidation()
