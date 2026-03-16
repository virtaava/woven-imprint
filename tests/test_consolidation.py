"""Tests for memory consolidation engine."""

import pytest

from woven_imprint.storage.sqlite import SQLiteStorage
from woven_imprint.memory.consolidation import ConsolidationEngine, _cluster_memories


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
            storage.save_memory({
                "id": f"m{i}", "character_id": "c1", "tier": "buffer",
                "content": f"memory {i}",
            })
        assert engine.needs_consolidation()

    def test_consolidate_too_few(self, storage):
        engine = ConsolidationEngine(storage, FakeLLM(), FakeEmbedder(), "c1")
        stats = engine.consolidate()
        assert stats["clusters"] == 0

    def test_consolidate_archives_originals(self, storage):
        # Create 15 buffer memories with identical embeddings so they cluster
        vec = [0.5, 0.5, 0.5]
        for i in range(15):
            storage.save_memory({
                "id": f"m{i}", "character_id": "c1", "tier": "buffer",
                "content": f"similar memory {i}", "embedding": vec,
            })

        engine = ConsolidationEngine(
            storage, FakeLLM(), FakeEmbedder(), "c1", threshold=10
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
            storage.save_memory({
                "id": f"m{i}", "character_id": "c1", "tier": "buffer",
                "content": f"memory {i}", "embedding": vec,
            })

        engine = ConsolidationEngine(storage, FakeLLM(), FakeEmbedder(), "c1")
        stats = engine.consolidate(dry_run=True)

        assert stats["summarized"] > 0
        # Nothing should be written in dry run
        active = storage.get_memories("c1", tier="buffer", status="active")
        assert len(active) == 15
