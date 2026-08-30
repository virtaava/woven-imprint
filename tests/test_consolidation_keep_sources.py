"""Tests for Tier 3c `consolidation_keep_sources` behavior.

Default (True): consolidation keeps source memories active/retrievable
instead of archiving them (multi-member clusters get `metadata.
consolidated_into`; qualifying singletons are promoted in place; low-importance
singletons get `metadata.consolidation_seen` so they stop recounting toward
the threshold). Passing `keep_sources=False` restores the pre-Tier-3c
archive-everything behavior byte-for-byte (see test H and the adapted
existing tests noted below).
"""

from __future__ import annotations

import pytest

from tests.helpers import make_test_engine
from woven_imprint.memory.consolidation import ConsolidationEngine
from woven_imprint.storage.sqlite import SQLiteStorage


class FakeEmbedder:
    """Deterministic 3-d embedder — vectors passed in explicitly by the tests
    below (via `storage.save_memory({..., "embedding": [...]})`), this class
    only exists to satisfy ConsolidationEngine's constructor / dimension
    guard on the LLM-summary embedding it computes internally."""

    def embed(self, text: str) -> list[float]:
        h = hash(text) % 1000
        return [h / 1000, (h * 7) % 1000 / 1000, (h * 13) % 1000 / 1000]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def dimensions(self) -> int:
        return 3


class FakeLLM:
    def generate(self, messages, temperature=0.3, max_tokens=300, **kw):
        return "Consolidated summary of related memories."

    def generate_json(self, messages, temperature=0.3, **kw):
        return {"facts": []}


CLUSTER_VEC = [0.5, 0.5, 0.5]
PROMOTE_VEC = [1.0, 0.0, 0.0]  # orthogonal to CLUSTER_VEC and SEEN_VEC
SEEN_VEC = [0.0, 1.0, 0.0]


@pytest.fixture
def storage():
    s = SQLiteStorage(":memory:")
    s.save_character("c1", "Alice", {})
    yield s
    s.close()


def _build_mixed_buffer(storage: SQLiteStorage) -> None:
    """8-member similarity cluster + 2 singletons (one >= 0.6 importance
    promotion candidate, one below) — 10 buffer rows total, clearing
    consolidate()'s `len(buffer) < 10` early-return floor in a single call."""
    for i in range(8):
        storage.save_memory(
            {
                "id": f"m-cluster-{i}",
                "character_id": "c1",
                "tier": "buffer",
                "content": f"similar memory about the lighthouse {i}",
                "embedding": CLUSTER_VEC,
                "importance": 0.4,
            }
        )
    storage.save_memory(
        {
            "id": "m-promote",
            "character_id": "c1",
            "tier": "buffer",
            "content": "an important singleton memory",
            "embedding": PROMOTE_VEC,
            "importance": 0.7,
        }
    )
    storage.save_memory(
        {
            "id": "m-seen",
            "character_id": "c1",
            "tier": "buffer",
            "content": "a minor singleton memory",
            "embedding": SEEN_VEC,
            "importance": 0.3,
        }
    )


def _engine(storage: SQLiteStorage, keep_sources: bool | None = True, threshold: int = 5):
    return ConsolidationEngine(
        storage, FakeLLM(), FakeEmbedder(), "c1", threshold=threshold, keep_sources=keep_sources
    )


class TestMultiMemberClusterKeepSources:
    """(a) multi-member cluster -> 1 core row, sources stay active/buffer."""

    def test_creates_one_consolidated_core_row(self, storage):
        _build_mixed_buffer(storage)
        engine = _engine(storage)
        stats = engine.consolidate()

        core = storage.get_memories("c1", tier="core", status="active")
        cluster_core = [m for m in core if "[Consolidated]" in m["content"]]
        assert len(cluster_core) == 1
        assert stats["kept"] == 8
        assert stats["archived"] == 0

    def test_sources_stay_active_with_consolidated_into_and_at(self, storage):
        _build_mixed_buffer(storage)
        engine = _engine(storage)
        engine.consolidate()

        core = storage.get_memories("c1", tier="core", status="active")
        core_id = next(m["id"] for m in core if "[Consolidated]" in m["content"])

        for i in range(8):
            row = storage.get_memory(f"m-cluster-{i}")
            assert row is not None
            assert row["status"] == "active"
            assert row["tier"] == "buffer"
            assert row["metadata"]["consolidated_into"] == core_id
            assert row["metadata"].get("consolidated_at")


class TestSingletonPromotion:
    """(b) singleton importance >= 0.6 promoted in place, no new row."""

    def test_promoted_in_place_same_id(self, storage):
        _build_mixed_buffer(storage)
        engine = _engine(storage)
        stats = engine.consolidate()

        row = storage.get_memory("m-promote")
        assert row is not None
        assert row["tier"] == "core"
        assert row["status"] == "active"
        assert row["metadata"]["promoted_from_buffer"] is True
        assert stats["promoted"] == 1

    def test_no_duplicate_row_created_for_promotion(self, storage):
        _build_mixed_buffer(storage)
        engine = _engine(storage)
        engine.consolidate()

        # Only the singleton's own (promoted) id should exist with its content —
        # no second copy under a new id.
        core = storage.get_memories("c1", tier="core", status="active")
        promoted_matches = [m for m in core if m["content"] == "an important singleton memory"]
        assert len(promoted_matches) == 1
        assert promoted_matches[0]["id"] == "m-promote"


class TestSingletonBelowThreshold:
    """(c) singleton importance < 0.6 gets metadata.consolidation_seen."""

    def test_marked_seen_not_moved(self, storage):
        _build_mixed_buffer(storage)
        engine = _engine(storage)
        stats = engine.consolidate()

        row = storage.get_memory("m-seen")
        assert row is not None
        assert row["tier"] == "buffer"
        assert row["status"] == "active"
        assert row["metadata"]["consolidation_seen"] is True
        assert stats["seen"] == 1


class TestNeedsConsolidationAfterPass:
    """(d) needs_consolidation() False after a pass even though the plain
    buffer count is still >= threshold — only unconsolidated rows count."""

    def test_needs_consolidation_false_despite_high_plain_count(self, storage):
        _build_mixed_buffer(storage)
        engine = _engine(storage, threshold=5)
        engine.consolidate()

        plain_count = storage.count_memories("c1", tier="buffer")
        assert plain_count >= 5  # 8 kept-cluster rows + 1 seen row = 9, still >= threshold
        assert engine.needs_consolidation() is False


class TestSecondPassIsNoOp:
    """(e) a second consolidate() call over the same rows is a no-op."""

    def test_second_call_is_noop(self, storage):
        _build_mixed_buffer(storage)
        engine = _engine(storage)
        engine.consolidate()

        second = engine.consolidate()
        assert second == {
            "clusters": 0,
            "summarized": 0,
            "created": 0,
            "archived": 0,
            "kept": 0,
            "promoted": 0,
            "seen": 0,
            "llm_calls": 0,
        }


class TestDrainTerminates:
    """(f) drain() terminates rather than looping forever."""

    def test_drain_terminates(self, storage):
        _build_mixed_buffer(storage)
        engine = _engine(storage, threshold=5)
        totals = engine.drain(max_chunks=5)

        assert totals["passes"] >= 1
        assert engine.needs_consolidation() is False


class TestRetrievalAfterConsolidation:
    """(g) a kept source memory is still returned by retrieval by exact
    distinctive phrase after consolidation."""

    def test_kept_source_retrievable(self):
        engine = make_test_engine()
        char = engine.create_character("Watcher")
        char.consolidator.keep_sources = True

        source_ids = []
        for i in range(12):
            mem = char.memory.add(
                f"the lighthouse keeper mentioned a rusty brass key today note {i}",
                tier="buffer",
            )
            source_ids.append(mem["id"])

        stats = char.consolidator.consolidate()
        assert stats["kept"] > 0

        # At least one of the original source rows must still be active and
        # findable by an exact distinctive phrase from its own content.
        hits = char.retriever.retrieve("rusty brass key mentioned lighthouse keeper", limit=5)
        hit_ids = {m["id"] for m in hits}
        assert hit_ids & set(source_ids), (
            f"expected a kept source in {hit_ids}, sources were {source_ids}"
        )
        # And it really is a still-active buffer source, not just the summary.
        kept_hit = next(m for m in hits if m["id"] in source_ids)
        assert kept_hit["status"] == "active"
        assert kept_hit["metadata"].get("consolidated_into")


class TestKeepSourcesFalseMatchesLegacyBehavior:
    """(h) keep_sources=False reproduces the pre-Tier-3c archive-everything
    behavior byte-for-byte.

    The existing suites `tests/test_consolidation.py::
    test_consolidate_archives_originals` and `tests/test_character_
    integration.py::test_consolidate_compresses_buffer` assert archival
    counts directly — both were adapted to pass `keep_sources=False`
    explicitly (constructor kwarg / `char.consolidator.keep_sources = False`)
    so their original intent is preserved under the new True default. This
    test exercises the same mixed-buffer fixture used by the keep_sources=True
    tests above, with the flag off, as the direct A/B comparison.
    """

    def test_cluster_archived_not_kept(self, storage):
        _build_mixed_buffer(storage)
        engine = _engine(storage, keep_sources=False)
        stats = engine.consolidate()

        assert stats["kept"] == 0
        # 8 cluster sources + the promoted singleton's original row, both
        # archived under the legacy path (see test_singleton_promotion_
        # copies_and_archives for the latter in isolation).
        assert stats["archived"] == 9
        for i in range(8):
            row = storage.get_memory(f"m-cluster-{i}")
            assert row is not None
            assert row["status"] == "archived"
            assert "consolidated_into" not in (row["metadata"] or {})

    def test_singleton_promotion_copies_and_archives(self, storage):
        _build_mixed_buffer(storage)
        engine = _engine(storage, keep_sources=False)
        stats = engine.consolidate()

        assert stats["promoted"] == 0
        original = storage.get_memory("m-promote")
        assert original is not None
        assert original["status"] == "archived"
        assert original["tier"] == "buffer"

        core = storage.get_memories("c1", tier="core", status="active")
        copies = [m for m in core if m["content"] == "an important singleton memory"]
        assert len(copies) == 1
        assert copies[0]["id"] != "m-promote"

    def test_low_importance_singleton_untouched(self, storage):
        _build_mixed_buffer(storage)
        engine = _engine(storage, keep_sources=False)
        stats = engine.consolidate()

        assert stats["seen"] == 0
        row = storage.get_memory("m-seen")
        assert row is not None
        assert row["status"] == "active"
        assert row["tier"] == "buffer"
        assert "consolidation_seen" not in (row["metadata"] or {})
