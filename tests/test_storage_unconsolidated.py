"""Tests for the `unconsolidated` buffer filter and `update_memory_metadata` (Tier 3c Task 1)."""

import pytest

from tests.helpers import FakeEmbedder
from woven_imprint.config import WovenConfig, reload_config
from woven_imprint.memory.store import MemoryStore
from woven_imprint.storage.sqlite import SQLiteStorage


@pytest.fixture
def storage():
    s = SQLiteStorage(":memory:")
    s.save_character("c1", "Alice", {})
    yield s
    s.close()


def _save(storage, mem_id, metadata=None):
    mem = {
        "id": mem_id,
        "character_id": "c1",
        "tier": "buffer",
        "content": f"content {mem_id}",
    }
    if metadata is not None:
        mem["metadata"] = metadata
    storage.save_memory(mem)


class TestUnconsolidatedFilter:
    def test_five_rows_all_count_without_filter(self, storage):
        _save(storage, "m-none")  # no metadata key passed -> stored as '{}'
        _save(storage, "m-empty", {})
        _save(storage, "m-consolidated", {"consolidated_into": "mem-x"})
        _save(storage, "m-seen", {"consolidation_seen": True})
        _save(storage, "m-other", {"pinned": True})

        assert storage.count_memories("c1", tier="buffer") == 5

    def test_unconsolidated_count_excludes_consolidated_and_seen(self, storage):
        _save(storage, "m-none")
        _save(storage, "m-empty", {})
        _save(storage, "m-consolidated", {"consolidated_into": "mem-x"})
        _save(storage, "m-seen", {"consolidation_seen": True})
        _save(storage, "m-other", {"pinned": True})

        assert storage.count_memories("c1", tier="buffer", unconsolidated=True) == 3

    def test_unconsolidated_get_memories_returns_expected_rows(self, storage):
        _save(storage, "m-none")
        _save(storage, "m-empty", {})
        _save(storage, "m-consolidated", {"consolidated_into": "mem-x"})
        _save(storage, "m-seen", {"consolidation_seen": True})
        _save(storage, "m-other", {"pinned": True})

        rows = storage.get_memories("c1", tier="buffer", unconsolidated=True)
        ids = {r["id"] for r in rows}
        assert ids == {"m-none", "m-empty", "m-other"}

    def test_unconsolidated_default_false_is_byte_identical(self, storage):
        _save(storage, "m-none")
        _save(storage, "m-consolidated", {"consolidated_into": "mem-x"})

        # Without the kwarg, behavior must be unchanged: both rows returned/counted.
        assert storage.count_memories("c1", tier="buffer") == 2
        assert len(storage.get_memories("c1", tier="buffer")) == 2


class TestUpdateMemoryMetadata:
    def test_merges_and_preserves_other_keys(self, storage):
        _save(storage, "m1", {"pinned": True})
        merged = storage.update_memory_metadata("m1", {"consolidated_into": "mem-core-1"})
        assert merged == {"pinned": True, "consolidated_into": "mem-core-1"}
        # Persisted.
        row = storage.get_memory("m1")
        assert row is not None
        assert row["metadata"] == {"pinned": True, "consolidated_into": "mem-core-1"}

    def test_none_value_removes_key(self, storage):
        _save(storage, "m1", {"pinned": True, "consolidated_into": "mem-core-1"})
        merged = storage.update_memory_metadata("m1", {"consolidated_into": None})
        assert merged == {"pinned": True}
        row = storage.get_memory("m1")
        assert row is not None
        assert row["metadata"] == {"pinned": True}

    def test_starts_from_null_or_empty_metadata(self, storage):
        _save(storage, "m1")  # stored as '{}'
        merged = storage.update_memory_metadata("m1", {"consolidation_seen": True})
        assert merged == {"consolidation_seen": True}

    def test_unknown_memory_id_raises_key_error(self, storage):
        with pytest.raises(KeyError):
            storage.update_memory_metadata("nope", {"x": 1})


class TestConsolidationKeepSourcesConfig:
    def test_default_is_true(self):
        cfg = WovenConfig()
        assert cfg.memory.consolidation_keep_sources is True

    def test_yaml_override_false(self, tmp_path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("memory:\n  consolidation_keep_sources: false\n")
        try:
            cfg = reload_config(str(cfg_file))
            assert cfg.memory.consolidation_keep_sources is False
        finally:
            reload_config()


class TestMemoryStoreUnconsolidated:
    def test_count_passthrough(self, storage):
        _save(storage, "m-none")
        _save(storage, "m-consolidated", {"consolidated_into": "mem-x"})
        store = MemoryStore(storage, FakeEmbedder(), "c1")

        assert store.count(tier="buffer") == 2
        assert store.count(tier="buffer", unconsolidated=True) == 1

    def test_needs_consolidation_ignores_consolidated_rows(self, storage):
        from woven_imprint.config import get_config

        # Use a low threshold so the test is cheap; restore afterward (shared global config).
        cfg = get_config()
        original_threshold = cfg.memory.consolidation_threshold
        cfg.memory.consolidation_threshold = 2
        try:
            store = MemoryStore(storage, FakeEmbedder(), "c1")
            _save(storage, "m1", {"consolidated_into": "mem-x"})
            _save(storage, "m2", {"consolidation_seen": True})
            # Two buffer rows total, but both already consolidated/seen.
            assert store.count(tier="buffer") == 2
            assert store.needs_consolidation(threshold=cfg.memory.consolidation_threshold) is False

            _save(storage, "m3")  # a genuinely unconsolidated row
            _save(storage, "m4")
            assert store.needs_consolidation(threshold=cfg.memory.consolidation_threshold) is True
        finally:
            cfg.memory.consolidation_threshold = original_threshold

    def test_needs_consolidation_honors_keep_sources_flag(self, storage):
        """MemoryStore.needs_consolidation must match ConsolidationEngine.needs_consolidation:
        with consolidation_keep_sources off, it's a plain buffer count that does NOT ignore
        consolidated/seen rows."""
        from woven_imprint.config import get_config

        cfg = get_config()
        original_keep = cfg.memory.consolidation_keep_sources
        try:
            store = MemoryStore(storage, FakeEmbedder(), "c1")
            _save(storage, "m1", {"consolidated_into": "mem-x"})
            _save(storage, "m2", {"consolidation_seen": True})

            # keep_sources=True (default): both rows are already consolidated/seen, so they
            # don't count toward the threshold.
            cfg.memory.consolidation_keep_sources = True
            assert store.needs_consolidation(threshold=2) is False

            # keep_sources=False: plain count, both rows count regardless of metadata.
            cfg.memory.consolidation_keep_sources = False
            assert store.needs_consolidation(threshold=2) is True
        finally:
            cfg.memory.consolidation_keep_sources = original_keep
