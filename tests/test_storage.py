"""Tests for SQLite storage backend."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from woven_imprint.storage.sqlite import _SCHEMA, SQLiteStorage


@pytest.fixture
def storage():
    s = SQLiteStorage(":memory:")
    yield s
    s.close()


class TestCharacterCRUD:
    def test_save_and_load(self, storage):
        storage.save_character("c1", "Alice", {"hard": {"name": "Alice"}})
        char = storage.load_character("c1")
        assert char is not None
        assert char["name"] == "Alice"
        assert char["persona"]["hard"]["name"] == "Alice"

    def test_load_nonexistent(self, storage):
        assert storage.load_character("nope") is None

    def test_list_characters(self, storage):
        storage.save_character("c1", "Alice", {})
        storage.save_character("c2", "Bob", {})
        chars = storage.list_characters()
        assert len(chars) == 2
        names = {c["name"] for c in chars}
        assert names == {"Alice", "Bob"}

    def test_delete_character(self, storage):
        storage.save_character("c1", "Alice", {})
        storage.delete_character("c1")
        assert storage.load_character("c1") is None

    def test_delete_character_removes_session_turns(self, storage):
        storage.save_character("c1", "Alice", {})
        storage.add_session_turn("s1", "c1", 1, "user", "hi")
        storage.add_session_turn("s1", "c1", 2, "assistant", "hello")
        assert storage.get_session_turns("s1") != []

        storage.delete_character("c1")

        assert storage.get_session_turns("s1") == []

    def test_upsert(self, storage):
        storage.save_character("c1", "Alice", {"v": 1})
        storage.save_character("c1", "Alice Updated", {"v": 2})
        char = storage.load_character("c1")
        assert char["name"] == "Alice Updated"


class TestMemoryCRUD:
    def test_save_and_get(self, storage):
        storage.save_character("c1", "Alice", {})
        mem = {
            "id": "m1",
            "character_id": "c1",
            "tier": "buffer",
            "content": "Hello world",
            "importance": 0.7,
        }
        storage.save_memory(mem)
        result = storage.get_memory("m1")
        assert result is not None
        assert result["content"] == "Hello world"
        assert result["importance"] == 0.7
        assert result["tier"] == "buffer"

    def test_get_memories_filtered(self, storage):
        storage.save_character("c1", "Alice", {})
        storage.save_memory({"id": "m1", "character_id": "c1", "tier": "buffer", "content": "a"})
        storage.save_memory({"id": "m2", "character_id": "c1", "tier": "core", "content": "b"})
        storage.save_memory({"id": "m3", "character_id": "c1", "tier": "buffer", "content": "c"})

        buffer = storage.get_memories("c1", tier="buffer")
        assert len(buffer) == 2
        core = storage.get_memories("c1", tier="core")
        assert len(core) == 1

    def test_count_memories(self, storage):
        storage.save_character("c1", "Alice", {})
        for i in range(5):
            storage.save_memory(
                {
                    "id": f"m{i}",
                    "character_id": "c1",
                    "tier": "buffer",
                    "content": f"memory {i}",
                }
            )
        assert storage.count_memories("c1") == 5
        assert storage.count_memories("c1", tier="buffer") == 5
        assert storage.count_memories("c1", tier="core") == 0

    def test_update_status(self, storage):
        storage.save_character("c1", "Alice", {})
        storage.save_memory({"id": "m1", "character_id": "c1", "tier": "buffer", "content": "x"})
        storage.update_memory_status("m1", "archived")
        mem = storage.get_memory("m1")
        assert mem["status"] == "archived"
        # Archived memories excluded from default get_memories
        assert len(storage.get_memories("c1")) == 0

    def test_certainty_update(self, storage):
        storage.save_character("c1", "Alice", {})
        storage.save_memory(
            {
                "id": "m1",
                "character_id": "c1",
                "tier": "core",
                "content": "x",
                "certainty": 0.5,
            }
        )
        new_val = storage.update_memory_certainty("m1", 0.15)
        assert abs(new_val - 0.65) < 0.01

        # Clamp to 1.0
        storage.update_memory_certainty("m1", 10.0)
        mem = storage.get_memory("m1")
        assert mem["certainty"] == 1.0

    def test_fts_search(self, storage):
        storage.save_character("c1", "Alice", {})
        storage.save_memory(
            {
                "id": "m1",
                "character_id": "c1",
                "tier": "buffer",
                "content": "The cat sat on the mat",
            }
        )
        storage.save_memory(
            {
                "id": "m2",
                "character_id": "c1",
                "tier": "buffer",
                "content": "Dogs are loyal companions",
            }
        )
        results = storage.fts_search("c1", "cat mat")
        assert len(results) >= 1
        assert results[0]["id"] == "m1"

    def test_embedding_roundtrip(self, storage):
        storage.save_character("c1", "Alice", {})
        vec = [0.1, 0.2, 0.3, 0.4]
        storage.save_memory(
            {
                "id": "m1",
                "character_id": "c1",
                "tier": "buffer",
                "content": "test",
                "embedding": vec,
            }
        )
        mem = storage.get_memory("m1")
        assert mem["embedding"] is not None
        assert len(mem["embedding"]) == 4
        assert abs(mem["embedding"][0] - 0.1) < 0.001

    def test_deterministic_memory_ordering(self, storage):
        """Regression test: memories with same created_at should order by rowid DESC.

        This ensures that when memories are inserted in rapid succession (same second),
        the newest-inserted memory comes first, even though created_at is identical.
        This is critical for deterministic ordering without relying on index fragmentation.
        """
        storage.save_character("c1", "Alice", {})

        # Insert 5 memories in rapid succession (will have same created_at)
        memory_ids = []
        for i in range(5):
            mem_id = f"m{i}"
            memory_ids.append(mem_id)
            storage.save_memory(
                {
                    "id": mem_id,
                    "character_id": "c1",
                    "tier": "buffer",
                    "content": f"memory {i}",
                }
            )

        # Get memories back and verify order is reverse-insertion (newest first)
        mems = storage.get_memories("c1")
        assert len(mems) == 5

        # Order should be m4, m3, m2, m1, m0 (newest inserted first)
        retrieved_ids = [m["id"] for m in mems]
        expected_order = list(reversed(memory_ids))
        assert retrieved_ids == expected_order, (
            f"Expected order {expected_order}, got {retrieved_ids}. "
            "Memories with identical created_at must order by rowid DESC."
        )


class TestRelationshipCRUD:
    def test_save_and_get(self, storage):
        storage.save_character("c1", "Alice", {})
        rel = {
            "id": "r1",
            "character_id": "c1",
            "target_id": "user1",
            "dimensions": {"trust": 0.5, "affection": 0.3},
        }
        storage.save_relationship(rel)
        result = storage.get_relationship("c1", "user1")
        assert result is not None
        assert result["dimensions"]["trust"] == 0.5

    def test_get_all_relationships(self, storage):
        storage.save_character("c1", "Alice", {})
        for i, target in enumerate(["user1", "user2", "bob"]):
            storage.save_relationship(
                {
                    "id": f"r{i}",
                    "character_id": "c1",
                    "target_id": target,
                    "dimensions": {"trust": 0.0},
                }
            )
        rels = storage.get_relationships("c1")
        assert len(rels) == 3


class TestMigrations:
    def test_v3_session_turns_applies_cleanly_on_v2_db(self):
        """A DB pinned at schema version 2 (pre-session_turns) must pick up
        the v3 migration — creating session_turns — the next time it's
        opened via SQLiteStorage, without losing existing data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "v2.db"

            # Build a raw "v2" database: full base schema (includes the
            # sessions.alias column added by migration 2) but pinned at
            # schema_version=2, with no session_turns table yet.
            conn = sqlite3.connect(str(db_path))
            conn.executescript(_SCHEMA)
            conn.execute("ALTER TABLE sessions ADD COLUMN alias TEXT;")
            conn.execute("DELETE FROM schema_version")
            conn.execute("INSERT INTO schema_version (version) VALUES (2)")
            conn.execute(
                "INSERT INTO characters (id, name, persona) VALUES (?, ?, ?)",
                ("c1", "Alice", "{}"),
            )
            conn.commit()
            conn.close()

            # Confirm session_turns doesn't exist yet in the v2 DB.
            conn = sqlite3.connect(str(db_path))
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert "session_turns" not in tables
            conn.close()

            # Reopen via SQLiteStorage — migration 3 should apply cleanly.
            storage = SQLiteStorage(str(db_path))
            try:
                version = storage._conn.execute(
                    "SELECT MAX(version) FROM schema_version"
                ).fetchone()[0]
                assert version == 3

                # Pre-existing data survived the migration.
                assert storage.load_character("c1")["name"] == "Alice"

                # session_turns is now usable.
                storage.add_session_turn("s1", "c1", 1, "user", "hello")
                turns = storage.get_session_turns("s1")
                assert len(turns) == 1
                assert turns[0]["content"] == "hello"
            finally:
                storage.close()
