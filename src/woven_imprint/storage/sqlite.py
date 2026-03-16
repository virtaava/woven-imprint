"""SQLite storage backend — local-first, zero dependency."""

from __future__ import annotations

import json
import sqlite3
import struct
from pathlib import Path
from typing import Any


_SCHEMA = """
CREATE TABLE IF NOT EXISTS characters (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    persona JSON NOT NULL,
    birthdate TEXT,
    state JSON DEFAULT '{}',
    created_at DATETIME DEFAULT (datetime('now')),
    updated_at DATETIME DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    character_id TEXT NOT NULL REFERENCES characters(id),
    tier TEXT NOT NULL CHECK(tier IN ('buffer', 'core', 'bedrock')),
    content TEXT NOT NULL,
    embedding BLOB,
    importance REAL DEFAULT 0.5,
    certainty REAL DEFAULT 1.0,
    status TEXT DEFAULT 'active' CHECK(status IN ('active', 'contradicted', 'archived')),
    source_refs JSON DEFAULT '[]',
    session_id TEXT,
    role TEXT,
    metadata JSON DEFAULT '{}',
    created_at DATETIME DEFAULT (datetime('now')),
    accessed_at DATETIME DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS relationships (
    id TEXT PRIMARY KEY,
    character_id TEXT NOT NULL REFERENCES characters(id),
    target_id TEXT NOT NULL,
    dimensions JSON NOT NULL,
    power_balance REAL DEFAULT 0.0,
    type TEXT DEFAULT 'stranger',
    trajectory TEXT DEFAULT 'stable',
    key_moments JSON DEFAULT '[]',
    formed_at DATETIME DEFAULT (datetime('now')),
    last_interaction DATETIME DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    character_id TEXT NOT NULL REFERENCES characters(id),
    summary TEXT,
    started_at DATETIME DEFAULT (datetime('now')),
    ended_at DATETIME
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, character_id UNINDEXED, tier UNINDEXED,
    content='memories', content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, character_id, tier)
    VALUES (new.rowid, new.content, new.character_id, new.tier);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, character_id, tier)
    VALUES ('delete', old.rowid, old.content, old.character_id, old.tier);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE OF content ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, character_id, tier)
    VALUES ('delete', old.rowid, old.content, old.character_id, old.tier);
    INSERT INTO memories_fts(rowid, content, character_id, tier)
    VALUES (new.rowid, new.content, new.character_id, new.tier);
END;

CREATE INDEX IF NOT EXISTS idx_memories_character ON memories(character_id, tier, status);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);
CREATE INDEX IF NOT EXISTS idx_relationships_character ON relationships(character_id);

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at DATETIME DEFAULT (datetime('now'))
);
INSERT OR IGNORE INTO schema_version (version) VALUES (1);
"""

# Future migrations go here: version → SQL
_MIGRATIONS: dict[int, str] = {
    # 2: "ALTER TABLE characters ADD COLUMN ...",
}


def _serialize_embedding(vec: list[float]) -> bytes:
    """Pack float list into compact binary."""
    return struct.pack(f"{len(vec)}f", *vec)


def _deserialize_embedding(blob: bytes) -> list[float]:
    """Unpack binary into float list."""
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


class SQLiteStorage:
    """SQLite-backed storage for characters, memories, relationships."""

    def __init__(self, db_path: str | Path = ":memory:"):
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript(_SCHEMA)
        self._run_migrations()

    def _run_migrations(self) -> None:
        """Apply pending schema migrations."""
        try:
            row = self._conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
            current = row[0] if row and row[0] else 1
        except Exception:
            current = 1

        for version in sorted(_MIGRATIONS.keys()):
            if version > current:
                self._conn.executescript(_MIGRATIONS[version])
                self._conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))
                self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ── Characters ──────────────────────────────────────────────

    def save_character(
        self,
        char_id: str,
        name: str,
        persona: dict,
        birthdate: str | None = None,
        state: dict | None = None,
    ) -> None:
        self._conn.execute(
            """INSERT INTO characters (id, name, persona, birthdate, state)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   name=excluded.name, persona=excluded.persona,
                   birthdate=excluded.birthdate, state=excluded.state,
                   updated_at=datetime('now')""",
            (char_id, name, json.dumps(persona), birthdate, json.dumps(state or {})),
        )
        self._conn.commit()

    def load_character(self, char_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM characters WHERE id = ?", (char_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["persona"] = json.loads(d["persona"])
        d["state"] = json.loads(d["state"])
        return d

    def list_characters(self) -> list[dict]:
        rows = self._conn.execute("SELECT id, name, created_at FROM characters").fetchall()
        return [dict(r) for r in rows]

    def delete_character(self, char_id: str) -> None:
        self._conn.execute("DELETE FROM memories WHERE character_id = ?", (char_id,))
        self._conn.execute("DELETE FROM relationships WHERE character_id = ?", (char_id,))
        self._conn.execute("DELETE FROM sessions WHERE character_id = ?", (char_id,))
        self._conn.execute("DELETE FROM characters WHERE id = ?", (char_id,))
        self._conn.commit()

    # ── Memories ────────────────────────────────────────────────

    def save_memory(self, memory: dict) -> None:
        """Save a memory dict. Must have: id, character_id, tier, content."""
        emb = memory.get("embedding")
        emb_blob = _serialize_embedding(emb) if emb else None
        self._conn.execute(
            """INSERT INTO memories
               (id, character_id, tier, content, embedding, importance, certainty,
                status, source_refs, session_id, role, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   content=excluded.content, embedding=excluded.embedding,
                   importance=excluded.importance, certainty=excluded.certainty,
                   status=excluded.status, source_refs=excluded.source_refs,
                   metadata=excluded.metadata, accessed_at=datetime('now')""",
            (
                memory["id"],
                memory["character_id"],
                memory["tier"],
                memory["content"],
                emb_blob,
                memory.get("importance", 0.5),
                memory.get("certainty", 1.0),
                memory.get("status", "active"),
                json.dumps(memory.get("source_refs", [])),
                memory.get("session_id"),
                memory.get("role"),
                json.dumps(memory.get("metadata", {})),
            ),
        )
        self._conn.commit()

    def get_memories(
        self, character_id: str, tier: str | None = None, status: str = "active", limit: int = 1000
    ) -> list[dict]:
        """Retrieve memories for a character, optionally filtered by tier."""
        q = "SELECT * FROM memories WHERE character_id = ? AND status = ?"
        params: list[Any] = [character_id, status]
        if tier:
            q += " AND tier = ?"
            params.append(tier)
        q += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self._conn.execute(q, params).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def get_memory(self, memory_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return self._row_to_memory(row) if row else None

    def update_memory_status(
        self, memory_id: str, status: str, certainty: float | None = None
    ) -> None:
        if certainty is not None:
            self._conn.execute(
                "UPDATE memories SET status = ?, certainty = ? WHERE id = ?",
                (status, certainty, memory_id),
            )
        else:
            self._conn.execute(
                "UPDATE memories SET status = ? WHERE id = ?",
                (status, memory_id),
            )
        self._conn.commit()

    def update_memory_certainty(self, memory_id: str, delta: float) -> float:
        """Adjust certainty by delta, clamp to [0, 1]. Returns new value."""
        row = self._conn.execute(
            "SELECT certainty FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if not row:
            return 0.0
        new_val = max(0.0, min(1.0, row["certainty"] + delta))
        self._conn.execute(
            "UPDATE memories SET certainty = ? WHERE id = ?",
            (new_val, memory_id),
        )
        self._conn.commit()
        return new_val

    def touch_memory(self, memory_id: str) -> None:
        """Update accessed_at timestamp."""
        self._conn.execute(
            "UPDATE memories SET accessed_at = datetime('now') WHERE id = ?",
            (memory_id,),
        )
        self._conn.commit()

    def touch_memories_batch(self, memory_ids: list[str]) -> None:
        """Update accessed_at for multiple memories in one transaction."""
        if not memory_ids:
            return
        self._conn.executemany(
            "UPDATE memories SET accessed_at = datetime('now') WHERE id = ?",
            [(mid,) for mid in memory_ids],
        )
        self._conn.commit()

    def count_memories(self, character_id: str, tier: str | None = None) -> int:
        q = "SELECT COUNT(*) as c FROM memories WHERE character_id = ? AND status = 'active'"
        params: list[Any] = [character_id]
        if tier:
            q += " AND tier = ?"
            params.append(tier)
        return self._conn.execute(q, params).fetchone()["c"]

    def fts_search(self, character_id: str, query: str, limit: int = 50) -> list[dict]:
        """Full-text search using FTS5 (BM25 ranking).

        Query is sanitized to prevent FTS5 operator injection.
        """
        # Sanitize: strip FTS5 operators, wrap each word in quotes
        import re

        words = re.findall(r"\w+", query)
        if not words:
            return []
        safe_query = " OR ".join(f'"{w}"' for w in words[:20])

        rows = self._conn.execute(
            """SELECT m.*, rank FROM memories_fts
               JOIN memories m ON memories_fts.rowid = m.rowid
               WHERE memories_fts MATCH ? AND m.character_id = ? AND m.status = 'active'
               ORDER BY rank LIMIT ?""",
            (safe_query, character_id, limit),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def _row_to_memory(self, row: sqlite3.Row) -> dict:
        d = dict(row)
        if d.get("embedding"):
            d["embedding"] = _deserialize_embedding(d["embedding"])
        d["source_refs"] = json.loads(d.get("source_refs") or "[]")
        d["metadata"] = json.loads(d.get("metadata") or "{}")
        return d

    # ── Relationships ───────────────────────────────────────────

    def save_relationship(self, rel: dict) -> None:
        self._conn.execute(
            """INSERT INTO relationships
               (id, character_id, target_id, dimensions, power_balance, type,
                trajectory, key_moments)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   dimensions=excluded.dimensions, power_balance=excluded.power_balance,
                   type=excluded.type, trajectory=excluded.trajectory,
                   key_moments=excluded.key_moments, last_interaction=datetime('now')""",
            (
                rel["id"],
                rel["character_id"],
                rel["target_id"],
                json.dumps(rel["dimensions"]),
                rel.get("power_balance", 0.0),
                rel.get("type", "stranger"),
                rel.get("trajectory", "stable"),
                json.dumps(rel.get("key_moments", [])),
            ),
        )
        self._conn.commit()

    def get_relationship(self, character_id: str, target_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM relationships WHERE character_id = ? AND target_id = ?",
            (character_id, target_id),
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["dimensions"] = json.loads(d["dimensions"])
        d["key_moments"] = json.loads(d["key_moments"])
        return d

    def get_relationships(self, character_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM relationships WHERE character_id = ?",
            (character_id,),
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["dimensions"] = json.loads(d["dimensions"])
            d["key_moments"] = json.loads(d["key_moments"])
            result.append(d)
        return result

    # ── Sessions ────────────────────────────────────────────────

    def save_session(self, session: dict) -> None:
        self._conn.execute(
            """INSERT INTO sessions (id, character_id, summary)
               VALUES (?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET
                   summary=excluded.summary, ended_at=datetime('now')""",
            (session["id"], session["character_id"], session.get("summary")),
        )
        self._conn.commit()

    def get_sessions(self, character_id: str, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM sessions WHERE character_id = ? ORDER BY started_at DESC LIMIT ?",
            (character_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]
