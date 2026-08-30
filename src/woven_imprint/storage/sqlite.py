"""SQLite storage backend — local-first, zero dependency."""

from __future__ import annotations

import json
import sqlite3
import struct
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..clock import sqlite_ts


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
CREATE INDEX IF NOT EXISTS idx_memories_accessed ON memories(character_id, accessed_at);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(character_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_relationships_character ON relationships(character_id);
CREATE INDEX IF NOT EXISTS idx_relationships_pair ON relationships(character_id, target_id);

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at DATETIME DEFAULT (datetime('now'))
);
INSERT OR IGNORE INTO schema_version (version) VALUES (1);
"""

# Future migrations go here: version → SQL
_MIGRATIONS: dict[int, str] = {
    2: "ALTER TABLE sessions ADD COLUMN alias TEXT;",
    3: """
CREATE TABLE IF NOT EXISTS session_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    character_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at DATETIME DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_session_turns_session ON session_turns(session_id, seq);
""",
    4: """
CREATE TABLE IF NOT EXISTS callbacks (
    id TEXT PRIMARY KEY,
    character_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK(kind IN ('open_thread', 'callback', 'milestone', 'curiosity')),
    hook TEXT NOT NULL,
    source_memory_ids JSON DEFAULT '[]',
    salience REAL DEFAULT 0.5,
    status TEXT DEFAULT 'ready' CHECK(status IN ('ready', 'consumed', 'expired')),
    created_at DATETIME DEFAULT (datetime('now')),
    consumed_at DATETIME
);
CREATE INDEX IF NOT EXISTS idx_callbacks_character ON callbacks(character_id, status, salience);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
""",
    5: """
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    character_id TEXT NOT NULL REFERENCES characters(id) ON DELETE CASCADE,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    object TEXT NOT NULL,
    statement TEXT NOT NULL,
    event_time TEXT,
    valid_from TEXT NOT NULL,
    valid_to TEXT,
    recorded_at TEXT NOT NULL,
    expired_at TEXT,
    certainty REAL DEFAULT 1.0,
    importance REAL DEFAULT 0.75,
    memory_id TEXT,
    superseded_by TEXT,
    session_id TEXT,
    user_id TEXT,
    metadata TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_facts_key ON facts(character_id, subject, predicate, valid_to);
CREATE INDEX IF NOT EXISTS idx_facts_recorded ON facts(character_id, recorded_at DESC);
ALTER TABLE relationships ADD COLUMN state TEXT DEFAULT '{}';
""",
}


def _serialize_embedding(vec: list[float]) -> bytes:
    """Pack float list into compact binary."""
    return struct.pack(f"{len(vec)}f", *vec)


def _deserialize_embedding(blob: bytes) -> list[float]:
    """Unpack binary into float list."""
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


class SQLiteStorage:
    """SQLite-backed storage for characters, memories, relationships.

    A single connection (``check_same_thread=False``) is shared between the
    caller thread and any BackgroundWorker threads (assessment, callbacks,
    etc). sqlite3's C-level statement handling is not safe to interleave
    across threads on one connection — concurrent execute/commit calls can
    corrupt cursor/transaction state (``cannot commit - no transaction is
    active``, ``InterfaceError``, even a raw ``SystemError``). ``self._lock``
    (an ``RLock``, reentrant so a method can call another locked method) is
    held across every unit of work that touches ``self._conn`` — reads
    included, since statement iteration races too. Per-thread connections are
    deliberately not used: this class also backs ``:memory:`` databases,
    which are private per-connection and would break under that model.
    """

    def __init__(self, db_path: str | Path = ":memory:"):
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._run_migrations()
            self.meta_set("schema_semver", "0.6.0-dev")

    def _run_migrations(self) -> None:
        """Apply pending schema migrations."""
        with self._lock:
            try:
                row = self._conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
                current = row[0] if row and row[0] else 1
            except Exception:
                current = 1

            for version in sorted(_MIGRATIONS.keys()):
                if version > current:
                    self._conn.executescript(_MIGRATIONS[version])
                    self._conn.execute(
                        "INSERT INTO schema_version (version) VALUES (?)", (version,)
                    )
                    self._commit()

    def _commit(self) -> None:
        """Commit the current transaction. Caller must hold self._lock."""
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
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
        state_json = json.dumps(state) if state is not None else None
        with self._lock:
            if state_json is not None:
                self._conn.execute(
                    """INSERT INTO characters (id, name, persona, birthdate, state)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(id) DO UPDATE SET
                           name=excluded.name, persona=excluded.persona,
                           birthdate=excluded.birthdate, state=excluded.state,
                           updated_at=datetime('now')""",
                    (char_id, name, json.dumps(persona), birthdate, state_json),
                )
            else:
                # Don't overwrite existing state when state param is not provided
                self._conn.execute(
                    """INSERT INTO characters (id, name, persona, birthdate, state)
                       VALUES (?, ?, ?, ?, '{}')
                       ON CONFLICT(id) DO UPDATE SET
                           name=excluded.name, persona=excluded.persona,
                           birthdate=excluded.birthdate,
                           updated_at=datetime('now')""",
                    (char_id, name, json.dumps(persona), birthdate),
                )
            self._commit()

    def load_character(self, char_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM characters WHERE id = ?", (char_id,)).fetchone()
            if not row:
                return None
            d = dict(row)
        d["persona"] = json.loads(d["persona"])
        d["state"] = json.loads(d["state"])
        return d

    def list_characters(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT id, name, created_at FROM characters").fetchall()
            return [dict(r) for r in rows]

    def delete_character(self, char_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM memories WHERE character_id = ?", (char_id,))
            self._conn.execute("DELETE FROM relationships WHERE character_id = ?", (char_id,))
            self._conn.execute("DELETE FROM session_turns WHERE character_id = ?", (char_id,))
            self._conn.execute("DELETE FROM callbacks WHERE character_id = ?", (char_id,))
            self._conn.execute("DELETE FROM sessions WHERE character_id = ?", (char_id,))
            self._conn.execute("DELETE FROM characters WHERE id = ?", (char_id,))
            self._commit()

    # ── Memories ────────────────────────────────────────────────

    def save_memory(self, memory: dict) -> None:
        """Save a memory dict. Must have: id, character_id, tier, content.

        `created_at`/`accessed_at` may be supplied as "YYYY-MM-DD HH:MM:SS" strings;
        otherwise both are stamped from the injectable clock. On conflict (id
        already exists), the upsert keeps the caller-supplied `accessed_at`
        (or the fresh stamp if none was given) — it is not automatically
        bumped to "now" independent of what's passed in.
        """
        emb = memory.get("embedding")
        emb_blob = _serialize_embedding(emb) if emb else None
        stamp = sqlite_ts()
        created_at = memory.get("created_at") or stamp
        accessed_at = memory.get("accessed_at") or stamp
        with self._lock:
            self._conn.execute(
                """INSERT INTO memories
                   (id, character_id, tier, content, embedding, importance, certainty,
                    status, source_refs, session_id, role, metadata, created_at, accessed_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       content=excluded.content, embedding=excluded.embedding,
                       importance=excluded.importance, certainty=excluded.certainty,
                       status=excluded.status, source_refs=excluded.source_refs,
                       metadata=excluded.metadata, accessed_at=excluded.accessed_at""",
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
                    created_at,
                    accessed_at,
                ),
            )
            self._commit()

    def get_memories(
        self,
        character_id: str,
        tier: str | None = None,
        status: str = "active",
        limit: int | None = 1000,
        oldest_first: bool = False,
        unconsolidated: bool = False,
        exclude_consolidated: bool = False,
    ) -> list[dict]:
        """Retrieve memories for a character, optionally filtered by tier.

        When oldest_first=True, orders by created_at ASC instead of DESC,
        ensuring the LIMIT captures the oldest rows (useful for TTL cleanup).

        `limit=None` returns all matching rows (no LIMIT clause) — the
        `ORDER BY` is still applied for deterministic ordering.

        `unconsolidated=True` additionally excludes rows already claimed by
        consolidation (`metadata.consolidated_into` or `metadata.consolidation_seen`
        set) — used by the consolidation threshold/chunk queries so re-consolidated
        rows aren't recounted. `json_extract` returns NULL for a missing key
        whether `metadata` is SQL NULL, `'{}'`, or any JSON lacking the key, so
        this is robust to all three storage shapes.

        `exclude_consolidated=True` excludes only rows with `metadata.consolidated_into`
        set — unlike `unconsolidated`, rows merely marked `consolidation_seen` are NOT
        excluded (they must stay eligible for e.g. TTL sweeps). Used by buffer hygiene
        so an oldest-first `LIMIT` window can't be entirely starved by kept consolidation
        sources (which would otherwise never age out of the window and block sweeping of
        genuinely stale, unrelated rows). Mutually independent of `unconsolidated` — pass
        at most one; both filter on the same column for different purposes.
        """
        q = "SELECT *, rowid FROM memories WHERE character_id = ? AND status = ?"
        params: list[Any] = [character_id, status]
        if tier:
            q += " AND tier = ?"
            params.append(tier)
        if unconsolidated:
            q += (
                " AND json_extract(metadata, '$.consolidated_into') IS NULL"
                " AND json_extract(metadata, '$.consolidation_seen') IS NULL"
            )
        if exclude_consolidated:
            q += " AND json_extract(metadata, '$.consolidated_into') IS NULL"
        if oldest_first:
            q += " ORDER BY created_at ASC, rowid ASC"
        else:
            q += " ORDER BY created_at DESC, rowid DESC"
        if limit is not None:
            q += " LIMIT ?"
            params.append(limit)
        with self._lock:
            rows = self._conn.execute(q, params).fetchall()
            return [self._row_to_memory(r) for r in rows]

    def iter_memory_rows_for_reembed(
        self, character_id: str, page: int = 500
    ) -> Iterator[list[dict]]:
        """Yield pages of up to `page` active memory rows for `character_id`,
        each row a plain dict with only `id`/`content`/`role`/`created_at`/
        `metadata` — the `embedding` column is never selected, so no vector
        is ever materialized by this path.

        Used by `MemoryStore.reembed()` to stream a full history without
        loading every row (embeddings included) into memory at once, the way
        `get_memories(..., limit=None)` does. Paged by rowid — keyset
        pagination (`WHERE rowid > <last id seen>`), not `OFFSET` — so a
        page's cost doesn't grow with how far into the table it is.
        """
        last_rowid = 0
        while True:
            with self._lock:
                rows = self._conn.execute(
                    """SELECT rowid, id, content, role, created_at, metadata FROM memories
                       WHERE character_id = ? AND status = 'active' AND rowid > ?
                       ORDER BY rowid ASC LIMIT ?""",
                    (character_id, last_rowid, page),
                ).fetchall()
            if not rows:
                return
            last_rowid = rows[-1]["rowid"]
            yield [
                {
                    "id": r["id"],
                    "content": r["content"],
                    "role": r["role"],
                    "created_at": r["created_at"],
                    "metadata": json.loads(r["metadata"] or "{}"),
                }
                for r in rows
            ]

    def get_memory(self, memory_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            return self._row_to_memory(row) if row else None

    def update_memory_status(
        self, memory_id: str, status: str, certainty: float | None = None
    ) -> None:
        with self._lock:
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
            self._commit()

    def update_memory_certainty(self, memory_id: str, delta: float) -> float:
        """Adjust certainty by delta, clamp to [0, 1]. Returns new value."""
        with self._lock:
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
            self._commit()
            return new_val

    _MEMORY_TIERS = ("buffer", "core", "bedrock")

    def update_memory_fields(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        embedding: list[float] | None = None,
        importance: float | None = None,
        tier: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Single UPDATE building only the provided SET clauses.

        `content` changes must come with a new `embedding` (re-embed on the
        caller side) — the two travel together or not at all.
        """
        sets: list[str] = []
        params: list[Any] = []
        if content is not None:
            if embedding is None:
                raise ValueError("content changes require a new embedding")
            sets += ["content = ?", "embedding = ?"]
            params += [content, _serialize_embedding(embedding)]
        elif embedding is not None:
            # Embedding-only update (no content change) — e.g. MemoryStore.reembed()
            # recomputing a vector from the current embed-text builder without
            # touching the stored content.
            sets.append("embedding = ?")
            params.append(_serialize_embedding(embedding))
        if importance is not None:
            sets.append("importance = ?")
            params.append(max(0.0, min(1.0, float(importance))))
        if tier is not None:
            if tier not in self._MEMORY_TIERS:
                raise ValueError(f"invalid tier {tier!r}")
            sets.append("tier = ?")
            params.append(tier)
        if metadata is not None:
            sets.append("metadata = ?")
            params.append(json.dumps(metadata))
        if not sets:
            return
        params.append(memory_id)
        with self._lock:
            self._conn.execute(f"UPDATE memories SET {', '.join(sets)} WHERE id = ?", params)
            self._commit()

    def update_memory_metadata(self, memory_id: str, patch: dict) -> dict:
        """Shallow-merge `patch` into a memory's metadata JSON and persist it.

        A `patch` value of `None` removes that key. Reads and writes under the
        storage lock so a concurrent metadata update can't clobber this merge.
        Returns the merged metadata dict. Raises `KeyError` if the memory
        doesn't exist.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT metadata FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            if row is None:
                raise KeyError(memory_id)
            current = json.loads(row["metadata"] or "{}")
            for key, value in patch.items():
                if value is None:
                    current.pop(key, None)
                else:
                    current[key] = value
            self._conn.execute(
                "UPDATE memories SET metadata = ? WHERE id = ?",
                (json.dumps(current), memory_id),
            )
            self._commit()
            return current

    def delete_memory(self, memory_id: str) -> bool:
        """Hard delete (the FTS trigger keeps the index in sync). Returns whether a row was removed."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self._commit()
            return cur.rowcount > 0

    def list_pinned_memories(self, character_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT *, rowid FROM memories WHERE character_id = ? AND status = 'active' "
                "AND json_extract(metadata, '$.pinned') = 1 ORDER BY created_at ASC, rowid ASC",
                (character_id,),
            ).fetchall()
            return [self._row_to_memory(r) for r in rows]

    def touch_memory(self, memory_id: str) -> None:
        """Update accessed_at timestamp."""
        with self._lock:
            self._conn.execute(
                "UPDATE memories SET accessed_at = ? WHERE id = ?",
                (sqlite_ts(), memory_id),
            )
            self._commit()

    def touch_memories_batch(self, memory_ids: list[str]) -> None:
        """Update accessed_at for multiple memories in one transaction."""
        if not memory_ids:
            return
        with self._lock:
            stamp = sqlite_ts()
            self._conn.executemany(
                "UPDATE memories SET accessed_at = ? WHERE id = ?",
                [(stamp, mid) for mid in memory_ids],
            )
            self._commit()

    def update_memory_embeddings_batch(self, pairs: list[tuple[str, list[float]]]) -> None:
        """Update just the `embedding` column for multiple memories in one transaction.

        Trivial batch variant of `update_memory_fields(id, embedding=...)` for
        `MemoryStore.reembed()` — one `executemany` per chunk instead of one
        UPDATE per row. `pairs` is `[(memory_id, embedding_vector), ...]`.
        """
        if not pairs:
            return
        with self._lock:
            self._conn.executemany(
                "UPDATE memories SET embedding = ? WHERE id = ?",
                [(_serialize_embedding(vec), mid) for mid, vec in pairs],
            )
            self._commit()

    def count_memories(
        self, character_id: str, tier: str | None = None, unconsolidated: bool = False
    ) -> int:
        """Count active memories, optionally filtered by tier.

        `unconsolidated=True` mirrors `get_memories(..., unconsolidated=True)` —
        see its docstring for the NULL/'{}'/missing-key handling.
        """
        q = "SELECT COUNT(*) as c FROM memories WHERE character_id = ? AND status = 'active'"
        params: list[Any] = [character_id]
        if tier:
            q += " AND tier = ?"
            params.append(tier)
        if unconsolidated:
            q += (
                " AND json_extract(metadata, '$.consolidated_into') IS NULL"
                " AND json_extract(metadata, '$.consolidation_seen') IS NULL"
            )
        with self._lock:
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

        with self._lock:
            rows = self._conn.execute(
                """SELECT m.*, m.rowid AS rowid, rank FROM memories_fts
                   JOIN memories m ON memories_fts.rowid = m.rowid
                   WHERE memories_fts MATCH ? AND m.character_id = ? AND m.status = 'active'
                   ORDER BY rank, m.rowid DESC LIMIT ?""",
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
        with self._lock:
            self._conn.execute(
                """INSERT INTO relationships
                   (id, character_id, target_id, dimensions, power_balance, type,
                    trajectory, key_moments, state)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       dimensions=excluded.dimensions, power_balance=excluded.power_balance,
                       type=excluded.type, trajectory=excluded.trajectory,
                       key_moments=excluded.key_moments, state=excluded.state,
                       last_interaction=datetime('now')""",
                (
                    rel["id"],
                    rel["character_id"],
                    rel["target_id"],
                    json.dumps(rel["dimensions"]),
                    rel.get("power_balance", 0.0),
                    rel.get("type", "stranger"),
                    rel.get("trajectory", "stable"),
                    json.dumps(rel.get("key_moments", [])),
                    json.dumps(rel.get("state") or {}),
                ),
            )
            self._commit()

    def get_relationship(self, character_id: str, target_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM relationships WHERE character_id = ? AND target_id = ?",
                (character_id, target_id),
            ).fetchone()
            if not row:
                return None
            d = dict(row)
        d["dimensions"] = json.loads(d["dimensions"])
        d["key_moments"] = json.loads(d["key_moments"])
        d["state"] = json.loads(d.get("state") or "{}")
        return d

    def get_relationships(self, character_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM relationships WHERE character_id = ?",
                (character_id,),
            ).fetchall()
            result = []
            for row in rows:
                d = dict(row)
                d["dimensions"] = json.loads(d["dimensions"])
                d["key_moments"] = json.loads(d["key_moments"])
                d["state"] = json.loads(d.get("state") or "{}")
                result.append(d)
            return result

    # ── Facts (bi-temporal) ─────────────────────────────────────────────
    _FACT_COLS = (
        "id",
        "character_id",
        "subject",
        "predicate",
        "object",
        "statement",
        "event_time",
        "valid_from",
        "valid_to",
        "recorded_at",
        "expired_at",
        "certainty",
        "importance",
        "memory_id",
        "superseded_by",
        "session_id",
        "user_id",
        "metadata",
    )

    def save_fact(self, fact: dict) -> None:
        row = {c: fact.get(c) for c in self._FACT_COLS}
        row["metadata"] = json.dumps(fact.get("metadata") or {})
        row["certainty"] = fact.get("certainty", 1.0)
        row["importance"] = fact.get("importance", 0.75)
        cols = ", ".join(self._FACT_COLS)
        marks = ", ".join("?" for _ in self._FACT_COLS)
        updates = ", ".join(f"{c}=excluded.{c}" for c in self._FACT_COLS if c != "id")
        with self._lock:
            self._conn.execute(
                f"INSERT INTO facts ({cols}) VALUES ({marks}) ON CONFLICT(id) DO UPDATE SET {updates}",
                tuple(row[c] for c in self._FACT_COLS),
            )
            self._commit()

    def _row_to_fact(self, row: sqlite3.Row) -> dict:
        d = dict(row)
        try:
            d["metadata"] = json.loads(d.get("metadata") or "{}")
        except (TypeError, ValueError):
            d["metadata"] = {}
        return d

    def get_fact(self, fact_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM facts WHERE id = ?", (fact_id,)).fetchone()
        return self._row_to_fact(row) if row else None

    def query_facts(
        self,
        character_id: str,
        *,
        subject: str | None = None,
        predicate: str | None = None,
        active_only: bool = True,
        as_of: str | None = None,
        limit: int | None = 200,
        order: str = "recorded_desc",
    ) -> list[dict]:
        q = "SELECT * FROM facts WHERE character_id = ?"
        params: list[Any] = [character_id]
        if subject is not None:
            q += " AND subject = ?"
            params.append(subject)
        if predicate is not None:
            q += " AND predicate = ?"
            params.append(predicate)
        if as_of is not None:
            q += " AND valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)"
            params += [as_of, as_of]
        elif active_only:
            q += " AND valid_to IS NULL AND expired_at IS NULL"
        q += (
            " ORDER BY valid_from ASC, rowid ASC"
            if order == "valid_asc"
            else " ORDER BY recorded_at DESC, rowid DESC"
        )
        if limit is not None:
            q += " LIMIT ?"
            params.append(limit)
        with self._lock:
            rows = self._conn.execute(q, params).fetchall()
        return [self._row_to_fact(r) for r in rows]

    def expire_fact(
        self, fact_id: str, *, valid_to: str, expired_at: str, superseded_by: str | None
    ) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE facts SET valid_to = ?, expired_at = ?, superseded_by = ? WHERE id = ?",
                (valid_to, expired_at, superseded_by, fact_id),
            )
            self._commit()

    def count_facts(self, character_id: str, active_only: bool = True) -> int:
        q = "SELECT COUNT(*) FROM facts WHERE character_id = ?"
        if active_only:
            q += " AND valid_to IS NULL AND expired_at IS NULL"
        with self._lock:
            return self._conn.execute(q, (character_id,)).fetchone()[0]

    def count_active_facts_for_memory(self, memory_id: str) -> int:
        """Count facts still referencing `memory_id` that are active — `valid_to
        IS NULL` and not `metadata.retracted` — used to guard against archiving
        a memory shared by more than one fact (semantic dedup can link two
        facts to the same row) when only one of them is being retracted."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM facts WHERE memory_id = ? AND valid_to IS NULL "
                "AND (json_extract(metadata, '$.retracted') IS NULL "
                "OR json_extract(metadata, '$.retracted') = 0)",
                (memory_id,),
            ).fetchone()
        return row[0]

    def update_fact_fields(
        self,
        fact_id: str,
        *,
        object: str | None = None,
        statement: str | None = None,
        certainty: float | None = None,
        importance: float | None = None,
        metadata: dict | None = None,
        memory_id: str | None = None,
    ) -> None:
        """`memory_id` re-links the fact to a different memory row — used by
        `FactStore.edit()`'s shared-row guard, which relinks an edited fact to a
        freshly created memory instead of rewriting a row still referenced by
        another active fact."""
        sets: list[str] = []
        params: list[Any] = []
        if object is not None:
            sets.append("object = ?")
            params.append(object)
        if statement is not None:
            sets.append("statement = ?")
            params.append(statement)
        if certainty is not None:
            sets.append("certainty = ?")
            params.append(max(0.0, min(1.0, float(certainty))))
        if importance is not None:
            sets.append("importance = ?")
            params.append(max(0.0, min(1.0, float(importance))))
        if metadata is not None:
            sets.append("metadata = ?")
            params.append(json.dumps(metadata))
        if memory_id is not None:
            sets.append("memory_id = ?")
            params.append(memory_id)
        if not sets:
            return
        params.append(fact_id)
        with self._lock:
            self._conn.execute(f"UPDATE facts SET {', '.join(sets)} WHERE id = ?", params)
            self._commit()

    def delete_fact(self, fact_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
            self._commit()
            return cur.rowcount > 0

    def unlink_fact_memory(self, memory_id: str) -> list[str]:
        """Set memory_id=NULL on every fact referencing it (active or historical).

        Returns only the ids of facts that were still *active* (`valid_to IS
        NULL`) at the time of unlinking — historical/superseded facts are
        unlinked too but are not returned, since a caller that retracts
        every returned id (e.g. `MemoryStore.delete`) should not stamp an
        already-expired fact as retracted.
        """
        with self._lock:
            ids = [
                r[0]
                for r in self._conn.execute(
                    "SELECT id FROM facts WHERE memory_id = ? AND valid_to IS NULL",
                    (memory_id,),
                ).fetchall()
            ]
            self._conn.execute(
                "UPDATE facts SET memory_id = NULL WHERE memory_id = ?", (memory_id,)
            )
            self._commit()
        return ids

    # ── Sessions ────────────────────────────────────────────────

    def save_session(self, session: dict) -> None:
        """`started_at`/`ended_at` may be supplied as "YYYY-MM-DD HH:MM:SS" strings;
        otherwise both are stamped from the injectable clock (ended_at is re-stamped
        on every update unless explicitly supplied)."""
        stamp = sqlite_ts()
        started_at = session.get("started_at") or stamp
        ended_at = session.get("ended_at") or stamp
        with self._lock:
            self._conn.execute(
                """INSERT INTO sessions (id, character_id, summary, started_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       summary=excluded.summary, ended_at=?""",
                (
                    session["id"],
                    session["character_id"],
                    session.get("summary"),
                    started_at,
                    ended_at,
                ),
            )
            self._commit()

    def get_sessions(self, character_id: str, limit: int = 20) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM sessions WHERE character_id = ? ORDER BY started_at DESC LIMIT ?",
                (character_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def rename_session(self, session_id: str, alias: str) -> None:
        """Set or update the alias for a session."""
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET alias = ? WHERE id = ?",
                (alias, session_id),
            )
            self._commit()

    def reopen_session(self, session_id: str) -> None:
        """Clear ended_at so the session is active again."""
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET ended_at = NULL WHERE id = ?",
                (session_id,),
            )
            self._commit()

    # ── Session turns (durable conversation buffer) ────────────────

    def add_session_turn(
        self, session_id: str, character_id: str, seq: int, role: str, content: str
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO session_turns (session_id, character_id, seq, role, content) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, character_id, seq, role, content),
            )
            self._commit()

    def get_session_turns(self, session_id: str, tail: int | None = None) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT seq, role, content, created_at FROM session_turns "
                "WHERE session_id = ? ORDER BY seq",
                (session_id,),
            ).fetchall()
            turns = [dict(r) for r in rows]
        if tail is not None:
            turns = turns[-tail:]
        return turns

    # ── Callbacks ─────────────────────────────────────────────

    def save_callback(self, cb: dict) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO callbacks "
                "(id, character_id, kind, hook, source_memory_ids, salience, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    cb["id"],
                    cb["character_id"],
                    cb["kind"],
                    cb["hook"],
                    json.dumps(cb.get("source_memory_ids", [])),
                    cb.get("salience", 0.5),
                    cb.get("status", "ready"),
                ),
            )
            self._commit()

    def get_callbacks(
        self, character_id: str, status: str = "ready", limit: int = 10
    ) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM callbacks WHERE character_id = ? AND status = ? "
                "ORDER BY salience DESC, created_at DESC LIMIT ?",
                (character_id, status, limit),
            ).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["source_memory_ids"] = json.loads(d.get("source_memory_ids") or "[]")
                out.append(d)
            return out

    def mark_callback(self, callback_id: str, status: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE callbacks SET status = ?, "
                "consumed_at = CASE WHEN ? = 'consumed' THEN datetime('now') ELSE consumed_at END "
                "WHERE id = ?",
                (status, status, callback_id),
            )
            self._commit()

    # ── Meta ──────────────────────────────────────────────────

    def meta_get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
            return row[0] if row else None

    def meta_set(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, str(value)),
            )
            self._commit()

    # ── Memory maintenance helpers ────────────────────────────

    def set_memory_importance(self, memory_id: str, importance: float) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE memories SET importance = ? WHERE id = ?",
                (max(0.0, min(1.0, importance)), memory_id),
            )
            self._commit()

    def archive_memories_batch(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        with self._lock:
            self._conn.executemany(
                "UPDATE memories SET status = 'archived' WHERE id = ?",
                [(mid,) for mid in memory_ids],
            )
            self._commit()
