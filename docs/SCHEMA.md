# Storage Schema — the Portable Contract

This document specifies the on-disk SQLite format used by Woven Imprint
(`src/woven_imprint/storage/sqlite.py`). It is the contract for **foreign
writers** — a Kotlin edge runtime, a game engine, a sync tool — that read or
write a character database without going through the Python library. If you
follow this document, the Python library and your runtime can share one DB
file.

Current schema version: **4** (integer, `schema_version` table).
Current semantic tag: **`0.6.0-dev`** (`meta` key `schema_semver`).

## Database-level settings

`SQLiteStorage` opens every database with:

```sql
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;
PRAGMA busy_timeout=5000;
```

Foreign writers should do the same. WAL is persistent (stored in the DB
header); `foreign_keys` and `busy_timeout` are per-connection and must be set
on each connection.

**FTS5 is a hard requirement.** The base schema creates a virtual table with
`CREATE VIRTUAL TABLE ... USING fts5(...)`. On a SQLite build compiled
without FTS5 this statement fails, and `SQLiteStorage(...)` construction
raises — the library does **not** currently degrade to a `LIKE`-scan
fallback. See [FTS5 dependency](#fts5-dependency) below.

## Tables

All `id` columns except `session_turns.id` (DB-generated `INTEGER PRIMARY KEY
AUTOINCREMENT`) are application-generated TEXT (the Python library uses
`generate_id(prefix)` — e.g. `mem-…`, `cb-…`). All `DATETIME DEFAULT
(datetime('now'))` columns use the timestamp format described in
[Timestamps](#timestamps).

### `characters`

| Column | Type | Constraints / default |
|---|---|---|
| `id` | TEXT | PRIMARY KEY |
| `name` | TEXT | NOT NULL |
| `persona` | JSON (TEXT) | NOT NULL — persona dict (hard/soft/temporal constraints) |
| `birthdate` | TEXT | nullable |
| `state` | JSON (TEXT) | DEFAULT `'{}'` — emotion/arc/session state |
| `created_at` | DATETIME | DEFAULT `datetime('now')` |
| `updated_at` | DATETIME | DEFAULT `datetime('now')` — bumped on upsert |

### `memories`

| Column | Type | Constraints / default |
|---|---|---|
| `id` | TEXT | PRIMARY KEY |
| `character_id` | TEXT | NOT NULL, REFERENCES `characters(id)` |
| `tier` | TEXT | NOT NULL, CHECK `tier IN ('buffer', 'core', 'bedrock')` |
| `content` | TEXT | NOT NULL |
| `embedding` | BLOB | nullable — see [Embedding BLOB format](#embedding-blob-format) |
| `importance` | REAL | DEFAULT `0.5` |
| `certainty` | REAL | DEFAULT `1.0` |
| `status` | TEXT | DEFAULT `'active'`, CHECK `status IN ('active', 'contradicted', 'archived')` |
| `source_refs` | JSON (TEXT) | DEFAULT `'[]'` — IDs of source memories (consolidation provenance) |
| `session_id` | TEXT | nullable |
| `role` | TEXT | nullable — `'user'`, `'character'`, `'observation'`, `'event'`, … |
| `metadata` | JSON (TEXT) | DEFAULT `'{}'` |
| `created_at` | DATETIME | DEFAULT `datetime('now')` |
| `accessed_at` | DATETIME | DEFAULT `datetime('now')` — touched on retrieval |

Indexes:

```sql
CREATE INDEX idx_memories_character ON memories(character_id, tier, status);
CREATE INDEX idx_memories_session  ON memories(session_id);
CREATE INDEX idx_memories_accessed ON memories(character_id, accessed_at);
CREATE INDEX idx_memories_created  ON memories(character_id, created_at DESC);
```

### `relationships`

| Column | Type | Constraints / default |
|---|---|---|
| `id` | TEXT | PRIMARY KEY |
| `character_id` | TEXT | NOT NULL, REFERENCES `characters(id)` |
| `target_id` | TEXT | NOT NULL — user ID or other character ID |
| `dimensions` | JSON (TEXT) | NOT NULL — `{trust, affection, respect, familiarity, tension}` |
| `power_balance` | REAL | DEFAULT `0.0` |
| `type` | TEXT | DEFAULT `'stranger'` |
| `trajectory` | TEXT | DEFAULT `'stable'` |
| `key_moments` | JSON (TEXT) | DEFAULT `'[]'` |
| `formed_at` | DATETIME | DEFAULT `datetime('now')` |
| `last_interaction` | DATETIME | DEFAULT `datetime('now')` — bumped on upsert |

Indexes:

```sql
CREATE INDEX idx_relationships_character ON relationships(character_id);
CREATE INDEX idx_relationships_pair      ON relationships(character_id, target_id);
```

### `sessions`

| Column | Type | Constraints / default |
|---|---|---|
| `id` | TEXT | PRIMARY KEY |
| `character_id` | TEXT | NOT NULL, REFERENCES `characters(id)` |
| `summary` | TEXT | nullable |
| `started_at` | DATETIME | DEFAULT `datetime('now')` |
| `ended_at` | DATETIME | nullable — NULL = session still open |
| `alias` | TEXT | nullable — added by **migration v2** |

### `session_turns` (migration v3)

Durable conversation buffer — turns are rehydrated into context on
`resume_session()`.

| Column | Type | Constraints / default |
|---|---|---|
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT |
| `session_id` | TEXT | NOT NULL (no FK constraint) |
| `character_id` | TEXT | NOT NULL (no FK constraint) |
| `seq` | INTEGER | NOT NULL — per-session ordering |
| `role` | TEXT | NOT NULL |
| `content` | TEXT | NOT NULL |
| `created_at` | DATETIME | DEFAULT `datetime('now')` |

Index: `CREATE INDEX idx_session_turns_session ON session_turns(session_id, seq);`

### `callbacks` (migration v4)

Batch-generated, paraphrased conversation hooks (see
[DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) "Callbacks & proactive initiation").

| Column | Type | Constraints / default |
|---|---|---|
| `id` | TEXT | PRIMARY KEY |
| `character_id` | TEXT | NOT NULL (no FK constraint) |
| `kind` | TEXT | NOT NULL, CHECK `kind IN ('open_thread', 'callback', 'milestone', 'curiosity')` |
| `hook` | TEXT | NOT NULL — paraphrased, in-character sentence (never a verbatim memory quote) |
| `source_memory_ids` | JSON (TEXT) | DEFAULT `'[]'` |
| `salience` | REAL | DEFAULT `0.5` |
| `status` | TEXT | DEFAULT `'ready'`, CHECK `status IN ('ready', 'consumed', 'expired')` |
| `created_at` | DATETIME | DEFAULT `datetime('now')` |
| `consumed_at` | DATETIME | nullable — set when status transitions to `'consumed'` |

Index: `CREATE INDEX idx_callbacks_character ON callbacks(character_id, status, salience);`

### `meta` (migration v4)

Simple key/value store for DB-level contract data.

| Column | Type | Constraints |
|---|---|---|
| `key` | TEXT | PRIMARY KEY |
| `value` | TEXT | NOT NULL |

Known keys:

| Key | Meaning |
|---|---|
| `embedding_dimensions` | Vector dimensionality of every embedding in this DB (e.g. `"768"`). Written on the first embedded memory write. |
| `embedding_model` | Embedding model name that produced the vectors (e.g. `"nomic-embed-text"`). Written alongside `embedding_dimensions`. |
| `schema_semver` | Semantic tag of the library that last opened the DB (currently `"0.6.0-dev"`). Stamped unconditionally by `_init_schema()` on every open — fresh and upgraded DBs alike. |

### `schema_version`

| Column | Type | Constraints |
|---|---|---|
| `version` | INTEGER | PRIMARY KEY |
| `applied_at` | DATETIME | DEFAULT `datetime('now')` |

One row per applied migration. The current schema version of a DB is
`SELECT MAX(version) FROM schema_version`.

### `memories_fts` (FTS5 virtual table)

```sql
CREATE VIRTUAL TABLE memories_fts USING fts5(
    content, character_id UNINDEXED, tier UNINDEXED,
    content='memories', content_rowid='rowid'
);
```

External-content table over `memories` — see next section.

## FTS5 dependency

`memories_fts` is an **external-content** FTS5 table (`content='memories'`,
`content_rowid='rowid'`). It stores only the index, not the text; it must be
kept in sync with `memories` by **three triggers** defined in the base
schema:

1. `memories_ai` — AFTER INSERT ON `memories`: inserts the new row into the index.
2. `memories_ad` — AFTER DELETE ON `memories`: issues an FTS5 `'delete'` command for the old row.
3. `memories_au` — AFTER UPDATE **OF `content`** ON `memories`: `'delete'` of the old row + insert of the new one.

Consequences for foreign writers:

- If you write to `memories` through SQL, the triggers keep the index in
  sync automatically — do not write to `memories_fts` directly.
- The update trigger fires only on `UPDATE ... SET content = ...`. Status,
  importance, and certainty updates deliberately skip FTS reindexing.
- If your runtime rebuilds the table by other means, resync with
  `INSERT INTO memories_fts(memories_fts) VALUES('rebuild');`.

**Behavior without FTS5**: if the SQLite build lacks FTS5, opening the DB
fails at schema creation (`sqlite3.OperationalError: no such module: fts5`)
— storage does not degrade. At *query* time the picture is softer: retrieval
wraps `fts_search()` in a `try/except` and, on any FTS error, proceeds with
an empty keyword ranking, so RRF retrieval still works via the remaining
strategies (semantic, recency, importance, relationship). A sanctioned
degraded mode for FTS5-less runtimes (keyword strategy falling back to
`LIKE '%term%'` scanning) is a **Phase C requirement** — it is not
implemented today.

## Embedding BLOB format

`memories.embedding` is a packed array of 32-bit IEEE-754 floats:

```python
struct.pack(f"{len(vec)}f", *vec)   # write
struct.unpack(f"{len(blob)//4}f", blob)  # read
```

- 4 bytes per component, no header, no length prefix — dimensionality is
  `len(blob) / 4`.
- **Byte order**: the code uses `struct`'s *native* format (no `<`/`>`
  prefix). All platforms Woven Imprint targets (x86-64, ARM64) are
  little-endian, so on-disk data is little-endian float32 in practice.
  Foreign writers MUST write little-endian float32; a big-endian runtime
  must byte-swap rather than write native order.
- `NULL` is valid — memories without embeddings participate in every
  retrieval strategy except semantic ranking.

**Dimension guard** (contract): all embeddings in one DB must come from the
same model with the same dimensionality.

- On the **first** embedded memory write, `MemoryStore.add()` records
  `meta.embedding_dimensions` (vector length as string) and
  `meta.embedding_model` (from config).
- On every subsequent embedded write it compares the new vector's length
  against `meta.embedding_dimensions` and **raises `ValueError`** on
  mismatch ("Embedding dimension mismatch: DB stores N-d vectors, got M-d
  ...") — the write is rejected, because mixed embedding spaces silently
  corrupt cosine similarity.
- The guard lives in the Python `MemoryStore` layer, **not** in the SQLite
  layer — `SQLiteStorage.save_memory()` itself does not check. Foreign
  writers writing directly to the `memories` table MUST perform the same
  check: read `meta.embedding_dimensions`; if absent, write it (plus
  `embedding_model`) before or with the first embedded row; if present,
  refuse to write vectors of any other length.

## Timestamps

All timestamps are produced by SQLite's `datetime('now')`:

- Format: `YYYY-MM-DD HH:MM:SS` (space separator, second precision, **no**
  timezone suffix).
- Timezone: **UTC always**. Readers in this codebase parse these strings
  with `datetime.fromisoformat()` and attach `timezone.utc` when the value
  is naive — a local-time timestamp would silently skew recency decay and
  callback freshness.

Foreign writers MUST write UTC in the same `YYYY-MM-DD HH:MM:SS` format
(easiest: let the column defaults do it, or use `datetime('now')` /
`strftime('%Y-%m-%d %H:%M:%S', 'now')` in SQL).

## Migration protocol

- The base schema (`_SCHEMA`) is version **1** and is applied with
  `CREATE ... IF NOT EXISTS`, so re-running it is a no-op. It ends with
  `INSERT OR IGNORE INTO schema_version (version) VALUES (1);`.
- Later versions live in the `_MIGRATIONS` dict (`version → SQL script`):
  - **v2** — `sessions.alias` column.
  - **v3** — `session_turns` table + index.
  - **v4** — `callbacks` table + index, `meta` table.
- On open, `SQLiteStorage` reads `MAX(version)` from `schema_version` and
  applies every migration with `version > current` in ascending order,
  recording each applied version as a new `schema_version` row.
- Migrations are forward-only. There are no down-migrations.

**Rule for all writers**: never write to a database whose
`MAX(schema_version.version)` is **newer** than the newest version you know.
A newer DB may contain tables, columns, or invariants your writer doesn't
maintain, and writing to it can corrupt data for the newer runtime. Check
the version, and fail closed. (Note: the current Python implementation does
not itself refuse to open a newer DB — it simply applies no migrations — so
this rule is a contract obligation on writers, not something the schema
enforces for you.)

`meta.schema_semver` complements the integer version with a human-readable
library tag; treat the integer `schema_version` as authoritative for
compatibility decisions.
