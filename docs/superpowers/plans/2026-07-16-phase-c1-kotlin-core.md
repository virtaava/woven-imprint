# Phase C1 — Kotlin Core Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A pure-JVM Kotlin library (`kotlin/` in this repo) that reads and writes the same SQLite databases as the Python library — storage, retrieval math, prompt assembly, callbacks, and the maintenance pipeline — proven equivalent by conformance fixtures generated from the Python implementation.

**Architecture:** Conformance-first port. A Python-side fixture generator (Task 2) produces golden artifacts (schema dump, a populated DB, retrieval rankings, prompt strings, reinforce arithmetic); every Kotlin subsystem lands with a test that compares against those goldens, so "equivalent to Python" is a failing test, not a claim. LLM-dependent paths are exercised with fakes (mirroring the Python test strategy); the real llama.cpp bridge is Phase C2 (separate plan, carries the spike's `GGML_CPU_KLEIDIAI=OFF` constraint). The Python toolchain remains the authoring/debugging environment for any DB (spec §4).

**Tech Stack:** Kotlin/JVM 2.x + Gradle (standalone build at `kotlin/`, consumable via `includeBuild`), androidx.sqlite `BundledSQLiteDriver` (KMP artifact — same API on JVM and Android, FTS5 guaranteed; xerial JDBC as contained fallback if the bundled JVM natives fail on linux-aarch64), kotlinx-serialization-json, JUnit 5.

## Global Constraints

- **`docs/SCHEMA.md` is the contract; the Python source is the normative spec for every port task.** Where this plan's code and the Python source disagree, the Python source + SCHEMA.md govern, and the conformance test is the arbiter. Port tasks name their exact Python source file.
- **Schema version 4**, `meta.schema_semver = "0.6.0-dev"` (single Kotlin constant `SCHEMA_SEMVER`, keep in sync with Python's `_init_schema`). Kotlin REFUSES to open a DB whose `MAX(schema_version.version) > 4` (throws `IllegalStateException`) — the writer contract Python leaves implicit, enforced here.
- **Embedding blobs: float32 little-endian, no header** (`ByteBuffer.order(ByteOrder.LITTLE_ENDIAN)`), NULL valid. Dimension guard lives in the store layer, not SQLite (mirrors Python).
- **Timestamps: `YYYY-MM-DD HH:MM:SS`, UTC, no timezone suffix** — SQLite `datetime('now')` for defaults, `nowUtc()` helper for explicit writes.
- **PRAGMAs per connection:** `journal_mode=WAL` (skip for `:memory:`), `foreign_keys=ON`, `busy_timeout=5000`.
- **Concurrency:** one lock held across every unit of work touching the connection, reads included (mirror of Python's `RLock` invariant — Phase B finding I1 was a real race). JVM: `ReentrantLock` inside the `Db` wrapper.
- **RRF formula bit-for-bit:** `score(id) = Σ_i weight_i / (k + rank_i + 1)`, `k=60` default (`src/woven_imprint/utils/rrf.py`).
- **`generateJsonRobust` retry policy exactly:** try at caller temperature, on parse failure retry ONCE at `temperature=0.1`. Not configurable.
- **FTS5:** available via the bundled driver; but implement SCHEMA.md's sanctioned degraded mode (Phase C requirement): if FTS5 is absent at open, skip the virtual table + triggers and route the keyword strategy through `LIKE '%term%'` scanning.
- **Out of scope for C1 (documented, not silent):** emotion/arc/consistency/growth/belief-revision subsystems (the game's deterministic sim owns creature state per the design laws; prompt assembly takes a caller-provided state description instead), `reflect`/`evolve` maintenance jobs (report `status="skipped"`, `reason="unsupported in kotlin-core"`), streaming, resilience/circuit-breaker parity (C2, where a real provider exists), YAML config loading (config is programmatic data classes).
- **Toolchain on Spark:** JDK 21 explicitly (`/usr/lib/jvm/java-21-openjdk-arm64` — system default 25 breaks Gradle; same trap as Sona Forge). Gradle via wrapper (8.14+). Never commit `kotlin/build/`, `.gradle/`, or downloaded artifacts — Task 1 adds the .gitignore.
- **Package:** `com.wovenimprint.core`. Module path `kotlin/core`. Commit per task on branch `phase-c1-kotlin-core`.
- **Run tests:** `cd ~/sona/projects/woven-imprint/kotlin && JAVA_HOME=<jdk21> ./gradlew :core:test` (exact JAVA_HOME recorded in `kotlin/README.md` by Task 1).
- **Python-side commands** run from repo root with `uv run` (dev extras: `uv run --extra dev pytest -q`). The Python suite must stay green — C1 adds Python files only under `kotlin/conformance/`.

## File Structure

```
kotlin/
  settings.gradle.kts, build.gradle.kts, gradle/ (wrapper + libs.versions.toml), .gitignore, README.md
  core/build.gradle.kts
  core/src/main/kotlin/com/wovenimprint/core/
    Db.kt                 # driver wrapper: lock, exec/query/transaction, hasFts5, PRAGMAs
    Schema.kt             # base schema SQL, migrations v2-v4, semver stamping, version refusal
    EmbeddingCodec.kt     # float32-LE encode/decode
    Storage.kt            # port of storage/sqlite.py public surface
    Config.kt             # MemoryConfig / MaintenanceConfig / ContextConfig data classes
    Providers.kt          # LlmProvider / EmbeddingProvider interfaces + generateJsonRobust
    Rrf.kt                # reciprocalRankFusion
    Retrieval.kt          # MemoryRetriever (5 strategies incl. LIKE fallback)
    MemoryStore.kt        # add/guard/observe-event write path
    Persona.kt            # PersonaModel.buildSystemPrompt (byte-parity with Python)
    PromptAssembler.kt    # stable prefix + volatile block + memory formatting
    Consolidation.kt      # ConsolidationEngine port
    Maintenance.kt        # Budget + MaintenanceRunner + 9 jobs
    Callbacks.kt          # CallbackEngine: get/refresh/composeInitiation
    WovenImprintCore.kt   # facade: open DB, wire subsystems per character
  core/src/test/kotlin/com/wovenimprint/core/   # one test file per main file + fakes
    Fakes.kt              # FakeLlm / FakeEmbedder / StubEmbedder
kotlin/conformance/
  generate_fixtures.py    # Python-side golden generator (runs with repo venv)
  fixtures/               # committed goldens: schema.sql-dump.txt, conformance.db,
                          # retrieval_golden.json, prompts_golden.json, reinforce_golden.json
```

---

### Task 1: Gradle scaffold + SQLite driver selection spike

**Files:**
- Create: `kotlin/settings.gradle.kts`, `kotlin/build.gradle.kts`, `kotlin/gradle/libs.versions.toml`, `kotlin/.gitignore`, `kotlin/README.md`, `kotlin/core/build.gradle.kts`, `kotlin/core/src/main/kotlin/com/wovenimprint/core/Db.kt`
- Test: `kotlin/core/src/test/kotlin/com/wovenimprint/core/DbTest.kt`

**Interfaces:**
- Produces: `class Db(path: String) : AutoCloseable` with `fun <T> withLock(block: () -> T): T`, `fun exec(sql: String, vararg args: Any?)`, `fun query(sql: String, vararg args: Any?): List<Map<String, Any?>>` (INTEGER→Long, REAL→Double, TEXT→String, BLOB→ByteArray, NULL→null), `fun <T> transaction(block: () -> T): T`, `val hasFts5: Boolean`, `override fun close()`. Constructor applies the three PRAGMAs (WAL skipped when `path == ":memory:"`). All later tasks build on exactly this API — if the bundled driver fails on linux-aarch64, swap Db's internals to xerial JDBC without changing the API.

- [ ] **Step 1: Scaffold the Gradle build**

`kotlin/settings.gradle.kts`:
```kotlin
rootProject.name = "woven-imprint-kt"
include(":core")
```
`kotlin/gradle/libs.versions.toml`:
```toml
[versions]
kotlin = "2.1.20"
sqliteBundled = "2.5.2"
serialization = "1.8.1"
junit = "5.11.4"

[libraries]
sqlite-bundled = { module = "androidx.sqlite:sqlite-bundled", version.ref = "sqliteBundled" }
kotlinx-serialization-json = { module = "org.jetbrains.kotlinx:kotlinx-serialization-json", version.ref = "serialization" }
junit-jupiter = { module = "org.junit.jupiter:junit-jupiter", version.ref = "junit" }

[plugins]
kotlin-jvm = { id = "org.jetbrains.kotlin.jvm", version.ref = "kotlin" }
kotlin-serialization = { id = "org.jetbrains.kotlin.plugin.serialization", version.ref = "kotlin" }
```
(If a listed version is gone from Maven Central, take the closest newer and record it in `kotlin/README.md`.)

`kotlin/core/build.gradle.kts`:
```kotlin
plugins {
    alias(libs.plugins.kotlin.jvm)
    alias(libs.plugins.kotlin.serialization)
}
dependencies {
    implementation(libs.sqlite.bundled)
    implementation(libs.kotlinx.serialization.json)
    testImplementation(libs.junit.jupiter)
}
tasks.test { useJUnitPlatform() }
kotlin { jvmToolchain(21) }
```
`kotlin/.gitignore`: `build/`, `.gradle/`, `.kotlin/`, `local.properties`. Root `kotlin/build.gradle.kts` can be empty/plugins-only. Generate the wrapper: `gradle wrapper --gradle-version 8.14.3` (or copy the wrapper from any local Gradle install; commit `gradlew`, `gradle/wrapper/*`).

- [ ] **Step 2: Write the failing driver test**

`DbTest.kt`:
```kotlin
package com.wovenimprint.core

import org.junit.jupiter.api.Test
import kotlin.test.*

class DbTest {
    @Test
    fun `opens in-memory db, pragmas applied, fts5 detected`() {
        Db(":memory:").use { db ->
            assertEquals(1L, db.query("PRAGMA foreign_keys").first().values.first())
            db.exec("CREATE TABLE t (id TEXT PRIMARY KEY, n INTEGER, x REAL, b BLOB)")
            db.exec("INSERT INTO t VALUES (?, ?, ?, ?)", "a", 1L, 2.5, byteArrayOf(1, 2))
            val row = db.query("SELECT * FROM t").single()
            assertEquals("a", row["id"]); assertEquals(1L, row["n"])
            assertEquals(2.5, row["x"]); assertContentEquals(byteArrayOf(1, 2), row["b"] as ByteArray)
            assertTrue(db.hasFts5, "bundled sqlite must ship FTS5")
        }
    }

    @Test
    fun `transaction rolls back on throw`() {
        Db(":memory:").use { db ->
            db.exec("CREATE TABLE t (id TEXT)")
            runCatching { db.transaction { db.exec("INSERT INTO t VALUES ('x')"); error("boom") } }
            assertEquals(0, db.query("SELECT count(*) c FROM t").single()["c"])
        }
    }
}
```

- [ ] **Step 3: Run to verify it fails** — `./gradlew :core:test` → compile error (Db undefined).

- [ ] **Step 4: Implement `Db.kt`** against `androidx.sqlite.driver.bundled.BundledSQLiteDriver` (`driver.open(path)` → `SQLiteConnection`; use `prepare(sql)` statements, bind by 1-based index: `bindText/bindLong/bindDouble/bindBlob/bindNull`, `step()`, column readers by type via `getColumnType`). `hasFts5` = `query("SELECT count(*) c FROM pragma_module_list WHERE name='fts5'")` > 0 (wrap in runCatching → false). `withLock` uses one `ReentrantLock`; `exec/query/transaction` all take it (reentrant, so nesting is safe). Transactions: `BEGIN`/`COMMIT`/`ROLLBACK` via exec.

- [ ] **Step 5: Run test.** If `BundledSQLiteDriver` throws `UnsatisfiedLinkError` on linux-aarch64: swap Db internals to xerial (`org.xerial:sqlite-jdbc:3.49+`, which ships linux-aarch64 natives and FTS5) keeping the exact same Db API, record the decision in `kotlin/README.md`, and note that Android (C2) will use the androidx driver behind the same class. Expected: 2 tests pass.

- [ ] **Step 6: Write `kotlin/README.md`** — how to build/test on Spark (exact JAVA_HOME that worked, driver decision, module layout) — and commit: `git add kotlin && git commit -m "feat(kotlin): C1 scaffold, Db driver wrapper with FTS5 detection (task 1)"`

---

### Task 2: Python-side conformance fixture generator

**Files:**
- Create: `kotlin/conformance/generate_fixtures.py`, `kotlin/conformance/fixtures/` (5 committed artifacts)

**Interfaces:**
- Produces (all committed, regenerable): `fixtures/schema_dump.txt` (normalized `sqlite_master` of a fresh Python-created DB), `fixtures/conformance.db` (Python-written DB: 1 character "conformance-char", 12 memories across tiers with FakeEmbedder-style embeddings + 2 NULL-embedding rows, 2 sessions + 6 session_turns, 3 callbacks in each status, meta stamped dims=50/model="fake-embedder"), `fixtures/retrieval_golden.json` (for 4 fixed queries: the query embedding as float list + expected ranked memory-id list from `MemoryRetriever.retrieve`, default config, plus one entry with `relationship_target`), `fixtures/prompts_golden.json` (persona dict → `build_system_prompt()` string, for 3 personas incl. one with birthdate + temporal fields; plus `_format_memories` output for a fixed memory list), `fixtures/reinforce_golden.json` (table of `(importance, certainty) → reinforce() result` pairs for 6 inputs, from `memory/belief.py`).
- Consumes: the Python library itself (`SQLiteStorage`, `MemoryRetriever`, `PersonaModel`, `BeliefSystem`, `tests/helpers.py::FakeEmbedder`).

- [ ] **Step 1: Write `generate_fixtures.py`.** Deterministic (fixed content strings, fixed FakeEmbedder instance so vocab order is stable, timestamps written explicitly via SQL so reruns are stable — use fixed literals like `'2026-07-01 10:00:00'` and step by hours across memories so recency ranking is deterministic and stable regardless of run date; IMPORTANT: recency decay in retrieval uses hours-since-now, so retrieval_golden must be regenerated against a **frozen now**: monkeypatch/parameterize the retriever's now — read `memory/retrieval.py` to find the time source and freeze it via `unittest.mock.patch`; record the frozen instant `2026-07-15 00:00:00` in the JSON so the Kotlin test freezes the same clock). Structure: `python generate_fixtures.py --out fixtures/` writes all five artifacts; normalize `schema_dump.txt` as `SELECT type, name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name`, one row per line, whitespace-collapsed (`" ".join(sql.split())`).

- [ ] **Step 2: Run it** — `cd ~/sona/projects/woven-imprint && uv run python kotlin/conformance/generate_fixtures.py --out kotlin/conformance/fixtures/`. Verify: 5 files; `conformance.db` opens in `sqlite3` with 14 memories, schema_version max = 4; rerunning produces byte-identical `schema_dump.txt` and JSON files (the .db may differ in WAL bytes — run `PRAGMA wal_checkpoint(TRUNCATE)` + `VACUUM` before closing to stabilize; if still not byte-stable, that is acceptable for the .db only — document in the script docstring).

- [ ] **Step 3: Python suite still green** — `uv run --extra dev pytest -q` → same pass count as master (374 passed, 6 skipped).

- [ ] **Step 4: Commit:** `git add kotlin/conformance && git commit -m "feat(kotlin): conformance fixture generator + committed goldens (task 2)"`

---

### Task 3: Schema, migrations, semver stamping, newer-DB refusal

**Files:**
- Create: `kotlin/core/src/main/kotlin/com/wovenimprint/core/Schema.kt`
- Test: `kotlin/core/src/test/kotlin/com/wovenimprint/core/SchemaTest.kt`
- Python source (normative): `src/woven_imprint/storage/sqlite.py` (`_SCHEMA`, `_MIGRATIONS`, `_init_schema`, `_run_migrations`) + `docs/SCHEMA.md`

**Interfaces:**
- Produces: `object Schema { const val VERSION = 4; const val SEMVER = "0.6.0-dev"; fun init(db: Db) }` — `init` = base schema (CREATE IF NOT EXISTS incl. FTS5 table + 3 triggers, but ONLY when `db.hasFts5`), migrations v2/v3/v4 applied per the protocol (read MAX(version), apply ascending, record each, per-migration transaction), unconditional `meta` upsert of `schema_semver`, and the refusal: `MAX(version) > VERSION` → `IllegalStateException("db schema vN is newer than supported v4")`. Also `fun schemaDump(db: Db): List<String>` (same normalization as the fixture).

- [ ] **Step 1: Failing tests** — `SchemaTest.kt`:
```kotlin
package com.wovenimprint.core

import org.junit.jupiter.api.Test
import java.nio.file.Files
import java.nio.file.Paths
import kotlin.test.*

class SchemaTest {
    private val fixtures = Paths.get("../conformance/fixtures")

    @Test
    fun `fresh kotlin db has byte-identical normalized schema to python`() {
        Db(":memory:").use { db ->
            Schema.init(db)
            val expected = Files.readAllLines(fixtures.resolve("schema_dump.txt"))
            assertEquals(expected, Schema.schemaDump(db))
        }
    }

    @Test
    fun `opens python-written conformance db without applying any migration`() {
        val copy = Files.createTempFile("conf", ".db")
        Files.copy(fixtures.resolve("conformance.db"), copy, java.nio.file.StandardCopyOption.REPLACE_EXISTING)
        Db(copy.toString()).use { db ->
            Schema.init(db)
            assertEquals(4L, db.query("SELECT MAX(version) v FROM schema_version").single()["v"])
            assertEquals("0.6.0-dev", db.query("SELECT value FROM meta WHERE key='schema_semver'").single()["value"])
        }
    }

    @Test
    fun `refuses a newer-schema db`() {
        Db(":memory:").use { db ->
            Schema.init(db)
            db.exec("INSERT INTO schema_version (version) VALUES (99)")
            assertFailsWith<IllegalStateException> { Schema.init(db) }
        }
    }

    @Test
    fun `migrates a v2 db to v4`() {
        Db(":memory:").use { db ->
            // hand-rolled v2: base tables without session_turns/callbacks/meta, versions 1+2 recorded
            // (port the v2 DDL from tests/test_storage.py::TestMigrations' v2 fixture builder)
            buildV2Schema(db)
            Schema.init(db)
            assertEquals(4L, db.query("SELECT MAX(version) v FROM schema_version").single()["v"])
            db.exec("INSERT INTO session_turns (session_id, character_id, seq, role, content) VALUES ('s','c',1,'user','hi')")
            db.exec("INSERT INTO callbacks (id, character_id, kind, hook) VALUES ('cb1','c','callback','h')")
        }
    }
}
```
`buildV2Schema` is a private test helper in the same file — transcribe the v2 DDL from `tests/test_storage.py` (the Python migration tests build exactly this fixture; copy its CREATE statements).

- [ ] **Step 2: Run to verify fail.** — `./gradlew :core:test --tests '*SchemaTest*'` → compile error.

- [ ] **Step 3: Implement `Schema.kt`.** Transcribe the DDL from `storage/sqlite.py` `_SCHEMA` string and `_MIGRATIONS` dict verbatim (same column order, same CHECK constraints, same index names — the schema-dump parity test fails on any drift). FTS5 block (virtual table + `memories_ai`/`memories_ad`/`memories_au` triggers) emitted only when `db.hasFts5`; store the flag for Task 8's fallback. Migration loop mirrors `_run_migrations` (default current=1 when the version read fails).

- [ ] **Step 4: Run tests** → 4 pass (plus Task 1's). The schema-dump test is the contract gate: iterate until byte-identical.

- [ ] **Step 5: Commit:** `git commit -am "feat(kotlin): schema v4 + migrations with python schema-dump parity (task 3)"`

---

### Task 4: Embedding codec + dimension guard

**Files:**
- Create: `kotlin/core/src/main/kotlin/com/wovenimprint/core/EmbeddingCodec.kt`
- Test: `kotlin/core/src/test/kotlin/com/wovenimprint/core/EmbeddingCodecTest.kt`
- Python source (normative): `src/woven_imprint/memory/store.py::guard_embedding_dimension`, SCHEMA.md blob section

**Interfaces:**
- Produces: `object EmbeddingCodec { fun encode(vec: FloatArray): ByteArray; fun decode(blob: ByteArray): FloatArray }` (little-endian float32, `require(blob.size % 4 == 0)`); `fun guardEmbeddingDimension(db: Db, embedding: FloatArray?, embeddingModel: String)` — null embedding → no-op; meta `embedding_dimensions` absent → stamp dims + `embedding_model`; present and mismatched → `IllegalArgumentException` with both dimensions in the message (mirror the Python error text).

- [ ] **Step 1: Failing tests** — round-trip (`encode(decode(x)) == x` for a 768-dim random-but-seeded vector); **cross-implementation read**: open `fixtures/conformance.db`, `SELECT embedding FROM memories WHERE embedding IS NOT NULL LIMIT 1`, decode → length 50, all finite, L2 norm ≈ 1.0 (FakeEmbedder normalizes; tolerance 1e-5); guard: fresh DB + Schema.init → guard with 50-dim stamps meta (`embedding_dimensions="50"`, `embedding_model="fake-embedder"`), second guard with 49-dim throws, guard(null) never throws.

- [ ] **Step 2: Run → fail. Step 3: Implement (ByteBuffer, LITTLE_ENDIAN, asFloatBuffer). Step 4: Run → pass. Step 5: Commit** `"feat(kotlin): embedding codec (float32-LE) + dimension guard with cross-impl read test (task 4)"`

---

### Task 5: Storage — memories, FTS search, concurrency lock

**Files:**
- Create: `kotlin/core/src/main/kotlin/com/wovenimprint/core/Storage.kt` (memories half)
- Test: `kotlin/core/src/test/kotlin/com/wovenimprint/core/StorageMemoriesTest.kt`
- Python source (normative): `src/woven_imprint/storage/sqlite.py` — `save_memory`, `get_memory`, `get_memories`, `update_memory`, `archive_memory`, `touch_memories`, `fts_search`

**Interfaces:**
- Produces: `class Storage(val db: Db)` created AFTER `Schema.init`. Data carrier: `typealias Row = Map<String, Any?>` (JSON columns decoded to kotlinx `JsonElement` by helper `Storage.json(row, col)` — raw TEXT in the map). Methods this task: `saveMemory(m: MemoryRecord): Row` where `data class MemoryRecord(val id: String, val characterId: String, val tier: String, val content: String, val embedding: FloatArray?, val importance: Double = 0.5, val certainty: Double = 1.0, val status: String = "active", val sourceRefs: String = "[]", val sessionId: String? = null, val role: String? = null, val metadata: String = "{}", val createdAt: String? = null)` (null createdAt → SQL default); `getMemory(id): Row?`; `getMemories(characterId, tier: String? = null, status: String = "active", limit: Int = 1000, oldestFirst: Boolean = false): List<Row>`; `updateMemory(id, fields: Map<String, Any?>)` (whitelist columns: importance, certainty, status, content, metadata, accessed_at); `archiveMemory(id)`; `touchMemories(ids: List<String>)` (single UPDATE ... IN, sets accessed_at=nowUtc()); `ftsSearch(characterId, query, limit = 50): List<Row>` — sanitization exactly as Python: tokenize `\w+`, cap 20 tokens, each double-quoted, joined with ` OR `, empty → emptyList; on FTS unavailable delegate to `likeSearch` (Task 8 wires it; for now return emptyList when `!db.hasFts5`). `fun nowUtc(): String` helper here.

- [ ] **Step 1: Failing tests** — save/get round-trip incl. embedding blob + JSON defaults; `getMemories` tier/status/limit/oldestFirst filters (assert ordering by created_at then rowid); `ftsSearch` finds a seeded memory by keyword, respects character_id isolation, empty query → empty, 25-word query doesn't throw (cap at 20); **concurrency stress** (the Phase B I1 mirror):
```kotlin
@Test
fun `concurrent writers and readers raise nothing`() {
    Db(":memory:").use { db ->
        Schema.init(db); val s = Storage(db)
        val errors = java.util.concurrent.ConcurrentLinkedQueue<Throwable>()
        val threads = (1..8).map { t ->
            Thread {
                runCatching {
                    repeat(50) { i ->
                        s.saveMemory(MemoryRecord("m$t-$i", "c", "buffer", "content $t $i", null))
                        s.getMemories("c", limit = 20)
                    }
                }.onFailure { errors.add(it) }
            }
        }
        threads.forEach { it.start() }; threads.forEach { it.join() }
        assertTrue(errors.isEmpty(), errors.toString())
        assertEquals(400L, db.query("SELECT count(*) c FROM memories").single()["c"])
    }
}
```

- [ ] **Step 2: Run → fail. Step 3: Implement** (every method body wrapped in `db.withLock { ... }`; SQL transcribed from the Python methods). **Step 4: Run → pass** (stress test must pass 3 consecutive runs: `./gradlew :core:test --tests '*StorageMemoriesTest*' --rerun` ×3). **Step 5: Commit** `"feat(kotlin): storage memories + fts search + locked concurrency (task 5)"`

---

### Task 6: Storage — characters, sessions, turns, callbacks, meta

**Files:**
- Modify: `kotlin/core/src/main/kotlin/com/wovenimprint/core/Storage.kt`
- Test: `kotlin/core/src/test/kotlin/com/wovenimprint/core/StorageRestTest.kt`
- Python source (normative): `storage/sqlite.py` — `save_character/get_character`, `create_session/end_session/reopen_session/get_sessions`, `add_session_turn/get_session_turns`, `save_callback/get_callbacks/mark_callback`, `meta_get/meta_set`

**Interfaces:**
- Produces on `Storage`: `saveCharacter(id, name, persona: String, birthdate: String?, state: String = "{}")` / `getCharacter(id): Row?` / `updateCharacterState(id, state: String)`; `createSession(id, characterId, alias: String? = null)` / `endSession(id, summary: String?)` / `reopenSession(id)` / `getSessions(characterId, openOnly: Boolean = false): List<Row>`; `addSessionTurn(sessionId, characterId, seq: Long, role, content)` / `getSessionTurns(sessionId, tail: Int? = null): List<Row>` (ordered by seq; tail slice client-side, mirroring Python); `saveCallback(id, characterId, kind, hook, sourceMemoryIds: String = "[]", salience: Double = 0.5)` / `getCallbacks(characterId, status: String = "ready", limit: Int = 10): List<Row>` (ORDER BY salience DESC, created_at DESC) / `markCallback(id, status)` (sets consumed_at=nowUtc() when status=="consumed"); `metaGet(key): String?` / `metaSet(key, value)` (upsert).

- [ ] **Step 1: Failing tests** covering: character round-trip with persona JSON; session open→turns→end→reopen lifecycle; `getSessionTurns(tail=2)` returns the LAST 2 in seq order; callback save→get ordering (two saliences)→mark consumed sets consumed_at and drops it from ready; kind CHECK constraint violation throws (invalid kind rejected by SQLite); metaSet upsert overwrites. Also a **cross-impl read**: from `fixtures/conformance.db` read the 6 session_turns and 3 callbacks Python wrote and assert exact contents/ordering match the counts+statuses documented in the fixture generator.

- [ ] **Step 2-4: fail → implement → pass. Step 5: Commit** `"feat(kotlin): storage sessions/turns/callbacks/meta with cross-impl reads (task 6)"`

---

### Task 7: Config data classes + provider interfaces + fakes + generateJsonRobust

**Files:**
- Create: `Config.kt`, `Providers.kt`, test `ProvidersTest.kt`, test-fixture `core/src/test/kotlin/com/wovenimprint/core/Fakes.kt`
- Python source (normative): `src/woven_imprint/config.py` (defaults), `src/woven_imprint/llm/base.py` (`generate_json_robust`), `tests/helpers.py` (FakeLLM/FakeEmbedder shapes)

**Interfaces:**
- Produces: `data class MemoryConfig(...)`, `data class MaintenanceConfig(...)`, `data class ContextConfig(...)` with EXACTLY the Python defaults (all 14 maintenance fields: maxLlmCallsPerRun=50, consolidateChunkSize=500, bufferTtlDays=14, bufferHygieneMaxImportance=0.55, importanceScoringBatch=30, dedupScanLimit=200, dedupSimilarity=0.92, reinforceSimilarity=0.85, contradictionCandidateSimilarity=0.70, contradictionMaxPairs=10, reflectImportanceSum=12.0, callbacksRefreshLimit=5, callbacksReadyCap=10, callbacksRefreshOnSessionEnd=true; memory fields incl. rrfK=60, weightSemantic/Keyword/Recency/Importance/Relationship=1.0, decayBedrock=0.9999, decayCore=0.999, decayBuffer=0.995, tierBoostBedrock=0.35, tierBoostCore=0.2, tierBoostBuffer=0.0, recencyAnchor="created", sessionSummaryImportance=0.85, factImportance=0.75, consolidationThreshold=100, clusteringSimilarity=0.75); `data class CoreConfig(val memory: MemoryConfig = MemoryConfig(), val maintenance: MaintenanceConfig = MaintenanceConfig(), val context: ContextConfig = ContextConfig())`.
- `interface LlmProvider { fun generate(messages: List<ChatMessage>, temperature: Double = 0.7, maxTokens: Int = 2048): String; fun generateJson(messages: List<ChatMessage>, temperature: Double = 0.3): JsonElement }` + `data class ChatMessage(val role: String, val content: String)` + extension `fun LlmProvider.generateJsonRobust(messages, temperature = 0.3): JsonElement` — try, catch (SerializationException/IllegalArgumentException) retry once at 0.1, rethrow second failure.
- `interface EmbeddingProvider { val model: String; fun embed(text: String): FloatArray }`.
- Test fakes: `class FakeLlm : LlmProvider` — `generate` returns `"I hear you."`; `generateJson` keyword-sniffs `messages[0].content.lowercase()` mirroring Python (extract/fact → `["A notable fact was shared"]`; summar → `"Session summary"`; contradict → `{"contradictory": false, "current": "unclear"}`; score/importance → array of 5s sized to the request... keep the Python cases and ADD: callback → `[{"hook":"they mentioned the storm","kind":"callback","sources":[0]}]`); a settable `nextJson: JsonElement?` override slot for per-test canned responses; call counter. `class FakeEmbedder(val dims: Int = 50) : EmbeddingProvider` — bag-of-words first-seen vocab hashing, L2-normalized (self-consistent; parity with Python NOT required — goldens carry their own vectors). `class StubEmbedder(val vector: FloatArray) : EmbeddingProvider` returning the fixed vector (used by retrieval golden tests).

- [ ] **Step 1: Failing tests** — generateJsonRobust: first call throws parse error → retried at temperature 0.1 (assert the fake recorded temps `[0.3, 0.1]`), second failure propagates; FakeEmbedder: same text → same vector, norm ≈ 1.0, different texts differ; config defaults spot-check (assert 6 values incl. rrfK=60 and bufferTtlDays=14).
- [ ] **Step 2-4: fail → implement → pass. Step 5: Commit** `"feat(kotlin): config defaults, provider interfaces, json-robust retry, test fakes (task 7)"`

---

### Task 8: RRF + retrieval (5 strategies, LIKE fallback, golden parity)

**Files:**
- Create: `Rrf.kt`, `Retrieval.kt`; Modify: `Storage.kt` (add `likeSearch`)
- Test: `RrfTest.kt`, `RetrievalTest.kt`
- Python source (normative): `utils/rrf.py`, `memory/retrieval.py`

**Interfaces:**
- Produces: `fun reciprocalRankFusion(rankedLists: List<List<String>>, k: Int = 60, weights: List<Double>? = null): List<Pair<String, Double>>` — exact formula, weights default all-1.0, result sorted score-descending (ties: stable by first-list order — check Python's tie behavior and mirror it; Python sorts by `-score` with dict insertion order as tiebreak).
- `class MemoryRetriever(val storage: Storage, val embedder: EmbeddingProvider, val characterId: String, val config: CoreConfig, val clock: () -> java.time.Instant = { java.time.Instant.now() })` with `fun retrieve(query: String, limit: Int = 10, relationshipTarget: String? = null): List<Row>`. Algorithm transcribed from `retrieval.py`: candidates = bedrock(50)+core(200)+buffer(100)+keyword-prefilter(50, FTS or LIKE), dedupe by id, sort by rowid; ranked lists: semantic (cosine desc, skipped when query blank), keyword (prefilter order), recency (`decay^hoursSinceAnchor`, per-tier decay from config, anchor created_at unless recencyAnchor=="accessed"), importance (`importance*certainty + tierBoost` + 0.2 user-affinity when metadata.user_id == relationshipTarget), relationship (binary, only when target given); fuse with rrfK + the 4-or-5 weights in that order; top limit; `storage.touchMemories(returnedIds)`. `cosine(a: FloatArray, b: FloatArray): Double` — 0.0 when either norm is 0.
- `Storage.likeSearch(characterId, query, limit = 50): List<Row>` — the SCHEMA.md Phase-C degraded mode: same `\w+`/20-token sanitization, `WHERE character_id=? AND status='active' AND (content LIKE '%t1%' OR ...)` (case-insensitive via SQLite LIKE default), ordered by number of matching terms desc then rowid desc (compute match count in SQL as a sum of `CASE WHEN content LIKE ...`), capped at limit. `ftsSearch` now delegates here when `!db.hasFts5` OR the FTS query throws (soft-degrade, mirroring `retrieval.py`'s try/except).

- [ ] **Step 1: Failing tests.** `RrfTest`: hand-computed 3-list example asserting exact scores (e.g. lists `[[a,b],[b,a],[b]]`, k=60, weights `[1,1,2]` → score(b)=1/61+1/62·… compute the literal expected doubles in the test); empty lists → empty. `RetrievalTest`: **the golden parity test** —
```kotlin
@Test
fun `retrieval ranking matches python golden`() {
    val golden = Json.parseToJsonElement(Files.readString(fixtures.resolve("retrieval_golden.json"))).jsonObject
    val frozenNow = java.time.Instant.parse(golden["frozen_now_iso"]!!.jsonPrimitive.content)
    val copy = Files.createTempFile("conf", ".db")
    Files.copy(fixtures.resolve("conformance.db"), copy, REPLACE_EXISTING)
    Db(copy.toString()).use { db ->
        Schema.init(db); val storage = Storage(db)
        for (case in golden["cases"]!!.jsonArray.map { it.jsonObject }) {
            val qvec = case["query_embedding"]!!.jsonArray.map { it.jsonPrimitive.float }.toFloatArray()
            val r = MemoryRetriever(storage, StubEmbedder(qvec), "conformance-char", CoreConfig(), clock = { frozenNow })
            val got = r.retrieve(case["query"]!!.jsonPrimitive.content, limit = 10,
                relationshipTarget = case["relationship_target"]?.jsonPrimitive?.contentOrNull)
                .map { it["id"] as String }
            assertEquals(case["expected_ids"]!!.jsonArray.map { it.jsonPrimitive.content }, got,
                "query=${case["query"]}")
        }
    }
}
```
plus: LIKE fallback returns the keyword-matching memory when FTS is bypassed (test by calling `likeSearch` directly and by a retriever wired to a Storage whose ftsSearch was forced down the fallback path — simplest: add an internal `Storage.forceLikeFallback: Boolean` test hook, default false); blank query skips semantic list but still returns results.

- [ ] **Step 2-4: fail → implement → iterate until golden-parity passes for ALL cases.** If a case mismatches, diff the per-strategy ranked lists against Python (add temporary debug output; the usual culprits: tie-break order, hours computation, tier boost application). Do not loosen the assertion. **Step 5: Commit** `"feat(kotlin): RRF + retrieval with python golden-ranking parity + LIKE fallback (task 8)"`

---

### Task 9: MemoryStore + observe event write

**Files:**
- Create: `MemoryStore.kt`; Test: `MemoryStoreTest.kt`
- Python source (normative): `memory/store.py`, `character.py::observe` (storage part only)

**Interfaces:**
- Produces: `class MemoryStore(val storage: Storage, val embedder: EmbeddingProvider, val characterId: String)` — `fun add(content: String, tier: String = "buffer", role: String? = null, sessionId: String? = null, importance: Double = 0.5, metadata: String = "{}"): Row` (embed → `guardEmbeddingDimension(storage.db, vec, embedder.model)` → saveMemory with UUID id); `fun addWithoutEmbedding(...): Row`; `get/getAll/count/touch/archive`; `fun needsConsolidation(threshold: Int = 100): Boolean` (buffer count >= threshold); `fun recordEvent(event: String, source: String = "world", importance: Double = 0.6, sessionId: String? = null): Row` — stores `"[Event] $event"` with metadata `{"source": source}` (the deterministic half of Python `observe()`; LLM assessment is C2/game).

- [ ] **Step 1: Failing tests** — add() stamps meta on first embedded write and rejects a wrong-dim embedder afterwards; recordEvent content prefix + metadata + importance; needsConsolidation threshold boundary. **Step 2-4: fail → implement → pass. Step 5: Commit** `"feat(kotlin): memory store with dimension guard + observe event write (task 9)"`

---

### Task 10: Persona prompt assembly (byte parity) + memory formatting

**Files:**
- Create: `Persona.kt`, `PromptAssembler.kt`; Test: `PersonaTest.kt`
- Python source (normative): `persona/model.py::build_system_prompt`, `character.py::_build_context`, `_format_memories`

**Interfaces:**
- Produces: `class PersonaModel(val persona: JsonObject, val birthdate: String? = null)` with `fun buildSystemPrompt(): String` — transcribe the Python assembly exactly (hard/temporal/soft levels, title-cased keys, age line arithmetic, ordering). `object PromptAssembler { fun formatMemories(memories: List<Row>): String; fun buildContext(stablePrefix: String, stateDescription: String, memories: List<Row>, relationshipContext: String?, history: List<ChatMessage>, userMessage: String, contextConfig: ContextConfig): List<ChatMessage> }` — message 0 = stable prefix system message (byte-identical across turns); message 1 = system volatile block: stateDescription + relationshipContext + formatMemories, included only while total chars fit `contextConfig.totalTokens * 4`, shed in the Python priority order (memories partial-include first; conversation compression is NOT ported — when history alone exceeds budget, truncate oldest turns and note it in KDoc as the C1 simplification); then history; then user message. C1 API adaptation (deliberate): `stateDescription` is caller-provided — the game's deterministic sim owns creature state; there is no emotion/arc engine here.

- [ ] **Step 1: Failing tests** — **golden parity**: for each persona in `fixtures/prompts_golden.json`, `PersonaModel(persona, birthdate).buildSystemPrompt()` equals the Python string EXACTLY (assertEquals on the full string — whitespace included); `formatMemories` output equals the fixture's formatted string for the fixed memory list (incl. `[tier]` tags, `(uncertain)` suffix, 200-char truncation, provenance header); buildContext: message 0 stable across two calls with different states, over-budget memories get shed before history.

- [ ] **Step 2-4: fail → implement → iterate to byte parity. Step 5: Commit** `"feat(kotlin): persona prompt byte-parity + context assembly (task 10)"`

---

### Task 11: ConsolidationEngine + belief reinforce

**Files:**
- Create: `Consolidation.kt` (+ `reinforceMemory` in it or `MemoryStore.kt`); Test: `ConsolidationTest.kt`
- Python source (normative): `memory/consolidation.py` (clustering, summarize, `drain`, no-progress break order: budget → dryRun → archived==0), `memory/belief.py::reinforce`

**Interfaces:**
- Produces: `class ConsolidationEngine(val store: MemoryStore, val storage: Storage, val llm: LlmProvider, val config: CoreConfig)` — `fun needsConsolidation(): Boolean`; `fun consolidate(dryRun: Boolean = false): ConsolidationReport` (cluster buffer memories by cosine >= clusteringSimilarity, LLM-summarize each cluster via generateJsonRobust/generate per the Python prompt, write core summary through the store path so `guardEmbeddingDimension` applies — the Phase B fix — archive members); `fun drain(budget: Budget): ConsolidationReport` with the exact Python break order (budget exhausted → dryRun → no progress). `fun reinforceMemory(storage: Storage, memoryId: String): Double` — port `belief.py::reinforce` arithmetic exactly; `data class Budget(val limit: Int?)` with `fun take(n: Int = 1): Boolean`, `val used: Int`, `val remaining: Int?` lives here (Maintenance imports it).
- Consumes: `fixtures/reinforce_golden.json` — the ported arithmetic must reproduce all 6 golden `(importance, certainty) → result` rows.

- [ ] **Step 1: Failing tests** — reinforce golden table (all 6 rows exact to 1e-9); Budget semantics (take beyond limit returns false WITHOUT consuming; limit=null unlimited); consolidation with FakeLlm: seed 6 similar + 2 dissimilar buffer memories (StubEmbedder-driven vectors chosen so clustering is deterministic), consolidate → core summary rows exist, members archived, dissimilar untouched; idempotency: second consolidate run does nothing (report shows zero clusters); drain respects an exhausted budget (0 LLM calls made).
- [ ] **Step 2-4: fail → implement → pass. Step 5: Commit** `"feat(kotlin): consolidation engine + belief reinforce with golden arithmetic (task 11)"`

---

### Task 12: MaintenanceRunner (9 jobs) + CallbackEngine

**Files:**
- Create: `Maintenance.kt`, `Callbacks.kt`; Test: `MaintenanceTest.kt`, `CallbacksTest.kt`
- Python source (normative): `maintenance.py` (job list/order, per-job thresholds, report shape, error isolation, 0.51 sentinel nudge), `callbacks.py` (VALID_KINDS coercion, ready-cap expiry, consume-on-use, freshness strings)

**Interfaces:**
- Produces: `class MaintenanceRunner(val characterId: String, val storage: Storage, val store: MemoryStore, val consolidator: ConsolidationEngine, val llm: LlmProvider, val embedder: EmbeddingProvider, val config: CoreConfig, val budget: Budget = Budget(config.maintenance.maxLlmCallsPerRun))` — `val DEFAULT_JOBS = listOf("consolidate","buffer_hygiene","score_importance","dedup","reinforce","contradictions","reflect","evolve","callbacks")`; `fun run(jobs: List<String>? = null): MaintenanceReport` (`data class MaintenanceReport(val characterId: String, val jobs: Map<String, JobResult>, val llmCallsUsed: Int)`, `data class JobResult(val status: String, val detail: Map<String, Any?> = emptyMap(), val llmCalls: Int = 0, val durationMs: Double = 0.0, val error: String? = null)`). Jobs transcribed from `maintenance.py` with the exact thresholds from MaintenanceConfig; a job exception → status "failed", never aborts the run; unknown job name → failed with `"unknown job: <name>"`. **`reflect` and `evolve` return `status="skipped", detail={"reason":"unsupported in kotlin-core"}`** (documented C1 scope cut). `score_importance` keeps the Python sentinel discipline: candidates are `importance == 0.5` exactly; a genuine 5/10 result is stored as **0.51**.
- `class CallbackEngine(val characterId: String, val storage: Storage, val llm: LlmProvider, val config: CoreConfig)` — `fun get(limit: Int = 3): List<Row>` (adds computed `"freshness"` "Nd"/"Nh"/"Nm" string); `fun refresh(budget: Budget? = null, limit: Int? = null): Int` (one generateJsonRobust call; kind coerced into `("open_thread","callback","milestone","curiosity")` else "callback"; after insert enforce `callbacksReadyCap` by expiring lowest-salience overflow); `fun composeInitiation(occasion: String = "greeting", stablePrefix: String, stateDescription: String): InitiationResult` (`data class InitiationResult(val text: String, val callback: Row?)`) — top ready callback (if any) paraphrase-not-quote instruction, `llm.generate(temperature=0.8, maxTokens=150)`, mark consumed. C1 adaptation: system prompt = caller-provided stablePrefix + stateDescription (no emotion engine).

- [ ] **Step 1: Failing tests.** Maintenance: full run with FakeLlm over a seeded DB → report has all 9 job keys in order, reflect/evolve skipped, llmCallsUsed <= budget; `buffer_hygiene` archives an old low-importance buffer row (insert with explicit created_at 20 days ago) but not a high-importance one; `score_importance` only touches 0.5-sentinel rows and stores 0.51 for a scored 5 (FakeLlm canned array of 5s), and a re-run finds zero candidates (idempotency); `dedup` archives the lower-importance of two >=0.92-cosine cores (StubEmbedder-crafted vectors) and reinforces the kept one; budget exhaustion: `Budget(0)` → all LLM jobs report skipped/0-calls, pure jobs still run; one job throwing (inject via a Storage subclass overriding getMemories to throw for tier="core") → that job failed, later jobs still ran. Callbacks: refresh persists hooks with coerced kinds; cap expiry expires the lowest salience beyond 10; get() freshness format; composeInitiation returns text, consumes the top callback (status flips, second call gets callback=null with FakeLlm still generating).
- [ ] **Step 2-4: fail → implement → pass. Step 5: Commit** `"feat(kotlin): maintenance runner (9 jobs) + callback engine (task 12)"`

---

### Task 13: WovenImprintCore facade + docs + final suite

**Files:**
- Create: `WovenImprintCore.kt`; Test: `WovenImprintCoreTest.kt`; Modify: `kotlin/README.md`, `docs/SCHEMA.md` (one line), `CHANGELOG.md`
- Python analog: `engine.py` (shape only — no LLM-seeded bedrock in C1; characters are created with persona JSON, seeds via `MemoryStore.add`)

**Interfaces:**
- Produces: `class WovenImprintCore(dbPath: String, val llm: LlmProvider, val embedder: EmbeddingProvider, val config: CoreConfig = CoreConfig()) : AutoCloseable` — opens Db, `Schema.init`, exposes `val storage: Storage`; `fun createCharacter(id: String, name: String, persona: JsonObject, birthdate: String? = null): CharacterHandle`; `fun getCharacter(id: String): CharacterHandle` (throws NoSuchElementException if missing); `fun runMaintenance(characterId: String? = null, jobs: List<String>? = null, budget: Int? = null): List<MaintenanceReport>` (single shared Budget across characters, default maxLlmCallsPerRun — mirrors Python Engine). `class CharacterHandle` bundling per-character `store: MemoryStore`, `retriever: MemoryRetriever`, `callbacks: CallbackEngine`, `persona: PersonaModel` plus `fun recordEvent(...)`, `fun addTurn(sessionId, seq, role, content)` convenience. All synchronous — no background worker in C1 (the game loop / C2 owns threading; KDoc states the Python `background=False` equivalence).

- [ ] **Step 1: Failing integration test** — end-to-end on `:memory:`: create character → add 8 memories via store → retrieve returns ranked results → recordEvent → runMaintenance full report → callbacks.refresh + composeInitiation round-trip → reopen the SAME db file (temp file, not :memory:) with a new WovenImprintCore and assert memories + callbacks survive.
- [ ] **Step 2-4: fail → implement → pass, then FULL suites:** `./gradlew :core:test` all green AND `uv run --extra dev pytest -q` unchanged (374/6).
- [ ] **Step 5: Docs** — `kotlin/README.md`: module map, what C1 covers/excludes (verbatim from Global Constraints scope-cut line), fixture regeneration command, includeBuild usage snippet for the game. `docs/SCHEMA.md`: update the LIKE-fallback line from "is a Phase C requirement — not implemented today" to note the Kotlin implementation (`kotlin/core .../Retrieval.kt`, Python still constructor-requires FTS5). `CHANGELOG.md`: C1 entry.
- [ ] **Step 6: Commit** `"feat(kotlin): WovenImprintCore facade, docs, C1 complete (task 13)"`

---

## Self-review notes (author-run)

- Spec coverage vs Phase C §4 scope statement: storage ✓ (T3-T6), retrieval math ✓ (T8), prompt assembly ✓ (T10), callback reads ✓ (T12), batch jobs ✓ logic-level (T11-T12; WorkManager mapping = C2), reference bridge = C2 by design, resilience parity = C2 (documented in Global Constraints). SCHEMA.md obligations: endianness ✓ T4, dimension guard ✓ T4/T9, FTS5 degrade ✓ T8, timestamps ✓ Global+T5, migration protocol + newer-DB refusal ✓ T3, semver ✓ T3.
- Phase B deferred items: LIKE fallback ✓ T8; demo seed-import meta-stamp gap is a PYTHON-side fix — out of C1's Kotlin scope, remains on the backlog (noted here so it isn't silently lost).
- Porting hazards from the dossier honored: RLock invariant (T5 stress test), stale "Strategy 5" tier-list trap (T8 transcribes the 4-or-5-list algorithm), session-id capture semantics (no background worker in C1 — hazard deferred to C2 with a KDoc note in T13), >1-system-message provider hazard (C2 bridge concern; PromptAssembler keeps the two-message shape deliberately).
