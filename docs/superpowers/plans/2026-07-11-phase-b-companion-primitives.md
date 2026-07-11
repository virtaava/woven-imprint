# Phase B — Companion Primitives: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give woven-imprint the primitives a companion app lives on: an offline batch-maintenance pipeline (nightly jobs), a paraphrased callback-surfacing API, proactive initiation, world-event ingestion, a health surface that makes small-model degradation visible, and a schema meta table for portability.

**Architecture:** All heavy/multi-pass LLM work moves into `MaintenanceRunner` (new `maintenance.py`) — budgeted, idempotent, per-character jobs callable headless via `engine.run_maintenance()` and `woven-imprint maintain`. Callbacks are *generated* in batch (new `callbacks.py` + `callbacks` table) and *read* instantly. `Character` gains `observe()`, `health()`, `get_callbacks()`, `compose_initiation()`. New DB objects arrive via migration v4.

**Tech Stack:** Python ≥3.11, sqlite3, stdlib only; pytest with `tests/helpers.py` FakeLLM/FakeEmbedder; ruff/pyright/eval CI.

**Spec:** `docs/superpowers/specs/2026-07-10-companion-grade-evolution-design.md` §3 (B1–B6).
**Phase-A carry-overs folded in (from `.superpowers/sdd/progress.md`):** health counters must count *subsystem-internal* failures (the `logger.debug` swallow sites); bookkeeping must capture `session_id` at submit time; config needs a separate `embedding_base_url` (single shared `base_url` blocks split endpoints like :11800 chat / :11801 embed).

## Global Constraints

- Python ≥3.11; CI = ruff lint+format (`src/ tests/`), pyright, pytest 3.11–3.13 `-x`, `python eval/run_eval.py`. Green after every task.
- Core runtime deps stay exactly `requests` + `pyyaml`; new code stdlib-only.
- Public API additive; existing signatures keep working.
- New knobs go through `config.py` dataclasses + `_apply_env` env vars; every new key also lands in `save_default_config()`'s template string.
- Paraphrase-not-verbatim is a HARD RULE for callbacks (HAI '23 finding): prompts must instruct natural in-character reference, never quoting stored text.
- The LLM narrates/decides; deterministic code owns state. No job may let the LLM invent memory content that isn't grounded in provided sources.
- Test commands: `cd ~/sona/projects/woven-imprint && venv/bin/python -m pytest tests/ -x -q`; lint `venv/bin/ruff check src/ tests/` + `venv/bin/ruff format src/ tests/`; typecheck `venv/bin/python -m pyright`; eval `venv/bin/python eval/run_eval.py`.
- Commit after every task, conventional style.

## Locked interfaces (used across tasks — exact names)

```python
# storage/sqlite.py additions (Task 1)
save_callback(cb: dict) -> None
get_callbacks(character_id: str, status: str = "ready", limit: int = 10) -> list[dict]
mark_callback(callback_id: str, status: str) -> None
meta_get(key: str) -> str | None
meta_set(key: str, value: str) -> None
set_memory_importance(memory_id: str, importance: float) -> None
archive_memories_batch(memory_ids: list[str]) -> None

# llm/base.py (Task 3)
generate_json_robust(messages, temperature: float = 0.3) -> dict | list   # 1 bounded retry at temp 0.1

# character.py (Tasks 4, 5, 10, 11)
health() -> dict
observe(event: str, source: str = "world", importance: float | None = None, user_id: str | None = None) -> dict
get_callbacks(limit: int = 3) -> list[dict]
refresh_callbacks(budget: "Budget | None" = None) -> int
compose_initiation(occasion: str = "greeting") -> dict
_note_failure(subsystem: str, exc: Exception) -> None
_note_success(subsystem: str) -> None

# maintenance.py (Tasks 7-9)
class Budget: __init__(limit: int | None); take(n: int = 1) -> bool; used: int; remaining: int | None
class MaintenanceRunner:
    DEFAULT_JOBS = ["consolidate", "buffer_hygiene", "score_importance", "dedup", "reinforce", "contradictions", "reflect", "evolve", "callbacks"]
    __init__(character: Character, budget: Budget | None = None)
    run(jobs: list[str] | None = None) -> dict   # {"character_id", "jobs": {name: {"status": "ok"|"skipped"|"failed", "llm_calls": int, "duration_ms": float, ...details}}, "llm_calls_used": int}

# callbacks.py (Tasks 10-11)
class CallbackEngine:
    __init__(character: Character)
    refresh(budget: Budget | None = None, limit: int | None = None) -> int   # callbacks created
    get(limit: int = 3) -> list[dict]   # instant DB read; adds "freshness" like "2d"
    compose_initiation(occasion: str = "greeting", budget: Budget | None = None) -> dict  # {"text": str, "callback": dict | None}

# engine.py (Task 9)
run_maintenance(character_id: str | None = None, jobs: list[str] | None = None, budget: int | None = None) -> list[dict]

# config.py (Task 2)
@dataclass class MaintenanceConfig:  # section name "maintenance"
    max_llm_calls_per_run: int = 50
    consolidate_chunk_size: int = 500
    buffer_ttl_days: int = 14
    buffer_hygiene_max_importance: float = 0.55
    importance_scoring_batch: int = 30
    dedup_scan_limit: int = 200
    dedup_similarity: float = 0.92
    reinforce_similarity: float = 0.85
    contradiction_candidate_similarity: float = 0.70
    contradiction_max_pairs: int = 10
    reflect_importance_sum: float = 12.0
    callbacks_refresh_limit: int = 5
    callbacks_ready_cap: int = 10
    callbacks_refresh_on_session_end: bool = True
LLMConfig.embedding_base_url: str | None = None
```

## File Structure

- **Create** `src/woven_imprint/maintenance.py` — Budget + MaintenanceRunner + all batch jobs (B1, B5-importance)
- **Create** `src/woven_imprint/callbacks.py` — CallbackEngine (B2 generation/read, B3 initiation)
- **Create** `docs/SCHEMA.md` — portable schema contract (B6)
- **Modify** `storage/sqlite.py` — migration v4 (`callbacks`, `meta`), new CRUD (Task 1)
- **Modify** `memory/store.py` — embedding-dimension guard via meta (Task 1)
- **Modify** `config.py` — MaintenanceConfig + embedding_base_url + env vars (Task 2)
- **Modify** `providers.py` — embedding factory honors embedding_base_url (Task 2)
- **Modify** `llm/base.py` — `generate_json_robust` (Task 3); call sites in `character.py`, `persona/emotion.py`, `narrative/arc.py`, `persona/growth.py` switch to it
- **Modify** `character.py` — health counters + session_id-at-submit (Task 4), `observe()` (Task 5), callback delegates + end_session refresh hook (Tasks 10–11)
- **Modify** `persona/emotion.py` — `assess_event()` (Task 5)
- **Modify** `memory/consolidation.py` — chunked drain + dry_run fix (Task 6)
- **Modify** `engine.py`, `cli.py` — run_maintenance + `maintain` subcommand (Task 9)
- **Modify** `server/sidecar.py`, `server/services.py`, `server/demo.py`, `mcp_server.py` — expose callbacks/health/observe/maintain (Task 12)
- **Tests:** `tests/test_maintenance.py`, `tests/test_callbacks.py`, `tests/test_observe.py`, `tests/test_health.py`, plus additions to `test_storage.py`, `test_consolidation.py`, `test_services.py`, `test_sidecar.py`, `test_providers.py`, `test_character.py`

---

### Task 1: Migration v4 — `callbacks` + `meta` tables, storage CRUD, embedding-dimension guard

**Files:**
- Modify: `src/woven_imprint/storage/sqlite.py` (`_MIGRATIONS` + new methods)
- Modify: `src/woven_imprint/memory/store.py` (dimension guard in `add`)
- Test: `tests/test_storage.py` (additions), `tests/test_metadata_guard.py` (new)

**Interfaces:**
- Produces: the storage methods listed in "Locked interfaces"; migration v4; `meta` keys used later: `schema_semver` ("0.6.0-dev"), `embedding_model`, `embedding_dimensions`.
- Consumes: existing migration machinery (`_MIGRATIONS` dict → `executescript`), `generate_id` from `utils/text.py`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_storage.py` (follow the file's existing `SQLiteStorage(":memory:")` style):

```python
class TestCallbacksTable:
    def test_callback_roundtrip(self):
        s = SQLiteStorage(":memory:")
        s.save_character("c1", "Cara", {})
        s.save_callback(
            {
                "id": "cb-1",
                "character_id": "c1",
                "kind": "open_thread",
                "hook": "You mentioned an interview coming up — how did it go?",
                "source_memory_ids": ["m1", "m2"],
                "salience": 0.8,
            }
        )
        ready = s.get_callbacks("c1")
        assert len(ready) == 1
        cb = ready[0]
        assert cb["kind"] == "open_thread"
        assert cb["source_memory_ids"] == ["m1", "m2"]
        assert cb["status"] == "ready"

    def test_mark_and_filter_status(self):
        s = SQLiteStorage(":memory:")
        s.save_character("c1", "Cara", {})
        for i, sal in enumerate([0.3, 0.9]):
            s.save_callback(
                {"id": f"cb-{i}", "character_id": "c1", "kind": "curiosity",
                 "hook": f"hook {i}", "salience": sal}
            )
        ready = s.get_callbacks("c1")
        assert [c["id"] for c in ready] == ["cb-1", "cb-0"]  # salience DESC
        s.mark_callback("cb-1", "consumed")
        assert [c["id"] for c in s.get_callbacks("c1")] == ["cb-0"]

    def test_meta_roundtrip(self):
        s = SQLiteStorage(":memory:")
        assert s.meta_get("embedding_model") is None
        s.meta_set("embedding_model", "nomic-embed-text")
        s.meta_set("embedding_model", "other")  # upsert
        assert s.meta_get("embedding_model") == "other"

    def test_set_importance_and_batch_archive(self):
        s = SQLiteStorage(":memory:")
        s.save_character("c1", "Cara", {})
        for i in range(3):
            s.save_memory({"id": f"m{i}", "character_id": "c1", "tier": "buffer",
                           "content": f"mem {i}", "importance": 0.5})
        s.set_memory_importance("m0", 0.9)
        assert s.get_memory("m0")["importance"] == 0.9
        s.archive_memories_batch(["m1", "m2"])
        active = s.get_memories("c1", tier="buffer")
        assert [m["id"] for m in active] == ["m0"]
```

Create `tests/test_metadata_guard.py`:

```python
import pytest

from tests.helpers import make_test_engine


def test_meta_dimensions_set_on_first_embed():
    engine = make_test_engine()
    char = engine.create_character("Dima")
    char.memory.add("hello world", tier="buffer")
    assert engine.storage.meta_get("embedding_dimensions") == "50"  # FakeEmbedder dims


def test_dimension_mismatch_raises():
    engine = make_test_engine()
    char = engine.create_character("Dima")
    char.memory.add("hello", tier="buffer")
    engine.storage.meta_set("embedding_dimensions", "768")  # simulate model swap
    with pytest.raises(ValueError, match="dimension"):
        char.memory.add("world", tier="buffer")
```

- [ ] **Step 2: Run to verify failure**

Run: `venv/bin/python -m pytest tests/test_storage.py::TestCallbacksTable tests/test_metadata_guard.py -v`
Expected: FAIL with AttributeError (`save_callback` missing).

- [ ] **Step 3: Implement migration v4**

In `_MIGRATIONS` (`storage/sqlite.py`), add:

```python
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
```

- [ ] **Step 4: Implement storage methods**

Add near the sessions section of `SQLiteStorage`:

```python
    # ── Callbacks ─────────────────────────────────────────────

    def save_callback(self, cb: dict) -> None:
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

    def get_callbacks(self, character_id: str, status: str = "ready", limit: int = 10) -> list[dict]:
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
        self._conn.execute(
            "UPDATE callbacks SET status = ?, "
            "consumed_at = CASE WHEN ? = 'consumed' THEN datetime('now') ELSE consumed_at END "
            "WHERE id = ?",
            (status, status, callback_id),
        )
        self._commit()

    # ── Meta ──────────────────────────────────────────────────

    def meta_get(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def meta_set(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(value)),
        )
        self._commit()

    # ── Memory maintenance helpers ────────────────────────────

    def set_memory_importance(self, memory_id: str, importance: float) -> None:
        self._conn.execute(
            "UPDATE memories SET importance = ? WHERE id = ?",
            (max(0.0, min(1.0, importance)), memory_id),
        )
        self._commit()

    def archive_memories_batch(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        self._conn.executemany(
            "UPDATE memories SET status = 'archived' WHERE id = ?",
            [(mid,) for mid in memory_ids],
        )
        self._commit()
```

Also delete callbacks in `delete_character()`: add `self._conn.execute("DELETE FROM callbacks WHERE character_id = ?", (char_id,))` alongside the existing deletes (session_turns precedent), and extend the existing delete test to assert callbacks are gone.

- [ ] **Step 5: Embedding-dimension guard**

In `memory/store.py` `MemoryStore.add`, after computing the embedding and before saving:

```python
        # Guard against mixed embedding models corrupting cosine math (meta contract)
        if embedding:
            known = self.storage.meta_get("embedding_dimensions")
            if known is None:
                self.storage.meta_set("embedding_dimensions", str(len(embedding)))
                from ..config import get_config

                self.storage.meta_set("embedding_model", get_config().llm.embedding_model)
            elif int(known) != len(embedding):
                raise ValueError(
                    f"Embedding dimension mismatch: DB stores {known}-d vectors, "
                    f"got {len(embedding)}-d. Mixed embedding models corrupt retrieval — "
                    f"re-embed the database or restore the original embedding model."
                )
```

(Adapt to the actual local variable name for the computed vector in `add` — read the method first.)

- [ ] **Step 6: Run tests, fix fallout, commit**

Run: `venv/bin/python -m pytest tests/ -x -q` — the dimension guard may surprise tests that swap embedders mid-DB; fix such tests by using consistent embedders (never weaken the guard). Then eval + ruff + pyright.

```bash
git add src/woven_imprint/storage/sqlite.py src/woven_imprint/memory/store.py tests/test_storage.py tests/test_metadata_guard.py
git commit -m "feat: migration v4 — callbacks + meta tables, importance/archive helpers, dimension guard (B6)"
```

---

### Task 2: MaintenanceConfig + split embedding base_url

**Files:**
- Modify: `src/woven_imprint/config.py`
- Modify: `src/woven_imprint/providers.py`
- Test: `tests/test_providers.py` (additions), `tests/test_config_maintenance.py` (new)

**Interfaces:**
- Produces: `MaintenanceConfig` dataclass exactly as in "Locked interfaces", wired as `WovenConfig.maintenance`; `LLMConfig.embedding_base_url: str | None = None`; env vars `WOVEN_IMPRINT_EMBEDDING_BASE_URL` → `llm.embedding_base_url`, `WOVEN_IMPRINT_MAINTENANCE_BUDGET` → `maintenance.max_llm_calls_per_run`. `create_embedding()` passes `embedding_base_url or base_url` to the OpenAI embedding provider (and to Ollama's host only if the Ollama embedding class accepts a base_url — read `embedding/ollama.py` and `providers.py:80-98` first; if Ollama embeds via `ollama_host`, leave that path untouched).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_config_maintenance.py`:

```python
from woven_imprint.config import WovenConfig, get_config, reload_config


def test_maintenance_defaults():
    cfg = WovenConfig()
    m = cfg.maintenance
    assert m.max_llm_calls_per_run == 50
    assert m.consolidate_chunk_size == 500
    assert m.buffer_ttl_days == 14
    assert m.dedup_similarity == 0.92
    assert m.reflect_importance_sum == 12.0
    assert m.callbacks_ready_cap == 10
    assert m.callbacks_refresh_on_session_end is True


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("WOVEN_IMPRINT_MAINTENANCE_BUDGET", "7")
    monkeypatch.setenv("WOVEN_IMPRINT_EMBEDDING_BASE_URL", "http://127.0.0.1:11801/v1")
    try:
        cfg = reload_config()
        assert cfg.maintenance.max_llm_calls_per_run == 7
        assert cfg.llm.embedding_base_url == "http://127.0.0.1:11801/v1"
    finally:
        monkeypatch.delenv("WOVEN_IMPRINT_MAINTENANCE_BUDGET")
        monkeypatch.delenv("WOVEN_IMPRINT_EMBEDDING_BASE_URL")
        reload_config()
```

Append to `tests/test_providers.py` (guarded like its existing openai test with `pytest.importorskip("openai")`):

```python
def test_embedding_base_url_split(monkeypatch):
    pytest.importorskip("openai")
    from woven_imprint.config import reload_config
    from woven_imprint.providers import create_embedding

    monkeypatch.setenv("WOVEN_IMPRINT_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("WOVEN_IMPRINT_BASE_URL", "http://chat:1/v1")
    monkeypatch.setenv("WOVEN_IMPRINT_EMBEDDING_BASE_URL", "http://embed:2/v1")
    monkeypatch.setenv("WOVEN_IMPRINT_API_KEY_LLM", "sk-test")
    try:
        cfg = reload_config()
        emb = create_embedding(cfg)
        assert "embed:2" in str(emb.client.base_url)
    finally:
        for var in ("WOVEN_IMPRINT_EMBEDDING_PROVIDER", "WOVEN_IMPRINT_BASE_URL",
                    "WOVEN_IMPRINT_EMBEDDING_BASE_URL", "WOVEN_IMPRINT_API_KEY_LLM"):
            monkeypatch.delenv(var)
        reload_config()
```

(Check the OpenAI embedding class's attribute for the client before asserting — read `embedding/openai.py`; adjust the assertion to its real structure.)

- [ ] **Step 2: Run to verify failure** — `venv/bin/python -m pytest tests/test_config_maintenance.py -v` → FAIL (`maintenance` attribute missing).

- [ ] **Step 3: Implement**

Add the `MaintenanceConfig` dataclass (fields exactly as in Locked interfaces) to `config.py`; add `maintenance: MaintenanceConfig = field(default_factory=MaintenanceConfig)` to `WovenConfig`; add `embedding_base_url: str | None = None` to `LLMConfig`; add both env_map entries; add a full `maintenance:` section + `# embedding_base_url: null` line to `save_default_config()`'s template.

In `providers.py` `create_embedding`, for the `openai` branch pass `base_url=cfg.llm.embedding_base_url or cfg.llm.base_url`.

- [ ] **Step 4: Run tests, commit**

```bash
git add src/woven_imprint/config.py src/woven_imprint/providers.py tests/test_config_maintenance.py tests/test_providers.py
git commit -m "feat: MaintenanceConfig section + split embedding_base_url (B1/B6 groundwork)"
```

---

### Task 3: `generate_json_robust` — uniform bounded JSON retry

**Files:**
- Modify: `src/woven_imprint/llm/base.py`
- Modify call sites: `src/woven_imprint/character.py` (`_extract_memories` L~1039, `_update_relationship` L~1080), `src/woven_imprint/persona/emotion.py` (`assess`), `src/woven_imprint/narrative/arc.py` (`analyze_beat`), `src/woven_imprint/persona/growth.py` (`detect_growth`)
- Test: `tests/test_streaming.py`? No — `tests/test_json_robust.py` (new)

**Interfaces:**
- Produces: `LLMProvider.generate_json_robust(messages, temperature=0.3)` — calls `generate_json`; on `ValueError` retries ONCE at `temperature=0.1`; re-raises if the retry also fails. All five listed call sites switch `generate_json(` → `generate_json_robust(`. `ConsistencyChecker` keeps its own existing retry (do not touch).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_json_robust.py
import pytest

from tests.helpers import FakeLLM


class FlakyJSONLLM(FakeLLM):
    def __init__(self, fail_times: int):
        super().__init__()
        self.fail_times = fail_times
        self.json_calls = 0
        self.temps: list[float] = []

    def generate_json(self, messages, temperature=0.3, **kw):
        self.json_calls += 1
        self.temps.append(temperature)
        if self.json_calls <= self.fail_times:
            raise ValueError("bad json")
        return {"ok": True}


def test_retries_once_at_low_temp():
    llm = FlakyJSONLLM(fail_times=1)
    from woven_imprint.llm.base import LLMProvider

    result = LLMProvider.generate_json_robust(llm, [{"role": "user", "content": "x"}])
    assert result == {"ok": True}
    assert llm.json_calls == 2
    assert llm.temps[1] == 0.1


def test_raises_after_retry_exhausted():
    llm = FlakyJSONLLM(fail_times=2)
    from woven_imprint.llm.base import LLMProvider

    with pytest.raises(ValueError):
        LLMProvider.generate_json_robust(llm, [{"role": "user", "content": "x"}])
    assert llm.json_calls == 2
```

- [ ] **Step 2: Run to verify failure** → AttributeError.

- [ ] **Step 3: Implement in `llm/base.py`** (non-abstract, after `generate_stream`):

```python
    def generate_json_robust(
        self, messages: list[dict[str, str]], temperature: float = 0.3
    ) -> dict[str, Any] | list[Any]:
        """generate_json with one bounded retry at temperature 0.1 on parse failure.

        Uniform policy for subsystem JSON calls on weak/small models
        (mirrors the consistency checker's existing retry behavior).
        """
        try:
            return self.generate_json(messages, temperature=temperature)
        except ValueError:
            return self.generate_json(messages, temperature=0.1)
```

- [ ] **Step 4: Switch the five call sites**

In each listed file, replace the subsystem's `self.llm.generate_json(messages...)` (or `llm.generate_json(...)`) with `generate_json_robust`. Read each call site first; keep surrounding try/excepts unchanged. Note FakeLLM in `tests/helpers.py` inherits nothing — `generate_json_robust` lives on `LLMProvider`, and FakeLLM is duck-typed. **Make FakeLLM inherit `LLMProvider`?** No — smaller change: in the five call sites call via a module-level helper? Simplest correct approach: add to `tests/helpers.py` FakeLLM a passthrough method:

```python
    def generate_json_robust(self, messages, temperature=0.3, **kw):
        return self.generate_json(messages, temperature=temperature, **kw)
```

and in production call sites use `self.llm.generate_json_robust(...)` directly (all real providers inherit it from `LLMProvider`). Also add the same passthrough to `eval/`'s `EvalLLM` classes (grep `eval/bench_*.py` for `generate_json` and mirror).

- [ ] **Step 5: Run full suite + eval, commit**

```bash
git add src/woven_imprint/llm/base.py src/woven_imprint/character.py src/woven_imprint/persona/emotion.py src/woven_imprint/narrative/arc.py src/woven_imprint/persona/growth.py tests/test_json_robust.py tests/helpers.py eval/
git commit -m "feat: generate_json_robust — uniform bounded JSON retry for subsystems (B5)"
```

---

### Task 4: Health surface + session_id-at-submit

**Files:**
- Modify: `src/woven_imprint/character.py`
- Test: `tests/test_health.py` (new)

**Interfaces:**
- Produces: `Character.health() -> dict` shaped:
  ```python
  {"subsystems": {name: {"success": int, "failure": int, "last_error": str | None}},
   "worker": {"alive": bool, "pending": int} | None,
   "generated_at": iso_str}
  ```
  Instrumented subsystem names: `"emotion"`, `"arc"`, `"extraction"`, `"relationship"`, `"consistency"`, `"observe"` (observe wired in Task 5), `"callbacks"` (wired in Task 10). Helpers `_note_failure(subsystem, exc)` / `_note_success(subsystem)` update `self._health_counters: dict[str, dict]` (created in `__init__`; plain dict — worker thread + GIL, matches existing counter precedent in BackgroundWorker).
- Also: bookkeeping captures `session_id` at submit — `_run_subsystems_sequential`/`_run_subsystems_parallel` gain a `session_id: str | None = None` parameter threaded into `_extract_memories(..., session_id=...)`, which uses it instead of reading `self._session_id` at execution time. Both submit sites (`chat`, `chat_stream`) pass `self._session_id`.
- Carry-over fix: the swallow sites at `character.py` `_run_subsystems_sequential` (emotion ~L599, arc ~L603), `_extract_memories` (~L1042), `_update_relationship` (~L1091), and the consistency except in `chat` (~L290) each call `_note_failure(...)` in the except and `_note_success(...)` on the success path.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_health.py
from tests.helpers import FakeEmbedder, FakeLLM, make_test_engine
from woven_imprint.engine import Engine


class BrokenJSONLLM(FakeLLM):
    def generate_json(self, messages, temperature=0.3, **kw):
        raise ValueError("small model garbage")

    def generate_json_robust(self, messages, temperature=0.3, **kw):
        raise ValueError("small model garbage")


def test_health_counts_subsystem_failures():
    engine = Engine(db_path=":memory:", llm=BrokenJSONLLM(), embedding=FakeEmbedder())
    char = engine.create_character("Sick")
    char.background = False
    char.parallel = False
    char.enforce_consistency = False
    char.chat("hello", user_id="u1")

    h = char.health()
    assert h["subsystems"]["emotion"]["failure"] >= 1
    assert h["subsystems"]["relationship"]["failure"] >= 1
    assert "small model garbage" in h["subsystems"]["emotion"]["last_error"]
    assert h["worker"] is None  # background off, never created


def test_health_counts_successes():
    engine = make_test_engine()
    char = engine.create_character("Well")
    char.chat("hello", user_id="u1")
    h = char.health()
    assert h["subsystems"]["emotion"]["success"] >= 1
    assert h["subsystems"]["emotion"]["failure"] == 0


def test_session_id_captured_at_submit():
    """Facts must be tagged with the session active when the turn happened."""
    engine = make_test_engine()
    char = engine.create_character("Tagger")
    sid1 = char.start_session()
    char.chat("I adopted a cat named Viima")  # extraction turn (turn 0 % 3 == 0)
    char.flush()
    facts = [m for m in char.memory.get_all(tier="core") if m.get("session_id")]
    assert all(f["session_id"] == sid1 for f in facts)
```

- [ ] **Step 2: Run to verify failure** → AttributeError (`health` missing).

- [ ] **Step 3: Implement**

In `Character.__init__`: `self._health_counters: dict[str, dict] = {}`.

Add methods:

```python
    def _note_success(self, subsystem: str) -> None:
        entry = self._health_counters.setdefault(
            subsystem, {"success": 0, "failure": 0, "last_error": None}
        )
        entry["success"] += 1

    def _note_failure(self, subsystem: str, exc: Exception) -> None:
        entry = self._health_counters.setdefault(
            subsystem, {"success": 0, "failure": 0, "last_error": None}
        )
        entry["failure"] += 1
        entry["last_error"] = f"{type(exc).__name__}: {exc}"[:300]

    def health(self) -> dict:
        """Per-subsystem success/failure counters — makes silent small-model
        degradation visible (memory extraction failing = the character quietly
        stops learning; this surface is how an app notices)."""
        from datetime import datetime, timezone

        worker = None
        if self._worker is not None:
            worker = {
                "alive": self._worker.is_alive,
                "pending": self._worker._queue.unfinished_tasks,
            }
        return {
            "subsystems": {k: dict(v) for k, v in self._health_counters.items()},
            "worker": worker,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
```

Instrument the swallow sites (pattern — apply at each listed site, keeping the existing logger.debug):

```python
            try:
                self.emotion = self.emotion_engine.assess(message, response, self.emotion, self.name)
                self._note_success("emotion")
            except Exception as e:
                logger.debug("Emotion assessment failed: %s", e)
                self._note_failure("emotion", e)
```

For `_extract_memories`: success = the extraction attempt parsed (note success after the `generate_json_robust` result is used; failure in the existing `except (ValueError, KeyError)` — broaden to `except Exception` there while adding the note, so non-JSON errors also surface instead of crashing bookkeeping). Same broadening at `_update_relationship`. For consistency: in `chat()`'s consistency try/except, add notes.

Thread session_id: change signatures to `_run_subsystems_sequential(self, message, response, user_id, session_id=None)` and `_run_subsystems_parallel(..., session_id=None)`; `_extract_memories(self, user_msg, response, user_id, session_id=None)` uses `session_id if session_id is not None else self._session_id` for `memory.add(session_id=...)` and `belief.contradict(session_id=...)`. Update the two submit sites and the parallel pool submit, and `ingest()`'s direct `_extract_memories` call (passes its current `self._session_id`).

- [ ] **Step 4: Run full suite (expect minor fallout in tests asserting exact debug behavior — none should exist), commit**

```bash
git add src/woven_imprint/character.py tests/test_health.py
git commit -m "feat: Character.health() with subsystem counters; capture session_id at submit (B5)"
```

---

### Task 5: `observe()` — world-event ingestion

**Files:**
- Modify: `src/woven_imprint/character.py`
- Modify: `src/woven_imprint/persona/emotion.py`
- Test: `tests/test_observe.py` (new)

**Interfaces:**
- Produces: `Character.observe(event, source="world", importance=None, user_id=None) -> dict` (returns the stored memory dict). Stores `content=f"[Event] {event}"`, `tier="buffer"`, `role="event"`, `importance=importance or 0.6`, `metadata={"source": source, "user_id": user_id}`. When `user_id` is given, relationship assessment runs with an EVENT-shaped prompt (new private `_update_relationship_event(event, user_id)`); emotion assessment runs via new `EmotionEngine.assess_event(event, current, character_name) -> EmotionalState`. Assessments respect `self.background` (submitted to the worker with label `"observe"`) and `self.lightweight` (skip emotion). Health notes under `"observe"`.
- Consumes: `generate_json_robust` (Task 3), `_note_failure/_note_success` (Task 4).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_observe.py
from tests.helpers import make_test_engine


def test_observe_stores_event_memory():
    engine = make_test_engine()
    char = engine.create_character("Watcher")
    mem = char.observe("The keeper fed you moonberries", source="game_sim")
    assert mem["content"] == "[Event] The keeper fed you moonberries"
    assert mem["role"] == "event"
    assert mem["importance"] == 0.6
    stored = char.memory.get(mem["id"])
    assert stored["metadata"]["source"] == "game_sim"


def test_observe_no_dialogue_pair_pollution():
    """Events must NOT run the dialogue-pair fact extractor (empty-string pollution)."""
    engine = make_test_engine()
    char = engine.create_character("Watcher")
    before = char.llm.call_count
    char.observe("It rained all day")  # no user_id → no assessments at all
    assert char.llm.call_count == before  # zero LLM calls


def test_observe_with_user_updates_relationship():
    engine = make_test_engine()
    char = engine.create_character("Watcher")
    char.observe("Keeper stayed up all night nursing you back to health", user_id="keeper")
    char.flush()
    rel = char.get_relationship("keeper")
    assert rel is not None  # relationship row created + assessed


def test_observe_importance_override():
    engine = make_test_engine()
    char = engine.create_character("Watcher")
    mem = char.observe("You evolved into a fledgling!", importance=0.95)
    assert mem["importance"] == 0.95
```

- [ ] **Step 2: Run to verify failure** → AttributeError.

- [ ] **Step 3: Implement `EmotionEngine.assess_event`** in `persona/emotion.py` (mirror `assess`'s structure/validation — read it first):

```python
    def assess_event(
        self, event: str, current: EmotionalState, character_name: str
    ) -> EmotionalState:
        """Assess emotional impact of a world event (no dialogue pair)."""
        messages = [
            {
                "role": "system",
                "content": (
                    f"You assess how an event affects {character_name}'s emotional state. "
                    "Return JSON: {\"mood\": <label>, \"intensity\": 0.0-1.0, \"cause\": <short reason>}. "
                    f"Valid moods: {', '.join(EMOTION_LABELS)}. "
                    "Small events cause small shifts; keep continuity with the current mood."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Current mood: {current.mood} (intensity {current.intensity:.1f}).\n"
                    f"Event: {event}\n\nHow does {character_name} feel now? Return JSON."
                ),
            },
        ]
        try:
            result = self.llm.generate_json_robust(messages)
            if not isinstance(result, dict):
                result = {}
            mood = result.get("mood", current.mood)
            if mood not in EMOTION_LABELS:
                mood = "neutral"
            intensity = result.get("intensity", current.intensity)
            new = EmotionalState(
                mood=mood,
                intensity=max(0.0, min(1.0, float(intensity))),
                cause=str(result.get("cause", ""))[:200],
                turns_held=0,
            )
            return new
        except Exception:
            current.decay()
            return current
```

- [ ] **Step 4: Implement `Character.observe` + `_update_relationship_event`**

```python
    def observe(
        self,
        event: str,
        source: str = "world",
        importance: float | None = None,
        user_id: str | None = None,
    ) -> dict:
        """Record a world event as a memory — no dialogue pair, no generation.

        This is the API a deterministic game/sim uses to narrate ground truth
        into the character's memory ("Keeper fed you", "It rained all day").
        """
        if not self._session_id:
            self.start_session()

        from .config import get_config

        _cfg = get_config()
        if len(event) > _cfg.memory.max_message_length:
            event = event[: _cfg.memory.max_message_length]

        memory = self.memory.add(
            content=f"[Event] {event}",
            tier="buffer",
            role="event",
            session_id=self._session_id,
            importance=importance if importance is not None else 0.6,
            metadata={"source": source, "user_id": user_id},
        )

        if user_id:
            if self.background:
                self._get_worker().submit(
                    "observe", self._assess_event, event, user_id, self._session_id
                )
            else:
                self._assess_event(event, user_id, self._session_id)

        return memory

    def _assess_event(self, event: str, user_id: str, session_id: str | None) -> None:
        """Event-shaped emotion + relationship assessment (worker-safe)."""
        if not self.lightweight:
            try:
                self.emotion = self.emotion_engine.assess_event(event, self.emotion, self.name)
                self._note_success("observe")
            except Exception as e:
                logger.debug("Event emotion assessment failed: %s", e)
                self._note_failure("observe", e)
        try:
            self._update_relationship_event(event, user_id)
        except Exception as e:
            logger.debug("Event relationship assessment failed: %s", e)
            self._note_failure("observe", e)

    def _update_relationship_event(self, event: str, user_id: str) -> None:
        current = self.relationships.get_or_create(user_id)
        dims = current["dimensions"]
        messages = [
            {
                "role": "system",
                "content": (
                    "You assess how an EVENT affects a relationship between a character "
                    "and another party. Return a JSON object with float fields between "
                    "-0.15 and 0.15 (0.0 = no change): trust, affection, respect, "
                    "familiarity (0.0 to 0.15 only), tension. Be conservative."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Current: trust={dims.get('trust', 0):.2f}, "
                    f"affection={dims.get('affection', 0):.2f}\n"
                    f"Event involving {user_id}: {event[:300]}\n\nReturn JSON."
                ),
            },
        ]
        result = self.llm.generate_json_robust(messages)
        if not isinstance(result, dict):
            return
        deltas = {}
        for key in ("trust", "affection", "respect", "familiarity", "tension"):
            val = result.get(key, 0.0)
            if isinstance(val, (int, float)):
                deltas[key] = float(val)
        if deltas:
            self.relationships.update(user_id, deltas)
```

- [ ] **Step 5: Run tests, commit**

```bash
git add src/woven_imprint/character.py src/woven_imprint/persona/emotion.py tests/test_observe.py
git commit -m "feat: Character.observe() — world-event ingestion without dialogue-pair pollution (B4)"
```

---

### Task 6: Consolidation — chunked drain + dry_run fix

**Files:**
- Modify: `src/woven_imprint/memory/consolidation.py`
- Test: `tests/test_consolidation.py` (additions)

**Interfaces:**
- Produces: `consolidate(dry_run=False, chunk_size=None)` — `chunk_size` defaults to `get_config().maintenance.consolidate_chunk_size`; replaces the hardcoded `limit=500` pull. New `drain(max_chunks=10, dry_run=False) -> dict` — loops `consolidate()` until `needs_consolidation()` is False or `max_chunks` reached; returns aggregated `{"passes": int, "clusters": int, "summarized": int, "created": int, "archived": int}`. dry_run must NOT call the LLM (current bug: `_summarize_cluster` runs before the dry_run write-guard — restructure so dry_run counts clusters without summarizing).
- Consumes: `MaintenanceConfig.consolidate_chunk_size` (Task 2).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_consolidation.py (reuse its existing setup helpers/fixtures — read the file)
def test_dry_run_makes_no_llm_calls(consolidation_setup):  # adapt fixture name to file's style
    engine, char = consolidation_setup
    for i in range(30):
        char.memory.add(f"the lake was calm on day {i}", tier="buffer")
    before = char.llm.call_count
    result = char.consolidator.consolidate(dry_run=True)
    assert char.llm.call_count == before
    assert result["clusters"] >= 1
    # nothing archived under dry_run
    assert engine.storage.count_memories(char.id, tier="buffer") == 30 + <seeded_buffer_count>


def test_drain_processes_beyond_single_chunk(consolidation_setup):
    engine, char = consolidation_setup
    for i in range(60):
        char.memory.add(f"note {i} about the {'lake' if i % 2 else 'forest'}", tier="buffer")
    char.consolidator.threshold = 10
    result = char.consolidator.drain(max_chunks=10)
    assert result["passes"] >= 1
    assert not char.consolidator.needs_consolidation()
```

(Adapt the fixture/seeded-count details to the file's existing conventions — read `tests/test_consolidation.py` first; the assertions above are the contract.)

- [ ] **Step 2: Run to verify failure** → `drain` missing / dry_run LLM-call assertion fails.

- [ ] **Step 3: Implement**

In `consolidate()`: replace `limit=500` with `limit=chunk_size` where `chunk_size = chunk_size or get_config().maintenance.consolidate_chunk_size` (new parameter). Restructure the multi-member-cluster branch so that when `dry_run` is True the summary LLM call is skipped entirely (count the cluster in `clusters`, skip `summarized/created/archived` increments and all writes).

Add:

```python
    def drain(self, max_chunks: int = 10, dry_run: bool = False) -> dict:
        """Run consolidation passes until the buffer is below threshold.

        Replaces the old single-pass 500-row cap: a heavy day fully drains
        across multiple bounded passes.
        """
        totals = {"passes": 0, "clusters": 0, "summarized": 0, "created": 0, "archived": 0}
        for _ in range(max_chunks):
            if not self.needs_consolidation():
                break
            result = self.consolidate(dry_run=dry_run)
            totals["passes"] += 1
            for key in ("clusters", "summarized", "created", "archived"):
                totals[key] += result.get(key, 0)
            if dry_run:
                break  # dry_run archives nothing → would loop forever
        return totals
```

- [ ] **Step 4: Run tests, commit**

```bash
git add src/woven_imprint/memory/consolidation.py tests/test_consolidation.py
git commit -m "fix: chunked consolidation drain + dry_run no longer calls the LLM (B1)"
```

---

### Task 7: MaintenanceRunner core + jobs: consolidate, buffer_hygiene, score_importance

**Files:**
- Create: `src/woven_imprint/maintenance.py`
- Test: `tests/test_maintenance.py` (new)

**Interfaces:**
- Produces: `Budget` and `MaintenanceRunner` exactly as in "Locked interfaces". Job registry: `MaintenanceRunner._job_<name>(self) -> dict` methods; each returns a details dict; the runner wraps with status/llm_calls/duration. Jobs in this task: `consolidate` (calls `character.consolidator.drain()`; llm_calls = summarized count), `buffer_hygiene` (archives active buffer rows older than `buffer_ttl_days` with `importance <= buffer_hygiene_max_importance`; zero LLM), `score_importance` (batch-scores up to `importance_scoring_batch` recent buffer rows still at default importance 0.5; ONE LLM call).
- Consumes: storage helpers (Task 1), config (Task 2), `generate_json_robust` (Task 3), `drain` (Task 6).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_maintenance.py
import json
from datetime import datetime, timedelta, timezone

from tests.helpers import FakeEmbedder, FakeLLM, make_test_engine
from woven_imprint.engine import Engine
from woven_imprint.maintenance import Budget, MaintenanceRunner


def _age_memory(engine, memory_id, days):
    old = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    engine.storage._conn.execute(
        "UPDATE memories SET created_at = ? WHERE id = ?", (old, memory_id)
    )
    engine.storage._conn.commit()


def test_budget():
    b = Budget(2)
    assert b.take() and b.take() and not b.take()
    assert b.used == 2
    assert Budget(None).take(100)  # unlimited


def test_buffer_hygiene_archives_old_low_importance():
    engine = make_test_engine()
    char = engine.create_character("Tidy")
    fresh = char.memory.add("fresh chatter", tier="buffer")
    old_low = char.memory.add("old chatter", tier="buffer")
    old_important = char.memory.add("old but important", tier="buffer", importance=0.8)
    _age_memory(engine, old_low["id"], 30)
    _age_memory(engine, old_important["id"], 30)

    runner = MaintenanceRunner(char)
    report = runner.run(jobs=["buffer_hygiene"])
    assert report["jobs"]["buffer_hygiene"]["status"] == "ok"
    assert report["jobs"]["buffer_hygiene"]["archived"] == 1
    active_ids = {m["id"] for m in char.memory.get_all(tier="buffer")}
    assert fresh["id"] in active_ids and old_important["id"] in active_ids
    assert old_low["id"] not in active_ids


class ScoringLLM(FakeLLM):
    def generate_json(self, messages, temperature=0.3, **kw):
        system = messages[0].get("content", "")
        if "importance" in system.lower() or "score" in system.lower():
            user = messages[-1]["content"]
            n = user.count("\n1.") + sum(user.count(f"\n{i}.") for i in range(2, 40))
            return [8] * max(1, n)
        return super().generate_json(messages, temperature=temperature, **kw)

    def generate_json_robust(self, messages, temperature=0.3, **kw):
        return self.generate_json(messages, temperature=temperature, **kw)


def test_score_importance_updates_defaults_only():
    engine = Engine(db_path=":memory:", llm=ScoringLLM(), embedding=FakeEmbedder())
    char = engine.create_character("Scored")
    char.background = False
    default_mem = char.memory.add("something notable happened", tier="buffer")
    scored_mem = char.memory.add("already scored", tier="buffer", importance=0.9)

    runner = MaintenanceRunner(char)
    report = runner.run(jobs=["score_importance"])
    assert report["jobs"]["score_importance"]["scored"] >= 1
    assert engine.storage.get_memory(default_mem["id"])["importance"] == 0.8  # 8/10
    assert engine.storage.get_memory(scored_mem["id"])["importance"] == 0.9  # untouched


def test_budget_exhaustion_skips_llm_jobs():
    engine = make_test_engine()
    char = engine.create_character("Broke")
    char.memory.add("something", tier="buffer")
    runner = MaintenanceRunner(char, budget=Budget(0))
    report = runner.run(jobs=["score_importance"])
    assert report["jobs"]["score_importance"]["status"] == "skipped"
    assert report["llm_calls_used"] == 0


def test_unknown_job_fails_loudly():
    engine = make_test_engine()
    char = engine.create_character("Oops")
    runner = MaintenanceRunner(char)
    report = runner.run(jobs=["nonexistent"])
    assert report["jobs"]["nonexistent"]["status"] == "failed"
```

- [ ] **Step 2: Run to verify failure** → ModuleNotFoundError.

- [ ] **Step 3: Implement `maintenance.py`**

```python
"""Offline batch maintenance — the nightly-job primitive.

All heavy multi-pass LLM work (consolidation, reflection, growth, dedup,
reinforcement, contradiction sweeps, callback generation) runs here:
budgeted, idempotent, per-character, callable headless. This is where
small models get retries and where "runs overnight while charging" lives.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

from .log import logger


class Budget:
    """LLM-call budget for one maintenance run. limit=None → unlimited."""

    def __init__(self, limit: int | None):
        self.limit = limit
        self.used = 0

    def take(self, n: int = 1) -> bool:
        if self.limit is not None and self.used + n > self.limit:
            return False
        self.used += n
        return True

    @property
    def remaining(self) -> int | None:
        return None if self.limit is None else max(0, self.limit - self.used)


class MaintenanceRunner:
    DEFAULT_JOBS = [
        "consolidate",
        "buffer_hygiene",
        "score_importance",
        "dedup",
        "reinforce",
        "contradictions",
        "reflect",
        "evolve",
        "callbacks",
    ]

    def __init__(self, character, budget: Budget | None = None):
        from .config import get_config

        self.character = character
        self.cfg = get_config().maintenance
        self.budget = budget if budget is not None else Budget(self.cfg.max_llm_calls_per_run)

    def run(self, jobs: list[str] | None = None) -> dict:
        report: dict = {"character_id": self.character.id, "jobs": {}, "llm_calls_used": 0}
        for name in jobs or self.DEFAULT_JOBS:
            started = time.perf_counter()
            used_before = self.budget.used
            handler = getattr(self, f"_job_{name}", None)
            entry: dict
            if handler is None:
                entry = {"status": "failed", "error": f"unknown job: {name}"}
            else:
                try:
                    details = handler()
                    entry = {"status": details.pop("_status", "ok"), **details}
                except Exception as e:
                    logger.warning("Maintenance job '%s' failed: %s", name, e)
                    entry = {"status": "failed", "error": f"{type(e).__name__}: {e}"[:300]}
            entry["llm_calls"] = self.budget.used - used_before
            entry["duration_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
            report["jobs"][name] = entry
        report["llm_calls_used"] = self.budget.used
        return report

    # ── Jobs ──────────────────────────────────────────────────

    def _job_consolidate(self) -> dict:
        char = self.character
        if not char.consolidator.needs_consolidation():
            return {"_status": "skipped", "reason": "buffer below threshold"}
        if not self.budget.take(1):  # coarse: at least one summary call likely
            return {"_status": "skipped", "reason": "budget exhausted"}
        result = char.consolidator.drain()
        # account for actual summary calls beyond the coarse reservation
        extra = max(0, result.get("summarized", 0) - 1)
        self.budget.take(extra)
        return result

    def _job_buffer_hygiene(self) -> dict:
        char = self.character
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.cfg.buffer_ttl_days)
        cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")
        candidates = char.storage.get_memories(char.id, tier="buffer", limit=1000)
        stale = [
            m["id"]
            for m in candidates
            if (m.get("created_at") or "") < cutoff_str
            and m.get("importance", 0.5) <= self.cfg.buffer_hygiene_max_importance
        ]
        char.storage.archive_memories_batch(stale)
        return {"archived": len(stale)}

    def _job_score_importance(self) -> dict:
        char = self.character
        candidates = [
            m
            for m in char.storage.get_memories(char.id, tier="buffer", limit=500)
            if m.get("importance") == 0.5
        ][: self.cfg.importance_scoring_batch]
        if not candidates:
            return {"_status": "skipped", "reason": "nothing to score"}
        if not self.budget.take(1):
            return {"_status": "skipped", "reason": "budget exhausted"}
        numbered = "\n".join(f"{i + 1}. {m['content'][:200]}" for i, m in enumerate(candidates))
        messages = [
            {
                "role": "system",
                "content": (
                    "You score the long-term importance of memories on a 1-10 scale "
                    "(1 = mundane small talk, 10 = life-changing). "
                    "Return a JSON array of integers, one per numbered memory, in order."
                ),
            },
            {"role": "user", "content": f"Score these memories:\n{numbered}"},
        ]
        scores = char.llm.generate_json_robust(messages)
        if not isinstance(scores, list):
            return {"_status": "failed", "error": "non-list score response"}
        scored = 0
        for m, s in zip(candidates, scores):
            if isinstance(s, (int, float)):
                char.storage.set_memory_importance(m["id"], max(0.1, min(0.9, float(s) / 10)))
                scored += 1
        return {"scored": scored, "candidates": len(candidates)}
```

- [ ] **Step 4: Run tests (test file covers Budget, hygiene, scoring, budget-skip, unknown job), commit**

```bash
git add src/woven_imprint/maintenance.py tests/test_maintenance.py
git commit -m "feat: MaintenanceRunner with budget + consolidate/buffer_hygiene/score_importance jobs (B1, B5)"
```

---

### Task 8: Jobs — dedup, reinforce, contradictions

**Files:**
- Modify: `src/woven_imprint/maintenance.py`
- Test: `tests/test_maintenance.py` (additions)

**Interfaces:**
- Produces: `_job_dedup` (embedding-similarity merge of near-duplicate active core observations: keep higher-importance one, `reinforce()` it, archive the other; NO LLM), `_job_reinforce` (recent buffer entries semantically matching core facts ≥ `reinforce_similarity` → `belief.reinforce(core_id)`, once per core per run; NO LLM), `_job_contradictions` (candidate core pairs with similarity in `[contradiction_candidate_similarity, dedup_similarity)` → LLM verdict per pair, budgeted, max `contradiction_max_pairs`; on contradiction the superseded memory gets `update_memory_status(id, "contradicted", certainty=0.0)`).
- Consumes: `BeliefReviser.reinforce(memory_id)`, `storage.update_memory_status`, `_cosine_similarity` (import from `memory.retrieval`), config thresholds (Task 2).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_maintenance.py

def test_dedup_archives_near_duplicates_and_reinforces_kept():
    engine = make_test_engine()
    char = engine.create_character("Dedup")
    a = char.memory.add("the user's sister anna loves rowing", tier="core",
                        role="observation", importance=0.8)
    b = char.memory.add("the user's sister anna loves rowing", tier="core",
                        role="observation", importance=0.7)
    unrelated = char.memory.add("the moon rises over the lake", tier="core",
                                role="observation", importance=0.7)
    runner = MaintenanceRunner(char)
    report = runner.run(jobs=["dedup"])
    assert report["jobs"]["dedup"]["archived"] == 1
    active = {m["id"] for m in char.memory.get_all(tier="core")}
    assert a["id"] in active and unrelated["id"] in active and b["id"] not in active
    kept = engine.storage.get_memory(a["id"])
    assert kept["certainty"] > 1.0 - 1e-9 or kept["certainty"] == 1.0  # reinforced (clamped at 1.0)


def test_dedup_idempotent():
    engine = make_test_engine()
    char = engine.create_character("Dedup2")
    char.memory.add("fact one about cats", tier="core", role="observation")
    char.memory.add("fact one about cats", tier="core", role="observation")
    runner = MaintenanceRunner(char)
    first = runner.run(jobs=["dedup"])["jobs"]["dedup"]["archived"]
    second = runner.run(jobs=["dedup"])["jobs"]["dedup"]["archived"]
    assert first == 1 and second == 0


def test_reinforce_strengthens_reencountered_fact():
    engine = make_test_engine()
    char = engine.create_character("Rein")
    core = char.memory.add("keeper works as a software developer", tier="core",
                           role="observation")
    engine.storage.update_memory_certainty(core["id"], -0.4)  # certainty 0.6
    char.memory.add("keeper works as a software developer", tier="buffer", role="user")
    runner = MaintenanceRunner(char)
    report = runner.run(jobs=["reinforce"])
    assert report["jobs"]["reinforce"]["reinforced"] == 1
    assert engine.storage.get_memory(core["id"])["certainty"] > 0.6


class ContradictionLLM(FakeLLM):
    def generate_json(self, messages, temperature=0.3, **kw):
        system = messages[0].get("content", "")
        if "contradict" in system.lower():
            return {"contradictory": True, "current": "second"}
        return super().generate_json(messages, temperature=temperature, **kw)

    def generate_json_robust(self, messages, temperature=0.3, **kw):
        return self.generate_json(messages, temperature=temperature, **kw)


def test_contradiction_sweep_marks_superseded(monkeypatch):
    engine = Engine(db_path=":memory:", llm=ContradictionLLM(), embedding=FakeEmbedder())
    char = engine.create_character("Contra")
    char.background = False
    old = char.memory.add("keeper lives in the city near the harbor", tier="core",
                          role="observation")
    new = char.memory.add("keeper lives in the country near the lake", tier="core",
                          role="observation")
    # FakeEmbedder bag-of-words: these share enough tokens to be candidates
    runner = MaintenanceRunner(char)
    report = runner.run(jobs=["contradictions"])
    assert report["jobs"]["contradictions"]["contradicted"] == 1
    assert engine.storage.get_memory(old["id"])["status"] == "contradicted"
    assert engine.storage.get_memory(new["id"])["status"] == "active"
```

(If the FakeEmbedder similarity for the contradiction pair falls outside `[0.70, 0.92)`, tune the two content strings — verify actual cosine in a quick REPL step — rather than changing thresholds.)

- [ ] **Step 2: Run to verify failure** → jobs report `unknown job`.

- [ ] **Step 3: Implement the three jobs** (in `maintenance.py`; import `from .memory.retrieval import _cosine_similarity`):

```python
    def _active_core_observations(self, limit: int) -> list[dict]:
        return [
            m
            for m in self.character.storage.get_memories(
                self.character.id, tier="core", limit=limit
            )
            if m.get("role") == "observation" and m.get("embedding")
        ]

    def _job_dedup(self) -> dict:
        mems = self._active_core_observations(self.cfg.dedup_scan_limit)
        archived: list[str] = []
        archived_set: set[str] = set()
        for i in range(len(mems)):
            if mems[i]["id"] in archived_set:
                continue
            for j in range(i + 1, len(mems)):
                if mems[j]["id"] in archived_set:
                    continue
                sim = _cosine_similarity(mems[i]["embedding"], mems[j]["embedding"])
                if sim >= self.cfg.dedup_similarity:
                    keep, drop = (
                        (mems[i], mems[j])
                        if mems[i].get("importance", 0) >= mems[j].get("importance", 0)
                        else (mems[j], mems[i])
                    )
                    archived.append(drop["id"])
                    archived_set.add(drop["id"])
                    self.character.belief.reinforce(keep["id"])
        self.character.storage.archive_memories_batch(archived)
        return {"archived": len(archived), "scanned": len(mems)}

    def _job_reinforce(self) -> dict:
        char = self.character
        cores = self._active_core_observations(self.cfg.dedup_scan_limit)
        buffers = [
            m
            for m in char.storage.get_memories(char.id, tier="buffer", limit=200)
            if m.get("embedding")
        ]
        reinforced: set[str] = set()
        for b in buffers:
            for c in cores:
                if c["id"] in reinforced:
                    continue
                if _cosine_similarity(b["embedding"], c["embedding"]) >= self.cfg.reinforce_similarity:
                    char.belief.reinforce(c["id"])
                    reinforced.add(c["id"])
        return {"reinforced": len(reinforced), "buffer_scanned": len(buffers)}

    def _job_contradictions(self) -> dict:
        char = self.character
        mems = self._active_core_observations(self.cfg.dedup_scan_limit)
        pairs = []
        for i in range(len(mems)):
            for j in range(i + 1, len(mems)):
                sim = _cosine_similarity(mems[i]["embedding"], mems[j]["embedding"])
                if self.cfg.contradiction_candidate_similarity <= sim < self.cfg.dedup_similarity:
                    pairs.append((sim, mems[i], mems[j]))
        pairs.sort(key=lambda p: p[0], reverse=True)
        contradicted = 0
        checked = 0
        for _sim, a, b in pairs[: self.cfg.contradiction_max_pairs]:
            if not self.budget.take(1):
                break
            checked += 1
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You check whether two remembered facts contradict each other. "
                        'Return JSON: {"contradictory": true|false, '
                        '"current": "first"|"second"|"unclear"} — "current" is the fact '
                        "that reflects the present state if they contradict."
                    ),
                },
                {
                    "role": "user",
                    "content": f"First: {a['content'][:300]}\nSecond: {b['content'][:300]}",
                },
            ]
            try:
                verdict = char.llm.generate_json_robust(messages)
            except ValueError:
                continue
            if not isinstance(verdict, dict) or not verdict.get("contradictory"):
                continue
            current = verdict.get("current")
            if current == "first":
                superseded = b
            elif current == "second":
                superseded = a
            else:
                continue  # unclear — leave both, low certainty is belief revision's job
            char.storage.update_memory_status(superseded["id"], "contradicted", certainty=0.0)
            contradicted += 1
        return {"contradicted": contradicted, "pairs_checked": checked,
                "candidates": len(pairs)}
```

- [ ] **Step 4: Run tests (verify similarity assumptions with the FakeEmbedder empirically; tune test strings if needed), commit**

```bash
git add src/woven_imprint/maintenance.py tests/test_maintenance.py
git commit -m "feat: dedup/reinforce/contradiction maintenance sweeps (B1)"
```

---

### Task 9: Jobs — reflect + evolve; `Engine.run_maintenance`; CLI `maintain`

**Files:**
- Modify: `src/woven_imprint/maintenance.py`, `src/woven_imprint/engine.py`, `src/woven_imprint/cli.py`
- Test: `tests/test_maintenance.py` (additions), `tests/test_engine.py` (additions)

**Interfaces:**
- Produces: `_job_reflect` (Stanford-style trigger: sum of active-buffer importance created after the last `[Reflection]` core memory ≥ `reflect_importance_sum` → `character.reflect()`, 1 budgeted LLM call), `_job_evolve` (delegates `character.evolve()` with config thresholds when core count ≥ `persona.growth_min_memories`; 1 budgeted call), `Engine.run_maintenance(character_id=None, jobs=None, budget=None) -> list[dict]` (all characters when id is None; budget int → one shared `Budget` across characters), CLI `woven-imprint maintain [--db PATH] [--character ID] [--jobs a,b,c] [--budget N] [--json]`.
- Consumes: everything above. Note `_job_callbacks` doesn't exist until Task 10 — the runner reports it `failed: unknown job` if requested early; `DEFAULT_JOBS` therefore must NOT include `"callbacks"` until Task 10 adds the job (ship this task with `DEFAULT_JOBS` ending at `"evolve"`, Task 10 appends `"callbacks"`).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_maintenance.py

def test_reflect_triggers_on_importance_sum():
    engine = make_test_engine()
    char = engine.create_character("Ref")
    for i in range(20):
        char.memory.add(f"noteworthy event {i}", tier="buffer", importance=0.7)  # sum 14 > 12
    report = MaintenanceRunner(char).run(jobs=["reflect"])
    assert report["jobs"]["reflect"]["status"] == "ok"
    reflections = [m for m in char.memory.get_all(tier="core")
                   if m["content"].startswith("[Reflection]")]
    assert len(reflections) == 1
    # second run: importance since last reflection is now ~0 → skipped
    report2 = MaintenanceRunner(char).run(jobs=["reflect"])
    assert report2["jobs"]["reflect"]["status"] == "skipped"


def test_evolve_skips_below_min_memories():
    engine = make_test_engine()
    char = engine.create_character("Evo")
    report = MaintenanceRunner(char).run(jobs=["evolve"])
    assert report["jobs"]["evolve"]["status"] == "skipped"


# append to tests/test_engine.py
def test_run_maintenance_all_characters():
    from tests.helpers import make_test_engine

    engine = make_test_engine()
    engine.create_character("A")
    engine.create_character("B")
    reports = engine.run_maintenance(jobs=["buffer_hygiene"])
    assert len(reports) == 2
    assert all(r["jobs"]["buffer_hygiene"]["status"] == "ok" for r in reports)
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement the jobs**

```python
    def _job_reflect(self) -> dict:
        char = self.character
        reflections = [
            m
            for m in char.storage.get_memories(char.id, tier="core", limit=1000)
            if m["content"].startswith("[Reflection]")
        ]
        last_ts = max((m.get("created_at") or "" for m in reflections), default="")
        buffer = char.storage.get_memories(char.id, tier="buffer", limit=1000)
        pending = sum(
            m.get("importance", 0.5)
            for m in buffer
            if (m.get("created_at") or "") > last_ts
        )
        if pending < self.cfg.reflect_importance_sum:
            return {"_status": "skipped", "reason": f"importance sum {pending:.1f} below threshold"}
        if not self.budget.take(1):
            return {"_status": "skipped", "reason": "budget exhausted"}
        char.reflect()
        return {"triggered_at_sum": round(pending, 1)}

    def _job_evolve(self) -> dict:
        from .config import get_config

        persona_cfg = get_config().persona
        char = self.character
        core_count = char.storage.count_memories(char.id, tier="core")
        if core_count < persona_cfg.growth_min_memories:
            return {"_status": "skipped", "reason": f"{core_count} core memories < min"}
        if not self.budget.take(1):
            return {"_status": "skipped", "reason": "budget exhausted"}
        events = char.evolve(
            min_memories=persona_cfg.growth_min_memories,
            threshold=persona_cfg.growth_threshold,
        )
        return {"growth_events": len(events)}
```

- [ ] **Step 4: `Engine.run_maintenance`** (in `engine.py`):

```python
    def run_maintenance(
        self,
        character_id: str | None = None,
        jobs: list[str] | None = None,
        budget: int | None = None,
    ) -> list[dict]:
        """Run offline maintenance jobs (the nightly-batch primitive).

        Args:
            character_id: One character, or None for all.
            jobs: Subset of MaintenanceRunner.DEFAULT_JOBS, or None for all.
            budget: Max LLM calls for the whole run (shared across characters);
                    None uses config maintenance.max_llm_calls_per_run.
        """
        from .config import get_config
        from .maintenance import Budget, MaintenanceRunner

        limit = budget if budget is not None else get_config().maintenance.max_llm_calls_per_run
        shared = Budget(limit)
        ids = [character_id] if character_id else [c["id"] for c in self.list_characters()]
        reports = []
        for cid in ids:
            char = self.load_character(cid)
            if char is None:
                continue
            reports.append(MaintenanceRunner(char, budget=shared).run(jobs=jobs))
            char.close()
        return reports
```

- [ ] **Step 5: CLI subcommand** — in `cli.py` add parser (mirror the existing subcommand style — read `main()` first):

```python
    p_maintain = sub.add_parser("maintain", help="Run offline maintenance jobs (nightly batch)")
    p_maintain.add_argument("--db", default=None, help="Database path")
    p_maintain.add_argument("--character", default=None, help="Character ID (default: all)")
    p_maintain.add_argument("--jobs", default=None, help="Comma-separated job list")
    p_maintain.add_argument("--budget", type=int, default=None, help="Max LLM calls")
    p_maintain.add_argument("--json", action="store_true", help="Machine-readable output")
```

with `cmd_maintain(args)` that builds the engine the same way other commands do (`_get_engine(args)` if that helper takes args — read it), parses `--jobs` via `args.jobs.split(",")`, calls `engine.run_maintenance(...)`, and prints either `json.dumps(reports, indent=2)` or a compact per-job table. Wire into the command dispatch.

- [ ] **Step 6: Run everything, commit**

```bash
git add src/woven_imprint/maintenance.py src/woven_imprint/engine.py src/woven_imprint/cli.py tests/test_maintenance.py tests/test_engine.py
git commit -m "feat: reflect/evolve jobs, Engine.run_maintenance, woven-imprint maintain CLI (B1)"
```

---

### Task 10: CallbackEngine — generation + read path

**Files:**
- Create: `src/woven_imprint/callbacks.py`
- Modify: `src/woven_imprint/maintenance.py` (add `_job_callbacks`, append `"callbacks"` to `DEFAULT_JOBS`)
- Modify: `src/woven_imprint/character.py` (`get_callbacks`, `refresh_callbacks`, end_session hook)
- Test: `tests/test_callbacks.py` (new)

**Interfaces:**
- Produces: `CallbackEngine` per "Locked interfaces". `refresh()` gathers sources (recent high-importance core facts importance ≥ 0.7 limit 15; relationship key_moments — plain strings — last 5 per relationship; arc beats last 5), makes ONE LLM call producing a JSON array `[{"hook": str, "kind": "open_thread"|"callback"|"milestone"|"curiosity", "sources": [int]}]`, maps source indices → memory ids where the source was a memory, saves via `storage.save_callback` with `salience` = mean source importance (0.5 for non-memory sources), then enforces `callbacks_ready_cap` (oldest ready beyond cap → `expired`). PROMPT MUST ENFORCE PARAPHRASE (see code). `get(limit)` reads ready callbacks and adds `"freshness"` (e.g. `"2d"`, `"3h"`). `Character.get_callbacks(limit=3)` and `Character.refresh_callbacks(budget=None)` delegate; `end_session()` calls `refresh_callbacks` when `maintenance.callbacks_refresh_on_session_end` (guarded try/except + health note under `"callbacks"`).
- Consumes: storage callback CRUD (Task 1), config (Task 2), `generate_json_robust` (Task 3), Budget (Task 7).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_callbacks.py
from tests.helpers import FakeEmbedder, FakeLLM, make_test_engine
from woven_imprint.engine import Engine


class CallbackLLM(FakeLLM):
    def generate_json(self, messages, temperature=0.3, **kw):
        system = messages[0].get("content", "")
        if "conversation hooks" in system.lower():
            return [
                {"hook": "You mentioned an interview — how did it go?",
                 "kind": "open_thread", "sources": [1]},
                {"hook": "Wonder how Anna's rowing season is going.",
                 "kind": "curiosity", "sources": [2]},
            ]
        return super().generate_json(messages, temperature=temperature, **kw)

    def generate_json_robust(self, messages, temperature=0.3, **kw):
        return self.generate_json(messages, temperature=temperature, **kw)


def _engine_with_callback_llm():
    engine = Engine(db_path=":memory:", llm=CallbackLLM(), embedding=FakeEmbedder())
    orig = engine.create_character

    def _mk(*a, **kw):
        c = orig(*a, **kw)
        c.background = False
        c.parallel = False
        return c

    engine.create_character = _mk
    return engine


def test_refresh_creates_paraphrased_callbacks():
    engine = _engine_with_callback_llm()
    char = engine.create_character("Hooky")
    char.memory.add("keeper has a job interview on friday", tier="core",
                    role="observation", importance=0.8)
    char.memory.add("keeper's sister anna loves rowing", tier="core",
                    role="observation", importance=0.75)
    created = char.refresh_callbacks()
    assert created == 2
    cbs = char.get_callbacks(limit=5)
    assert len(cbs) == 2
    assert cbs[0]["kind"] in ("open_thread", "callback", "milestone", "curiosity")
    assert "freshness" in cbs[0]
    assert cbs[0]["source_memory_ids"]  # mapped from source indices


def test_get_callbacks_is_instant_db_read():
    engine = _engine_with_callback_llm()
    char = engine.create_character("Quick")
    char.memory.add("something important happened", tier="core",
                    role="observation", importance=0.9)
    char.refresh_callbacks()
    before = char.llm.call_count
    char.get_callbacks()
    assert char.llm.call_count == before  # zero LLM on read


def test_ready_cap_expires_oldest():
    engine = _engine_with_callback_llm()
    char = engine.create_character("Capped")
    for i in range(6):
        engine.storage.save_callback(
            {"id": f"old-{i}", "character_id": char.id, "kind": "curiosity",
             "hook": f"old hook {i}", "salience": 0.1}
        )
    from woven_imprint.config import get_config
    get_config().maintenance.callbacks_ready_cap = 5
    try:
        char.memory.add("fresh important fact", tier="core", role="observation",
                        importance=0.9)
        char.refresh_callbacks()
        ready = engine.storage.get_callbacks(char.id, status="ready", limit=50)
        assert len(ready) <= 5
    finally:
        from woven_imprint.config import reload_config
        reload_config()


def test_maintenance_job_callbacks_registered():
    from woven_imprint.maintenance import MaintenanceRunner

    assert "callbacks" in MaintenanceRunner.DEFAULT_JOBS
    engine = _engine_with_callback_llm()
    char = engine.create_character("Job")
    char.memory.add("keeper started learning the violin", tier="core",
                    role="observation", importance=0.8)
    report = MaintenanceRunner(char).run(jobs=["callbacks"])
    assert report["jobs"]["callbacks"]["status"] == "ok"
    assert report["jobs"]["callbacks"]["created"] >= 1
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement `callbacks.py`**

```python
"""Callback surfacing — the research-backed #1 companion feature.

Callbacks are generated in BATCH (maintenance job / session end) and read
INSTANTLY from the DB. Hard rule: hooks are paraphrased, in-character
references — never verbatim quotes of stored memories (verbatim reads as
creepy; paraphrase captures the benefit).
"""

from __future__ import annotations

from datetime import datetime, timezone

from .log import logger
from .utils.text import generate_id

VALID_KINDS = ("open_thread", "callback", "milestone", "curiosity")


def _freshness(created_at: str) -> str:
    try:
        created = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - created
        if delta.days >= 1:
            return f"{delta.days}d"
        hours = delta.seconds // 3600
        return f"{hours}h" if hours else f"{delta.seconds // 60}m"
    except (ValueError, AttributeError):
        return "?"


class CallbackEngine:
    def __init__(self, character):
        from .config import get_config

        self.character = character
        self.cfg = get_config().maintenance

    # ── Read path (instant) ───────────────────────────────────

    def get(self, limit: int = 3) -> list[dict]:
        rows = self.character.storage.get_callbacks(
            self.character.id, status="ready", limit=limit
        )
        for r in rows:
            r["freshness"] = _freshness(r.get("created_at", ""))
        return rows

    # ── Generation (batch) ────────────────────────────────────

    def _gather_sources(self) -> list[dict]:
        char = self.character
        sources: list[dict] = []
        for m in char.storage.get_memories(char.id, tier="core", limit=200):
            if m.get("role") == "observation" and m.get("importance", 0) >= 0.7:
                sources.append(
                    {"text": m["content"][:200], "memory_id": m["id"],
                     "importance": m.get("importance", 0.5)}
                )
        sources = sources[:15]
        for rel in char.relationships.get_all():
            for moment in (rel.get("key_moments") or [])[-5:]:
                sources.append({"text": str(moment)[:200], "memory_id": None,
                                "importance": 0.5})
        for beat in char.arc.beats[-5:]:
            sources.append({"text": beat.description[:200], "memory_id": None,
                            "importance": 0.5})
        return sources

    def refresh(self, budget=None, limit: int | None = None) -> int:
        char = self.character
        limit = limit or self.cfg.callbacks_refresh_limit
        sources = self._gather_sources()
        if not sources:
            return 0
        if budget is not None and not budget.take(1):
            return 0

        numbered = "\n".join(f"{i + 1}. {s['text']}" for i, s in enumerate(sources))
        messages = [
            {
                "role": "system",
                "content": (
                    f"You create conversation hooks for {char.name} — things the "
                    "character would naturally bring up with the person they know. "
                    f"Return a JSON array (max {limit}) of objects: "
                    '{"hook": <one natural in-character sentence>, '
                    '"kind": "open_thread"|"callback"|"milestone"|"curiosity", '
                    '"sources": [<numbers of the memories it draws on>]}. '
                    "RULES: paraphrase naturally — NEVER quote the memory text "
                    "verbatim. open_thread = unresolved thing to ask about; "
                    "callback = warm reference to a shared moment; milestone = "
                    "anniversary or achievement; curiosity = something the "
                    "character genuinely wonders about."
                ),
            },
            {"role": "user", "content": f"Memories and moments:\n{numbered}"},
        ]
        try:
            result = char.llm.generate_json_robust(messages)
        except ValueError as e:
            logger.debug("Callback generation failed: %s", e)
            return 0
        if not isinstance(result, list):
            return 0

        created = 0
        for item in result[:limit]:
            if not isinstance(item, dict) or not item.get("hook"):
                continue
            kind = item.get("kind", "callback")
            if kind not in VALID_KINDS:
                kind = "callback"
            idxs = [i for i in item.get("sources", []) if isinstance(i, int)]
            mem_ids = [
                sources[i - 1]["memory_id"]
                for i in idxs
                if 0 < i <= len(sources) and sources[i - 1]["memory_id"]
            ]
            sals = [sources[i - 1]["importance"] for i in idxs if 0 < i <= len(sources)]
            char.storage.save_callback(
                {
                    "id": generate_id("cb-"),
                    "character_id": char.id,
                    "kind": kind,
                    "hook": str(item["hook"])[:400],
                    "source_memory_ids": mem_ids,
                    "salience": round(sum(sals) / len(sals), 3) if sals else 0.5,
                }
            )
            created += 1

        # Enforce ready-queue cap: expire oldest beyond cap
        ready = char.storage.get_callbacks(char.id, status="ready", limit=1000)
        if len(ready) > self.cfg.callbacks_ready_cap:
            # get_callbacks orders salience DESC, created_at DESC → expire the tail
            for stale in ready[self.cfg.callbacks_ready_cap:]:
                char.storage.mark_callback(stale["id"], "expired")
        return created
```

- [ ] **Step 4: Wire into Character and maintenance**

`character.py`:

```python
    def get_callbacks(self, limit: int = 3) -> list[dict]:
        """Ready-to-use paraphrased conversation hooks (instant DB read)."""
        from .callbacks import CallbackEngine

        return CallbackEngine(self).get(limit=limit)

    def refresh_callbacks(self, budget=None) -> int:
        """Regenerate the callback queue (one LLM call — batch/offline use)."""
        from .callbacks import CallbackEngine

        return CallbackEngine(self).refresh(budget=budget)
```

In `end_session()`, after the summary is stored and before the auto-consolidation block:

```python
        from .config import get_config as _gc

        if _gc().maintenance.callbacks_refresh_on_session_end:
            try:
                self.refresh_callbacks()
                self._note_success("callbacks")
            except Exception as e:
                logger.debug("Session-end callback refresh failed: %s", e)
                self._note_failure("callbacks", e)
```

`maintenance.py`: append `"callbacks"` to `DEFAULT_JOBS` and add:

```python
    def _job_callbacks(self) -> dict:
        created = self.character.refresh_callbacks(budget=self.budget)
        if created == 0 and (self.budget.remaining == 0):
            return {"_status": "skipped", "reason": "budget exhausted", "created": 0}
        return {"created": created}
```

- [ ] **Step 5: Run tests (existing end_session tests may now make one extra LLM call — check `test_character_integration.py`/`test_services.py` fallout; fix expectations toward the new behavior or disable via config in those tests), commit**

```bash
git add src/woven_imprint/callbacks.py src/woven_imprint/maintenance.py src/woven_imprint/character.py tests/test_callbacks.py
git commit -m "feat: CallbackEngine — batch-generated paraphrased hooks, instant reads (B2)"
```

---

### Task 11: `compose_initiation` — proactive generation

**Files:**
- Modify: `src/woven_imprint/callbacks.py`, `src/woven_imprint/character.py`
- Test: `tests/test_callbacks.py` (additions)

**Interfaces:**
- Produces: `CallbackEngine.compose_initiation(occasion="greeting", budget=None) -> dict` returning `{"text": str, "callback": dict | None}`; consumes (marks `consumed`) the top ready callback when one exists; composes in character voice using the persona system prompt + emotion + top callback hook. Without any ready callback it still composes an occasion-appropriate opener (`callback: None`). `Character.compose_initiation(occasion="greeting")` delegates. Scheduling stays the app's job — the library only composes.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_callbacks.py

def test_compose_initiation_consumes_top_callback():
    engine = _engine_with_callback_llm()
    char = engine.create_character("Init")
    engine.storage.save_callback(
        {"id": "cb-top", "character_id": char.id, "kind": "open_thread",
         "hook": "Ask how the interview went", "salience": 0.9}
    )
    result = char.compose_initiation("morning greeting")
    assert isinstance(result["text"], str) and result["text"]
    assert result["callback"]["id"] == "cb-top"
    # consumed: no longer in ready queue
    assert engine.storage.get_callbacks(char.id, status="ready") == []
    assert engine.storage.get_callbacks(char.id, status="consumed")[0]["id"] == "cb-top"


def test_compose_initiation_without_callbacks():
    engine = _engine_with_callback_llm()
    char = engine.create_character("Lonely")
    result = char.compose_initiation("greeting")
    assert result["text"]
    assert result["callback"] is None
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement** (in `CallbackEngine`):

```python
    def compose_initiation(self, occasion: str = "greeting", budget=None) -> dict:
        """Compose a character-initiated message. Consumes the top callback.

        Scheduling WHEN to send is the app's responsibility; the library's
        job is having something in-character to say (scarcity by design —
        each callback is consumed on use).
        """
        char = self.character
        top = self.get(limit=1)
        callback = top[0] if top else None
        if budget is not None and not budget.take(1):
            return {"text": "", "callback": None}

        system = char.persona.build_system_prompt()
        emotion_desc = char.emotion.describe()
        if emotion_desc:
            system += f"\n\n{emotion_desc}"
        instruction = (
            f"Compose a short (1-2 sentence) message where you, {char.name}, "
            f"initiate contact. Occasion: {occasion}."
        )
        if callback:
            instruction += (
                f"\nNaturally work in this thought of yours (paraphrase, don't "
                f"quote): {callback['hook']}"
            )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": instruction},
        ]
        text = char.llm.generate(messages, temperature=0.8, max_tokens=150)
        if callback:
            char.storage.mark_callback(callback["id"], "consumed")
        return {"text": text.strip(), "callback": callback}
```

`character.py`:

```python
    def compose_initiation(self, occasion: str = "greeting") -> dict:
        """Character-initiated message (proactive). See CallbackEngine."""
        from .callbacks import CallbackEngine

        return CallbackEngine(self).compose_initiation(occasion=occasion)
```

- [ ] **Step 4: Run tests, commit**

```bash
git add src/woven_imprint/callbacks.py src/woven_imprint/character.py tests/test_callbacks.py
git commit -m "feat: compose_initiation — proactive character-initiated messages (B3)"
```

---

### Task 12: Server + MCP exposure

**Files:**
- Modify: `src/woven_imprint/server/services.py`, `src/woven_imprint/server/sidecar.py`, `src/woven_imprint/server/demo.py`, `src/woven_imprint/mcp_server.py`
- Test: `tests/test_services.py`, `tests/test_sidecar.py`, `tests/test_demo_server.py` (additions)

**Interfaces:**
- Produces services: `get_callbacks_service(engine, character_id, limit=3)`, `observe_service(engine, character_id, event, source="world", importance=None, user_id=None)`, `health_service(engine, character_id)`, `maintain_service(engine, character_id, jobs=None, budget=None)`. Sidecar routes: `GET /characters/{id}/callbacks?limit=`, `GET /characters/{id}/health`, `POST /observe` (JSON body `{character_id, event, source?, importance?, user_id?}`). Demo routes: `GET /api/characters/{id}/callbacks`, `GET /api/characters/{id}/health`, `POST /api/observe`, `POST /api/characters/{id}/maintain` (mutation-locked). MCP tools: `get_callbacks(character_id, limit=3)`, `observe(character_id, event, source="world", user_id="")`, `get_health(character_id)`, `maintain(character_id, jobs="")`.
- IMPORTANT for demo/MCP: use the CACHED character instance where a cache exists (demo `_char_cache`, MCP `_char_cache`) so health counters and background workers are the live ones — follow each file's existing `_get_character` pattern exactly.

- [ ] **Step 1: Write the failing tests** (follow each test file's existing fixture/client conventions — read them first; the contracts):

```python
# tests/test_services.py additions
def test_observe_and_callbacks_services(service_engine):  # adapt fixture name
    engine = service_engine
    char_info = create_character_service(engine, "Svc", {}, None)
    cid = char_info["id"]
    observe_service(engine, cid, "Keeper fed you", source="sim")
    result = get_callbacks_service(engine, cid, limit=3)
    assert isinstance(result, list)
    h = health_service(engine, cid)
    assert "subsystems" in h

# tests/test_sidecar.py additions — follow its HTTP-request helper style
def test_sidecar_observe_and_health(sidecar_client):
    # POST /observe returns 200 with the stored memory id
    # GET /characters/{id}/health returns subsystems dict
    ...
```

Write the sidecar/demo tests concretely against each file's existing test helpers (they exist for every current route — mirror the closest analog: `/record` for POST routes, `/relationships` for GET routes). Do not leave `...` stubs in the actual test code you write.

- [ ] **Step 2–3: Implement services then routes**, mirroring the existing patterns exactly (sidecar handler → service; demo endpoint → service or cached char; MCP tool → `_get_character`). `observe_service` must call `char.observe(...)` and return `{"memory_id": mem["id"], "content": mem["content"]}`. `health_service` returns `char.health()`. `maintain_service` calls `engine.run_maintenance(character_id=..., jobs=jobs, budget=budget)` and returns the single report. Demo's maintain endpoint acquires `_character_mutation(character_id)` like other mutating routes.

- [ ] **Step 4: Run full suite, commit**

```bash
git add src/woven_imprint/server/ src/woven_imprint/mcp_server.py tests/test_services.py tests/test_sidecar.py tests/test_demo_server.py
git commit -m "feat: expose callbacks/health/observe/maintain over sidecar, demo API, MCP (B2/B4/B5)"
```

---

### Task 13: SCHEMA.md + docs + changelog + CI close-out

**Files:**
- Create: `docs/SCHEMA.md`
- Modify: `CHANGELOG.md`, `docs/CONFIGURATION.md`, `docs/DEVELOPER_GUIDE.md`, `README.md` (feature bullets), `docs/ARCHITECTURE.md` (maintenance + callbacks sections)
- Test: none (docs) — full CI-equivalent run

**Interfaces:** none — close-out.

- [ ] **Step 1: Write `docs/SCHEMA.md`** — the portable contract (B6). Must document, from the actual code (verify each claim against `storage/sqlite.py` as it now stands): every table + column with types and CHECK constraints (`characters`, `memories`, `relationships`, `sessions`, `session_turns`, `callbacks`, `meta`, `schema_version`, `memories_fts`); the embedding BLOB format (little-endian float32, `struct.pack(f"{n}f")`, dimension recorded in `meta.embedding_dimensions`, model in `meta.embedding_model` — writers MUST verify dimensions before writing); timestamp format (`datetime('now')` → `YYYY-MM-DD HH:MM:SS` UTC, writers must match); the FTS5 dependency (external-content table + 3 sync triggers) and the sanctioned fallback for runtimes without FTS5 (keyword strategy degrades to `LIKE '%term%'` scanning — retrieval still works via the other RRF strategies); migration protocol (`schema_version` table, `_MIGRATIONS` dict, writers must not write to DBs with a newer version than they know); `meta.schema_semver` set to `0.6.0-dev` (add `meta_set("schema_semver", "0.6.0-dev")` to storage `_init_schema` in this task — one line + test assert in test_storage.py).

- [ ] **Step 2: Update the other docs** — CONFIGURATION.md: full `maintenance` section + `llm.embedding_base_url` + env vars `WOVEN_IMPRINT_MAINTENANCE_BUDGET`/`WOVEN_IMPRINT_EMBEDDING_BASE_URL`. DEVELOPER_GUIDE.md: sections "Nightly maintenance" (runner, jobs table, budget, CLI, WorkManager-style scheduling note), "Callbacks & proactive initiation" (refresh/get/compose_initiation lifecycle, paraphrase rule, consumed-on-use scarcity), "World events" (`observe()` vs `ingest()`), "Health surface". CHANGELOG.md [Unreleased]: all Phase B features + behavior notes (end_session now refreshes callbacks by default — config-gated; dimension guard may raise on mixed-embedder DBs). README.md: refresh the feature list with callbacks/proactive/maintenance/health.

- [ ] **Step 3: Full CI-equivalent**

Run: `venv/bin/ruff check src/ tests/ scripts/ && venv/bin/ruff format --check src/ tests/ scripts/ && venv/bin/python -m pyright && venv/bin/python -m pytest tests/ -q && venv/bin/python eval/run_eval.py`
Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add docs/ CHANGELOG.md README.md src/woven_imprint/storage/sqlite.py tests/test_storage.py
git commit -m "docs: SCHEMA.md portable contract, Phase B config/guide/changelog (B6)"
```

---

## Self-review (done at plan time)

- **Spec coverage:** B1 → Tasks 6–9 (all seven spec'd job types: consolidate ✓, reflect ✓, evolve ✓, dedup ✓, reinforce ✓, contradiction ✓, buffer hygiene ✓; idempotent/skippable/budget-aware ✓; CLI + engine API ✓). B2 → Tasks 1, 10 (paraphrase rule enforced in prompt + doc; batch generation + instant reads + HTTP/MCP exposure in Task 12). B3 → Task 11. B4 → Task 5. B5 → Tasks 3 (JSON retry), 4 (health + carry-over), 7 (write-time importance in batch). B6 → Tasks 1 (meta), 2 (embedding_base_url carry-over), 13 (SCHEMA.md + semver + FTS5 fallback doc). Phase-A carry-overs: subsystem-internal failure counting (Task 4), session_id-at-submit (Task 4), embedding_base_url (Task 2).
- **Deliberate scope choices:** callback "temporal triggers/anniversaries" from the spec's source list reduced to milestone-kind detection by the LLM over provided sources (no date engine — YAGNI for this pass; note for Phase C-era work). `unresolved threads` are discovered by the LLM from the source list rather than a dedicated thread-tracker.
- **Type consistency check:** `Budget.take/remaining`, `MaintenanceRunner.run` report shape, `CallbackEngine.get/refresh/compose_initiation`, storage method names, config field names — cross-checked across tasks; `DEFAULT_JOBS` gains `"callbacks"` only in Task 10 (Task 7 ships without it; Task 9 explicitly notes this).
- **Known judgment calls for executors:** FakeEmbedder cosine values in Task 8 tests must be verified empirically; each server test file's fixture conventions must be read before writing route tests (Task 12); `consolidation_setup` fixture name in Task 6 must be adapted to the real file.
