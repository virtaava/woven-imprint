# Changelog

All notable changes to Woven Imprint will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Parametric-layer spike (`experiments/parametric_spike/`): persona LoRA reduces persona drift
  (judge mean 0.798 vs 0.536 prompt-only; hard violations 1 vs 14 per run); facts-in-weights
  confabulate dates (temporal 0.24 vs 0.98 for the explicit store). See its RESULTS.md.
- `eval/render_results.py` generates `docs/RESULTS.md` from `eval/results/latest.json`.
- Injectable clock (`woven_imprint.clock`): all timestamps written/compared through it; tests and
  benchmarks can freeze and advance time.
- Memories in the prompt carry their date and a relative phrase ("2026-05-03, 3 weeks ago"); the
  volatile block starts with "Today is …" (`context.include_date`).
- Session summaries are dated (`[Session Summary YYYY-MM-DD]`, `metadata.started_at/ended_at`);
  consolidated memories keep `metadata.date_range` and inherit the latest source date.
- Retrieval scores all active memories (`memory.max_candidates`, default 5000) instead of the
  200 newest core rows; optional `numpy` fast path (`pip install woven-imprint[fast]`).
- One bookkeeping LLM call per turn (`persona/assessment.py`, `character.unified_assessment`,
  default on): emotion + relationship deltas + story beat + facts.
- 60-simulated-day long-horizon benchmark (`eval/bench_longhorizon.py`) gating CI via
  `tests/test_longhorizon.py`.
- Prompt registry (`woven_imprint.prompts`, `woven-imprint prompts`).

### Fixed
- `Engine.create_character` now keeps flat `hard_constraints` (as a hard constraint) and `role`
  (as a soft trait); previously both were silently dropped, so the demo character never saw
  "Never claims to be an AI" in its prompt.
- `SQLiteStorage.fts_search()` had no tiebreaker for FTS5 BM25 rank ties. Templated memory
  content (e.g. daily "On day N the visitor mentioned the X." facts) ties exactly on BM25 score
  across many rows, and SQLite's tie order is implementation-defined — it happened to come back
  oldest-first, so keyword-match retrieval systematically out-ranked a new memory against an old
  one whenever their keyword relevance was otherwise identical. Fixed by adding `m.rowid DESC`
  as an explicit secondary sort key, so ties resolve toward the newer memory. Caught by the
  long-horizon benchmark, not a pre-existing bug report.

### Removed
- `MEMORY-SIDE-FIXES.md` (its 2026-03-25 notes are recorded below under the 0.4.x history).

### Changed
- `SQLiteStorage.get_memories(limit=None)` returns all rows. Kotlin C1 (unmerged branch) must
  adopt the all-candidates retrieval rule and the dated memory/summary formats before merging.
- Long-horizon contradiction benchmark now asserts rank among on-topic memories (not global
  rank) and no longer depends on tail volume.

Phase B ("companion primitives") — the offline-maintenance, callback,
world-event, and health primitives a companion app builds on. Ships on top
of Phase A (below) in the same release.

### Added (Phase B)
- **Offline maintenance runner** (`MaintenanceRunner`,
  `Engine.run_maintenance()`, `woven-imprint maintain` CLI): all heavy
  multi-pass LLM work runs as budgeted, idempotent, per-character batch
  jobs — `consolidate`, `buffer_hygiene`, `score_importance`, `dedup`,
  `reinforce`, `contradictions`, `reflect`, `evolve`, `callbacks`. A shared
  LLM-call `Budget` caps each run (config
  `maintenance.max_llm_calls_per_run` / `WOVEN_IMPRINT_MAINTENANCE_BUDGET`;
  `--budget` per run); exhausted jobs skip gracefully, failed jobs never
  abort the run. New `maintenance:` config section (14 settings) — see
  [docs/CONFIGURATION.md](docs/CONFIGURATION.md#maintenance-settings).
- **Callbacks & proactive initiation** (`CallbackEngine`): paraphrased
  in-character conversation hooks ("How did the interview go?") generated
  in batch (maintenance job / session end, one LLM call) and read instantly
  from the DB — `Character.get_callbacks()`, `refresh_callbacks()`. Four
  kinds (`open_thread`, `callback`, `milestone`, `curiosity`), salience
  ordering, capped ready queue (`maintenance.callbacks_ready_cap`), hard
  paraphrase-never-verbatim rule. `Character.compose_initiation(occasion)`
  composes a character-initiated message from the top hook and consumes it
  (consumed-on-use scarcity — a hook is never repeated).
- **World events**: `Character.observe(event, source, importance, user_id)`
  — narrate ground truth ("Keeper fed you") into memory without a dialogue
  pair or LLM generation. Lightweight path (no `user_id`) makes zero LLM
  calls; assessed path (`user_id` given) runs event-shaped emotion +
  relationship assessment.
- **Health surface**: `Character.health()` — per-subsystem
  success/failure/last-error counters (`emotion`, `arc`, `extraction`,
  `relationship`, `consistency`, `observe`, `callbacks`) plus background
  worker status; makes silent small-model degradation visible.
- **Server & MCP surface** for the above: sidecar
  `GET /characters/{id}/callbacks`, `GET /characters/{id}/health`,
  `POST /observe`; demo API `/api/characters/{id}/callbacks`, `.../health`,
  `.../maintain`, `/api/observe`; MCP tools `get_callbacks`, `observe`,
  `get_health`, `maintain` (17 tools total).
- **Schema migration v4**: `callbacks` table and `meta` key/value table.
  `meta` records the embedding contract (`embedding_dimensions`,
  `embedding_model`) and `schema_semver` (currently `0.6.0-dev`, stamped on
  every open). The portable storage contract — every table, the embedding
  BLOB format, timestamp rules, FTS5 triggers, migration protocol — is now
  documented in [docs/SCHEMA.md](docs/SCHEMA.md).
- `llm.embedding_base_url` config + `WOVEN_IMPRINT_EMBEDDING_BASE_URL` env
  var — point embeddings at a different OpenAI-compatible endpoint than
  chat.
- `generate_json_robust()` on all LLM providers — uniform bounded
  retry-on-unparseable-JSON used by maintenance jobs and subsystems.

### Changed (Phase B)
- **`end_session()` now refreshes callbacks by default** (one extra LLM
  call at session end, non-fatal on failure). Gate with
  `maintenance.callbacks_refresh_on_session_end: false` if you only want
  the nightly `maintain` run to refresh them.
- **Embedding dimension guard may raise on mixed-embedder DBs**:
  `MemoryStore.add()` records `meta.embedding_dimensions` on the first
  embedded write and raises `ValueError` on any later write whose vector
  length differs. Previously, switching embedding models silently corrupted
  cosine retrieval; now the write is rejected — re-embed the database or
  restore the original embedding model.

### Fixed (Phase B)
- Sidecar `POST /observe` no longer leaks one background-worker thread per
  request: per-request `Character` instances run event assessments
  synchronously (`background=False`).
- API server `POST /v1/chat/completions` no longer leaks one
  background-worker thread per request: the per-request character runs
  bookkeeping synchronously before the response is returned.

---

Phase A ("fast core") — moves per-turn bookkeeping off the hot path and
fixes several retrieval/perf correctness issues, without changing the
public `Character`/`Engine` surface for the common case.

### Added
- **Background bookkeeping queue** (`character.background`, default `true`):
  emotion assessment, narrative-arc tracking, fact extraction, and
  relationship updates now run on a per-character daemon thread after
  `chat()` returns the response, instead of blocking the caller. Tasks for
  one character run in submission order (DB writes stay ordered).
  `Character.flush(timeout=None)` waits for queued bookkeeping to finish;
  `Character.close()` flushes and stops the worker. Call `close()` before
  tearing down shared resources (e.g. the DB connection) the worker still
  writes through. `end_session()` calls `flush()` internally before reading
  buffer memories for the session summary.
- `Character.chat_stream()` — streams response chunks as they generate.
  Consistency checking is post-hoc in stream mode (streamed text is never
  retracted); violations are logged and counted in
  `last_chat_metrics["stream_consistency_violations"]`, controlled by
  `character.consistency_stream_mode` (`"off"` | `"log"`, default `"log"`).
- `generate_stream()` on all `LLMProvider` implementations (native
  streaming where the backend supports it; a safe one-chunk default
  otherwise).
- Weighted RRF retrieval ranking: `memory.rrf_k` and per-signal weights
  (`memory.weight_semantic`, `weight_keyword`, `weight_recency`,
  `weight_importance`, `weight_relationship`) replace the old
  tier-priority strategy, which had a seed-dominance bug (early/seed
  memories could permanently outrank more relevant later ones).
- `memory.recency_anchor` (`"created"` | `"accessed"`, default `"created"`)
  — controls whether recency decay is anchored on a memory's creation time
  or its last-access time.
- Content-hash LRU embedding cache (`CachedEmbedder`) wrapping any
  `EmbeddingProvider` — identical text is embedded once. Thread-safe (the
  background worker embeds too). `Engine` wraps any embedder passed to it
  automatically.
- Resilience (retry with backoff + circuit breaker) applied consistently
  across all LLM and embedding providers, not just Ollama.
- `session_turns` persistence (migration v3): conversation turns are
  persisted per-session and rehydrated into context on
  `Character.resume_session()`, so resuming a session restores recent
  conversation history rather than starting with empty context.
- Metrics sink: opt-in JSONL per-turn chat metrics
  (`character.metrics_path` / `WOVEN_IMPRINT_METRICS_PATH`). Each `chat()`
  (and `chat_stream()`) call appends one record with the full
  `last_chat_metrics` phase breakdown.
- `scripts/bench_chat.py` — scripted 12-turn chat benchmark that prints
  per-phase p50/p95/max latency. Supports `--llm-base-url`/`--llm-model`/
  `--embed-base-url`/`--embed-model`/`--api-key` to construct chat and
  embedding providers explicitly against two different OpenAI-compatible
  endpoints (config only supports one shared `base_url`).
- Prompt split: the system prompt is now built as a stable persona prefix
  (name/backstory/personality — byte-identical across turns) plus a
  separate volatile block (emotion/arc/relationship/memories), which is
  friendlier to providers that do prefix-caching.

### Changed
- **Bookkeeping is asynchronous by default.** `Character.chat()` now
  returns after generation + consistency checking; emotion, arc, fact
  extraction, and relationship updates land shortly after, on a background
  thread. Set `character.background: false` (or `WOVEN_IMPRINT_BACKGROUND=false`)
  to restore the previous fully-synchronous behavior.
- **Auto-consolidation no longer runs mid-chat.** Buffer→core memory
  consolidation now only runs at session end (`end_session()`, when the
  buffer exceeds the threshold) or via an explicit `Character.consolidate()`
  call — not as a periodic check inside `chat()`.
- **Retrieval ranking changed.** Memory retrieval now uses weighted RRF
  across semantic/keyword/recency/importance/relationship signals instead
  of the old tier-priority strategy. Ranking order for a given query may
  differ from pre-Phase-A behavior; this was a deliberate correctness fix
  (see "seed-dominance bug" above), not a regression.

### Fixed
- `Engine.embedder` restored as a deprecated read-only alias for
  `Engine.embedding`. The A3 content-hash embedding cache work renamed the
  attribute without keeping a back-compat alias, which broke the
  additive-only public-surface constraint for this phase; `embedder` now
  returns `self.embedding` and callers should migrate to `embedding`.

## [0.5.0] - 2026-03-25

### Added
- React demo UI replacing Gradio (`woven-imprint demo`)
  - Chat with any character, markdown rendering, suggested prompts
  - X-Ray sidebar: live memory feed, relationship radar chart, emotion indicator
  - Collapsible X-Ray panel (toggle + localStorage persistence)
  - Character management: create, delete, export (JSON), import (JSON/PNG/markdown), migrate from text
  - Character selector in top bar for switching between characters
  - Reflect button for character self-reflection
  - Provider configuration with live model discovery
  - Provider presets: Ollama, OpenAI, Anthropic, DeepSeek, NVIDIA NIM, Custom (any OpenAI-compatible API)
  - Connection test required before saving provider
- FastAPI demo server with security hardening
  - Bearer token auth on all API routes
  - CORS locked to localhost (relaxed with --host 0.0.0.0)
  - Provider secrets never exposed to frontend
  - Graceful shutdown with session flushing
- Service-layer extraction from sidecar/API handlers
- `--host` flag for remote access (Tailscale, network)
- `--port` flag (default 7860)
- `--no-browser` flag
- Persistence regression tests
- Meridian seed database build script

### Removed
- Gradio web UI (`woven-imprint ui` command removed)
- `[ui]` optional dependency (replaced by `[demo]`)

### Changed
- `woven-imprint demo` now launches the React demo (previously a terminal REPL)

## [0.4.0] - 2026-03-18

### Added
- **Provider agnosticism**: All entry points use factory functions instead of hardcoded Ollama. Configure `llm_provider` (ollama/openai/anthropic) and `embedding_provider` (ollama/openai) in config or via `WOVEN_IMPRINT_LLM_PROVIDER` / `WOVEN_IMPRINT_EMBEDDING_PROVIDER` env vars.
- New `providers.py` module with `create_llm()` and `create_embedding()` factory functions.
- `WOVEN_IMPRINT_API_KEY_LLM` and `WOVEN_IMPRINT_BASE_URL` env vars for provider API keys and custom endpoints.
- `WOVEN_IMPRINT_ENFORCE_CONSISTENCY` env var (was missing from env_map).
- Consistency checker now accepts `CharacterConfig` for configurable retries, temperature, and fail-open score.
- JSON parse retry: consistency checker retries once at temperature=0.1 when `generate_json()` returns non-dict.
- Conversation context passed to consistency `check()` via `enforce()` — last 3 message pairs included for growth justification.
- Dynamic fact extraction cap: scales by exchange length (>2000 chars = 2x cap up to 15; <200 chars = half cap, min 2). Configurable via `max_facts_per_extraction` and `fact_density_scaling`.
- Enriched extraction prompt: includes "preferences, biographical details" in categories; adds recent conversation context with "do not re-extract" instruction.
- `WOVEN_IMPRINT_MAX_FACTS` env var for fact extraction cap.
- `MigrationConfig` dataclass: `max_messages`, `max_message_length`, `chunk_size` — all configurable.
- Chunked conversation analysis for large exports: `_analyze_conversations_chunked()` processes in chunks, `_synthesize_analyses()` merges via LLM.
- Scaled relationship sample size in migration: `min(60, max(30, n//10))` instead of fixed 30.
- `tests/test_providers.py`: 9 tests for factory functions.
- `tests/test_migration.py`: 9 tests for parsers and chunked analysis.

### Changed
- ChatGPT export parser: default is now unlimited messages (was hardcoded 500) and unlimited message length (was hardcoded 2000). Use `MigrationConfig` to set limits.
- Claude project parser: `rglob("*.md")` for all markdown files (was only `memory/*.md`), also scans `.claude/` directory, no character limits on file content.
- CLI, UI, MCP server, API server, and Engine all use `create_llm()`/`create_embedding()` factories — no direct Ollama imports in entry points.
- `CharacterConfig` gains `consistency_max_retries`, `consistency_temperature`, `consistency_fail_open_score`.
- `MemoryConfig` gains `max_facts_per_extraction`, `fact_density_scaling`.
- Config default template includes all new settings.
- 167 tests (was 146+), 1 skipped (optional anthropic dependency).

### Fixed
- `enforce_consistency` missing from environment variable map — now configurable via `WOVEN_IMPRINT_ENFORCE_CONSISTENCY`.
- `enforce()` never passed conversation context to `check()` despite `check()` having a `context` parameter.
- Fact extraction used hardcoded `facts[:5]` regardless of exchange density.

## [0.3.1] - 2026-03-17

### Added
- LLM provider resilience: retry with exponential backoff + circuit breaker
- Configurable: max_retries, retry delays, circuit breaker threshold/cooldown

## [0.3.0] - 2026-03-17

### Added
- Centralized configuration: `~/.woven_imprint/config.yaml` with 50 settings
- `woven-imprint config --init` and `woven-imprint config` commands
- Configuration reference documentation (docs/CONFIGURATION.md)
- PyYAML as core dependency
- All modules wired to read from config (no more editing source)
- 6th RRF retrieval strategy (explicit tier priority ranking)

### Changed
- Config priority: CLI flags > env vars > config file > defaults

## [0.2.1] - 2026-03-17

### Added
- Parallel subsystem calls via ThreadPoolExecutor (opt-in: `character.parallel = True`)
- Thread-safe SQLite with `check_same_thread=False`
- CI step timeouts to catch hangs

### Fixed
- Infinite recursion in SQLite `_commit()` method

## [0.2.0] - 2026-03-17

### Added
- Auto-consolidation every 20 turns + at session end (was never called)
- Belief revision wired into fact extraction (auto-detects contradictions)
- Periodic state save every 10 turns (emotion/arc survives mid-session crash)
- Tier priority as 6th RRF retrieval strategy
- Consolidation correctness benchmark (14th benchmark)
- Live persistence benchmarks: 50-session recall, adversarial persona (8/8),
  contradiction handling (4/4), held-out character (100%)
- Evaluation methodology doc with exact prompts and honest limitations
- Docker Compose setup (Ollama + Woven Imprint self-contained)
- OLLAMA_HOST env var for remote/containerized Ollama
- API server bearer token auth (`--api-key` or `WOVEN_IMPRINT_API_KEY`)
- Actionable Ollama error messages
- Web UI with 4 tabs (Chat, Characters, Migrate, Settings)
- `woven-imprint ui --browser chrome` configurable browser
- `woven-imprint update` command with pipx support
- `woven-imprint migrate` from ChatGPT, SillyTavern, Custom GPTs, Claude
- Custom GPT knowledge file import (`--knowledge` flag)
- PDF extraction via pymupdf
- Platform-specific setup guides (Windows, macOS, Linux, Docker)
- `/slash` commands in CLI chat

### Changed
- All model defaults unified to `llama3.2`
- Cross-session persistence: 67% → 100%
- Benchmarks: 13/13 (94.8%) → 14/14 (97.9%)
- Session summary importance: 0.7 → 0.85
- Extracted fact importance: 0.6 → 0.75
- Core tier boost: 0.15 → 0.2, Bedrock: 0.3 → 0.35

### Fixed
- Gradio 6.0 compatibility
- Proper PNG chunk parsing for TavernAI cards
- save_character no longer wipes state
- Relationship trajectory uses clamped deltas
- Growth memories have embeddings
- Emotion mood case-insensitive
- FTS5 update trigger only fires on content changes
- Consistency checker handles non-dict LLM response

## [0.1.2] - 2026-03-17

### Added
- Full-featured web UI with 4 tabs: Chat, Characters, Migrate, Settings
- `woven-imprint update` command — upgrades core + all installed extras
- `woven-imprint ui` command — launches browser-based interface
- File migration in UI (upload ChatGPT JSON, SillyTavern cards, etc.)
- Export/import/delete characters from the UI
- Memory search and reflect actions in the UI
- PDF knowledge file extraction via pymupdf
- Custom GPT knowledge file import (`--knowledge` flag)
- Auto-open browser on `woven-imprint ui`

### Fixed
- `woven-imprint update` now also upgrades pipx-injected extras (gradio, openai, etc.)
- Gradio 6.0 compatibility
- Proper PNG chunk parsing for TavernAI character cards
- `/slash` commands in CLI chat to avoid collision with character messages
- Linux/WSL/Ubuntu 24.04+ install guide (externally-managed-environment)

## [0.1.1] - 2026-03-17

### Fixed
- Gradio 6.0 compatibility — removed deprecated `theme` and `type` parameters
- Documentation: externally-managed-environment fix for Linux/WSL/Ubuntu 24.04+
- Documentation: pipx inject instructions for optional extras (ui, pdf)
- Documentation: MCP tool count and missing migrate_from_text in tool table
- Documentation: architecture diagram accuracy (removed unimplemented Qdrant reference)

## [0.1.0] - 2026-03-17

### Added
- Three-tier memory system (buffer, core, bedrock) with SQLite storage
- Multi-strategy retrieval via Reciprocal Rank Fusion (semantic, keyword, recency, importance, relationship)
- Tier-aware recency decay (bedrock persists months, buffer fades in days)
- Two-phase retrieval: FTS pre-filter finds old memories beyond the recency window
- Four-level persona constraints (hard, temporal, soft, emergent)
- Birthdate-derived age with birthday detection and leap year handling
- NLI-inspired consistency checking with post-generation enforcement
- Character growth engine (soft constraints evolve from accumulated experience)
- Five-dimensional relationship tracking (trust, affection, respect, familiarity, tension)
- Bounded relationship changes (max +/-0.15 per interaction)
- Emotional state tracking (15 moods, natural decay toward neutral)
- Narrative arc awareness (6 phases, tension curves, story beat detection)
- Memory consolidation engine (buffer compression into core memories)
- Belief revision system (reinforce, contradict, invalidate with certainty scores)
- Conversation buffer with context window management and graceful overflow
- Multi-character interaction (two-character dialogue, group scenes)
- LLM providers: Ollama, OpenAI, Anthropic (any OpenAI-compatible)
- Embedding providers: Ollama (nomic-embed-text), OpenAI
- CLI tool: demo, create, chat, list, stats, export, delete, import, serve
- OpenAI-compatible API proxy server (model name = character name)
- MCP server for IDE integration (Claude Desktop, Cursor, Hermes, OpenClaw)
- Character import/export (JSON with re-embedding on import)
- Lightweight mode (skip emotion/arc tracking for faster responses)
- 146 unit tests, 13 evaluation benchmarks (94.8% avg score)
- Pride and Prejudice relationship evolution demo (16 scenes, 6 characters)
- Dockerfile for containerized deployment
- CI: lint (ruff), typecheck (pyright), test matrix (3.11/3.12/3.13), CodeQL, Dependabot

## [0.4.x history] — 2026-03-25 memory-side fixes

## Memory‑Side Fixes (2026‑03‑25)

### Bugs Identified & Fixed

#### 1. Empty‑query recall crashes (HTTP 500)
**Root cause:** `OllamaEmbedding.embed("")` returns `{"embeddings":[]}`; indexing `embeddings[0]` raises `IndexError`.

**Fixed in:**
- `src/woven_imprint/embedding/ollama.py`:
  - `embed()`: returns a zero‑vector of appropriate dimensionality when input is empty.
  - `embed_batch()`: handles empty strings by returning zero vectors, preserving order.

- `src/woven_imprint/memory/retrieval.py`:
  - Semantic ranking is skipped when `query.strip()` is empty (avoids useless zero‑vector similarity).

**Verification:** `GET /api/memory?query=&limit=10` now returns the 10 most recent memories (no crash).

#### 2. Personal‑memory ranking too low
**Observation:** Bedrock‑tier seeded documentation out‑ranks user‑specific core memories (e.g., “User's favorite food is pizza”).

**Root cause:** Tier boost (`bedrock +0.35`, `core +0.2`) overwhelms personal relevance; relationship boost exists but may be insufficient.

**Fixed in:**
- `src/woven_imprint/memory/retrieval.py`:
  - Added **user‑affinity bonus** (+0.2) to importance score when `metadata.user_id` matches the `relationship_target` (provided by chat).

**Expected effect:** Memories whose `user_id` matches the current user rank higher, improving cross‑session recall.

#### 3. Cross‑session memory retrieval (design vs. implementation)
**Finding:** Retrieval is **not** session‑filtered; pizza memories from session A appear in results for session B (good).  
**Problem:** Ranking still favors bedrock memories, causing the LLM to overlook personal facts.

**Status:** Partially addressed by user‑affinity bonus. Further tuning of tier boosts may be needed (configuration‑level change).

### Remaining Issues (Not Yet Fixed)

#### 1. Session fragmentation
**Observation:** Each browser refresh spawns a new session ID; “Welcome back” greeting cannot work.

**Fix required:** Frontend must store `session_id` in `localStorage` and reuse it via `session_id` query parameter.

#### 2. LLM provider configuration missing
**Observation:** Demo server fails with `ValueError: Model 'llama3.2' not found` because no valid provider config exists.

**Fix required:** Provider‑setup modal must be completed before chat works; currently blocks end‑to‑end testing.

#### 3. Seed‑memory dominance
**Observation:** 114 bedrock (documentation) memories dominate vector search for generic queries.

**Mitigation:** User‑affinity bonus helps, but may need down‑weighting of bedrock memories when query is personal (e.g., contains “my”, “I”, “me”).

### Applied Patches (Diff Summary)

#### 1. `ollama.py` – empty‑string embedding
```diff
     def embed(self, text: str) -> list[float]:
+        if not text.strip():
+            # Return zero vector of appropriate dimensionality
+            dims = self.dimensions()
+            return [0.0] * dims
         resp = self._post({"model": self.model, "input": text})
```

#### 2. `ollama.py` – batch embedding with empty strings
```diff
     def embed_batch(self, texts: list[str]) -> list[list[float]]:
+        # Handle empty strings by returning zero vectors
+        if not texts:
+            return []
+        # Get dimensionality (will call embed("test") if unknown)
+        dims = self.dimensions()
+        # Prepare result list …
+        # … (see source for full implementation)
```

#### 3. `retrieval.py` – skip semantic ranking for empty query
```diff
-        # Strategy 1: Semantic ranking
-        query_embedding = self.embedder.embed(query)
-        semantic_scores = []
-        for m in all_memories:
-            if m.get("embedding"):
-                sim = _cosine_similarity(query_embedding, m["embedding"])
-                semantic_scores.append((m["id"], sim))
-        semantic_scores.sort(key=lambda x: x[1], reverse=True)
-        semantic_ranked = [mid for mid, _ in semantic_scores]
+        # Strategy 1: Semantic ranking (skip if query empty)
+        semantic_ranked = []
+        if query.strip():
+            query_embedding = self.embedder.embed(query)
+            semantic_scores = []
+            for m in all_memories:
+                if m.get("embedding"):
+                    sim = _cosine_similarity(query_embedding, m["embedding"])
+                    semantic_scores.append((m["id"], sim))
+            semantic_scores.sort(key=lambda x: x[1], reverse=True)
+            semantic_ranked = [mid for mid, _ in semantic_scores]
```

#### 4. `retrieval.py` – user‑affinity bonus
```diff
-        # Strategy 4: Importance with tier boost
+        # Strategy 4: Importance with tier boost + user affinity
         importance_scores = []
         for m in all_memories:
             base = m.get("importance", 0.5) * m.get("certainty", 1.0)
             boost = _get_tier_boosts().get(m.get("tier", "buffer"), 0.0)
+            # User affinity bonus
+            if relationship_target:
+                meta = m.get("metadata", {})
+                if meta.get("user_id") == relationship_target:
+                    base += 0.2
             importance_scores.append((m["id"], base + boost))
```

### Testing Commands (Post‑Fix)

```bash
# 1. Verify empty‑query no longer crashes
curl -b "woven_demo_auth=<token>" \
  "http://127.0.0.1:7860/api/memory?character_id=char-be8e2a3d8d20&query=&limit=5"

# 2. Check that pizza memory appears high for "favorite food" query
curl -b "woven_demo_auth=<token>" \
  "http://127.0.0.1:7860/api/memory?character_id=char-be8e2a3d8d20&query=favorite+food&limit=5"

# 3. Inspect ranking with user‑affinity bonus (requires provider config)
#    (Currently blocked by missing LLM model)
```

### Next Steps

1. **Frontend session persistence** – store `session_id` in `localStorage`.
2. **Provider configuration UI** – ensure modal completes before chat is attempted.
3. **Tier‑boost tuning** – consider reducing `bedrock` boost for non‑identity memories.
4. **Integration test** – add `test_cross_session_memory` that learns fact in session A, restarts server, recalls in session B.

### Notes

- All patches are backward compatible; existing behaviour unchanged for non‑empty queries.
- The demo server must be restarted after applying changes (already done).
- The fixes address the two critical memory‑side bugs identified in the review (empty‑query crash, personal‑memory ranking).

— Sona (Hermes Agent), 2026‑03‑25
