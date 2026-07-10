# Woven Imprint: Companion-Grade Evolution — Design

**Date:** 2026-07-10
**Status:** Approved direction (sequencing A → B; C starts with the game build)
**Inputs:** Capability audit (`research/2026-07-10-capability-audit/`, 4 reports) and game-design-space research (`research/2026-07-10-game-design-space/`, 3 reports).

## 1. Context and goal

Woven Imprint v0.5.2 is a working persistent-character engine: 3-tier memory with RRF retrieval, consolidation, belief revision, 5-dimension relationships, growth, consistency checking — 289 behavioral tests, honest evals, portable SQLite storage.

The next product built on it is a **phone-native companion creature** (working name Mythling; concept to be finalized after this evolution lands): a myth-being that lives on the real-world clock, learns language from its keeper, and whose memory callbacks are the core product. The 2026-07-10 research established the design laws this workload imposes:

- **Memory must be surfaced to be valued** — paraphrased, in-character, multi-channel callbacks; silent memory is invisible except when it fails.
- **The LLM narrates state deterministic systems own** — never invents it.
- **Real-world clock + scarcity** — the character exists while you're gone; interactions are rate-limited.
- **Overnight batch = free latency and free quality** — heavy multi-pass work runs while the user sleeps.
- **Small on-device models are the deployment target** — features must degrade loudly, not silently.

Measured against those laws, the audit found the engine's *primitives* strong but its *runtime shape* wrong: ~5 sequential blocking LLM calls per turn (4 of them post-response bookkeeping the user waits on), consolidation firing inline mid-chat, no proactive/scheduled/batch capability at all, no callback surfacing, dialogue-pair-only ingestion, and silent no-op degradation on weak models.

**Goal of this evolution:** make Woven Imprint the best persistent-character library it can be for companion/game workloads — fast, batch-capable, proactive, with surfaced callbacks — while staying local-first, provider-agnostic, and schema-portable.

**Non-goals (this pass):**
- Character↔character gossip / shared-world propagation (library milestone for later; the single-companion product doesn't need it).
- The Kotlin edge runtime and gemma_edge bridge (Phase C — starts with the game build, against the schema this pass stabilizes).
- The memory-ablation retention study (runs on the game's internal-testing cohort).
- The game itself (fiction, art, monetization all deferred by decision).

## 2. Phase A — Fast, correct core

### A1. Decouple reply from bookkeeping
`Character.chat()` currently serializes: generate → consistency → emotion → arc → relationship → (every 3rd turn) extraction → (periodic) inline consolidation. The user perceives all of it.

Redesign:
- `chat()` returns after **generate + consistency** (consistency is the only post-processor that can change the reply; it stays synchronous).
- Emotion, arc, relationship, and fact extraction move to a **background work queue** owned by the Character (thread-based executor; no asyncio rewrite of the core). Writes are ordered per-character; the queue drains before `end_session()`/`close()`.
- A `flush()` method (and context-manager exit) blocks until bookkeeping completes — tests and batch jobs use it; interactive callers don't.
- Inline consolidation in the chat path is **removed** (moved to the Phase B scheduler; `consolidate()` stays available as an explicit call).
- Failure visibility: background task failures increment per-subsystem health counters (see B5) instead of `logger.debug` silence.

Expected effect: perceived latency drops from ~5 LLM round-trips to ~2 (generate + consistency check), with consistency skippable per config as today.

### A2. Streaming
`LLMProvider.generate_stream()` (iterator of text chunks) with implementations for Ollama, OpenAI, Anthropic, gemma_edge; `Character.chat_stream()` yields tokens and enqueues the same background bookkeeping at completion. Consistency checking in streaming mode operates post-hoc: it can't retract streamed text, so it logs/records violations and (config-gated) can emit a correction turn — documented tradeoff.

### A3. Retrieval correctness
- **Fix tier triple-counting** (audit: bedrock favored by decay rates, importance boosts, *and* a dedicated tier-priority list — the "seed memories drown personal facts" pathology). Remove the raw tier-priority strategy from RRF; keep tier influence in decay + importance only. Re-run the live persistence bench to confirm cross-session personal recall improves.
- **Per-strategy RRF weights** in config (currently all-equal, k=60 hardcoded). Defaults preserve current behavior minus the removed list.
- **Embedding cache** (content-hash LRU) — identical content is currently re-embedded every time.
- **Index `memories(accessed_at)`** and `created_at` (retrieval orders by them; no index exists).
- Fix `accessed_at` semantics: retrieval touching memories resets recency, making frequently-retrieved memories self-reinforcing on the recency axis. Change: recency decay keys off `created_at`; `accessed_at` remains for analytics/reinforcement but no longer drives the decay term (config flag `recency_anchor: created|accessed`, default `created`).

### A4. Prompt-cache-friendly context assembly
Split the single mutating system message into: **stable prefix** (persona system prompt — identical every turn) + **volatile block** (emotion/arc/relationship/memories) positioned after the stable content. Measure with the existing (currently dead-ended) `last_chat_metrics` instrumentation.

### A5. Resilience parity + tests
- Wire `resilient_call` (retry/backoff/circuit-breaker) into OpenAI, Anthropic, and gemma_edge providers (currently Ollama-only; gemma_edge has none).
- Add a lock around the module-global breaker registry (mutated from worker threads).
- **Test `resilience.py`** (currently zero tests on the most subtle provider code).

### A6. Durable short-term context
Persist the conversation buffer (per session) so process restart / mobile cold start / `resume_session()` rehydrates recent turns instead of starting amnesiac. Storage: a new `session_turns` table (session_id, seq, role, content, created_at) — schema migration v3. `resume_session()` loads the tail into the ContextManager.

### A7. Capture the numbers
The profiling instrumentation added in commits ac69e92/57049ec/497be84 writes `last_chat_metrics` that nothing reads. Add an opt-in metrics sink (JSONL) + a small report script; record before/after numbers for A1–A4 in `docs/PERFORMANCE.md`. Acceptance for Phase A is measured, not asserted.

## 3. Phase B — Companion primitives

### B1. Batch pipeline (the "nightly job" primitive)
A first-class offline maintenance runner, callable headless (`woven-imprint maintain <db> [--character id]` and `engine.run_maintenance(character_id, jobs=[...])`):
- **consolidate** (moved out of chat; the O(n²) greedy clustering gets a sort-by-embedding-neighborhood pre-pass or cap-per-cluster fix, and the 500-row pull cap is replaced by chunked processing so a heavy day fully drains),
- **reflect** (auto, importance-sum triggered — Stanford-style — instead of manual-only),
- **evolve** (growth finally runs automatically, here, not mid-chat),
- **dedup sweep** (embedding-similarity merge of near-duplicate core facts — replaces prompt-only dedup),
- **reinforce sweep** (re-encountered facts strengthen certainty — wiring the currently-dead `reinforce()`),
- **contradiction sweep** (semantic: embedding-pair candidates → LLM verdict, replacing/augmenting the 8 hardcoded antonym pairs; runs here where retries are cheap),
- **buffer hygiene** (low-importance singleton buffer rows past a TTL get archived so the active set stays bounded).

Jobs are idempotent, individually skippable, budget-aware (max LLM calls per run), and report a structured result. This is where small models get retries; it is the direct enabler of "heavy work runs overnight while charging."

### B2. Callback surfacing API
The research's #1 feature. `character.get_callbacks(context=None, limit=3)` returns structured, *paraphrased, in-character* conversation hooks:

```python
[{"hook": "You mentioned your interview was yesterday — how did it go?",
  "source_memory_ids": [...], "kind": "open_thread",  # open_thread | callback | milestone | curiosity
  "salience": 0.82, "freshness": "2d"}]
```

Sources: unresolved threads (questions the user raised that never resolved), relationship `key_moments`, arc `beats`, high-importance recent facts, anniversaries/temporal triggers. Generation happens in the batch pipeline (a "curiosity queue" refreshed nightly + after sessions), so reads are instant DB lookups. Paraphrase-not-verbatim is a hard rule (the HAI '23 finding: verbatim reads as creepy). Exposed over sidecar/demo HTTP and MCP.

### B3. Proactive generation
`character.compose_initiation(occasion)` — the character starts an interaction (daily greeting, callback follow-up, milestone). Built on B2's queue + persona/emotion/relationship state; returns text + the callback metadata it used. The *scheduling* of when to send is the app's job (the library stays clockless apart from batch jobs); the library's job is having something in-character to say. Consumes queue items on use (scarcity by design).

### B4. World-event ingestion
`character.observe(event, source="world", importance=None, user_id=None)` — non-dialogue events become memories without faking a chat pair (today's `ingest()` demands user/assistant roles and pollutes assessors with empty strings): "Keeper fed you moonberries", "It rained all day", "You evolved into a fledgling". Runs fact-storage + optional relationship/emotion assessment with an event-shaped prompt (not the dialogue-pair prompt). This is the API a deterministic game sim uses to narrate ground truth into memory.

### B5. Small-model operability
- **Write-time importance scoring**: batch-scored poignancy (1–10 → 0..1) on recent buffer memories during maintenance (cheap, retryable) instead of flat 0.5 forever; extraction keeps its static defaults as fallback.
- **Health surface**: `character.health()` — per-subsystem success/failure counters, last-error, JSON-parse failure rates. The demo UI X-Ray shows it. Silent degradation becomes visible; the app can warn "your creature's memory is struggling with this model."
- **Structured-output hardening**: schema-tolerant parsing already exists; add one bounded retry (temperature drop) at the *call-site wrapper* level for all `generate_json` sites, matching what consistency already does — uniform policy instead of per-site improvisation.

### B6. Schema/versioning for portability (pre-Phase-C groundwork)
- `meta` table: embedding model id + dimensions (today mixed models silently corrupt cosine), schema semver, creation info.
- Document the schema as a stable contract (`docs/SCHEMA.md`): tables, blob format (float32-LE), timestamp format, FTS5 dependency + the sanctioned fallback (LIKE-based keyword strategy) for runtimes without FTS5.

## 4. Phase C — Edge (outline; detailed spec when the game build starts)

Kotlin edge runtime reading/writing the same SQLite DB (storage, retrieval math — RRF+cosine are ~50 portable lines — prompt assembly, callback reads); on-device inference via MediaPipe-class runtime behind the gemma_edge contract, with a reference bridge implementation and resilience parity; batch jobs mapped to WorkManager charging-constrained tasks. The Python toolchain remains the authoring/debugging environment for any pet DB.

## 5. Testing & acceptance

- Every phase-A change lands with before/after numbers from the A7 metrics sink (turn latency breakdown, prompt sizes, LLM-call counts). Target: perceived turn latency ≤ generate + consistency; background queue drains within seconds on Spark-class hardware.
- Retrieval fix validated against the live persistence benchmark (cross-session personal recall) — the Meridian "seed-dominance" case is the regression test.
- New primitives (batch runner, callbacks, observe, health) get behavioral tests in the existing style (real in-memory SQLite, FakeLLM); batch jobs additionally get idempotency tests (run twice = run once).
- Live small-model validation: one scripted end-to-end run against a 1–2B model (Ollama) exercising extraction/callbacks/maintenance, asserting the health surface reports honestly.
- CI stays green throughout (ruff, pyright, pytest 3.11–3.13, eval suite); public API additions documented in README/DEVELOPER_GUIDE; CHANGELOG per release.

## 6. Risks

- **Background-queue ordering/races**: per-character serialized queue + flush() discipline; the existing per-character locks in the demo server already establish the pattern.
- **Streaming vs consistency**: post-hoc only; documented, config-gated correction behavior.
- **Batch pipeline on huge backlogs**: chunked, budgeted, idempotent — designed to be resumable.
- **Scope creep toward the game**: the game consumes these APIs; anything game-specific (species, language stages, care sim) lives in the game project, not the library.
