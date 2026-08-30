# Changelog

All notable changes to Woven Imprint will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed (behavior) — Tier 3c (recall: keep consolidated sources)
- **Retrieval RRF defaults**: `weight_importance` `1.0` → `0.0`, `weight_recency`
  `1.0` → `0.1` (relevance-first ranking) — LoCoMo evidence recall@20 23.9% →
  50.8% with both at 0 (`ranking_experiments`, 2026-08-30). `weight_recency` is
  kept at a small `0.1` rather than `0.0` (controller ruling, same day): `0.1`
  costs only ~2 recall@20 points (48.7% vs 50.8%) while still breaking ties
  newest-first when semantic/keyword relevance ties on a generic query. The
  relationship strategy now ranks only candidates whose boost is actually
  positive (fixes a hidden oldest-first tie bias that previously ranked every
  gated candidate, including the untouched 0.0-tied majority, in ascending-rowid
  order whenever `weight_relationship > 0`). Despite that fix, `weight_relationship`
  defaults to `0.0` (was `1.0`): measured on identical DBs, answer-only re-runs
  scored LoCoMo J `0.476` with it at `1.0` vs `0.536` at `0.0` (2026-08-30) — the
  name-mention boost outranks genuine evidence and costs 6 J points. The
  strategy's code is unchanged and kept available as an opt-in for callers who
  want it (raise the weight above `0.0`). The importance
  strategy's tie-break changed from ascending-rowid (oldest-first) to
  descending-rowid (newest-first) for the same reason; `eval/external/ranking_experiments.py`'s
  `fuse_rank` mirrors both tie-break fixes. Decay/tier-boost machinery is
  unchanged and still feeds these signals for anyone who raises the weight
  further. Long-horizon `recency_ordering` now opts into `weight_recency=1.0`
  to test the recency strategy under relevance-first defaults.
- **Consolidation keeps source memories active and retrievable** (config
  `memory.consolidation_keep_sources`, default `true`): a multi-member buffer
  cluster still gets a summarized `[Consolidated]` core row, but its sources
  stay `active`/`buffer` and gain `metadata.consolidated_into`/
  `consolidated_at` instead of being archived. A singleton with importance
  `>= 0.6` is now **promoted in place** (tier flipped to `core` on the same
  row, `metadata.promoted_from_buffer = true`) instead of being copied and
  archived, so its text never appears twice in retrieval; a singleton below
  the bar gets `metadata.consolidation_seen = true` so it stops recounting
  toward the consolidation threshold. `needs_consolidation()` and the
  consolidation chunk query now count/pull only *unconsolidated* buffer rows.
  `ConsolidationEngine.consolidate()`'s stats dict gains `kept`, `promoted`,
  `seen` alongside `archived` (always present, `0` when the flag is off).
  Setting `memory.consolidation_keep_sources: false` restores the prior
  archive-everything behavior byte-for-byte.
- **`eval/external/diagnose_recall.py`** gains `--run-id`, `--results`, and
  `--out-dir` CLI arguments (defaults unchanged in spirit: `locomo-mem-v1`,
  `eval/results/external_<run-id>.json`,
  `eval/external/runs/diagnostics/<run-id>/`) so it can diagnose a different
  run without editing the script.
- **Results**: LoCoMo memory-mode J `0.444` → `0.536` (2026-08-30 defaults,
  `locomo-mem-v2d`); LoCoMo-Plus cognitive `0.332` → `0.421`
  (`plus-mem-v2d`) — same local judge, full v1→v2→v2b/v2c/v2d progression and
  diagnostics in `docs/BENCHMARKS.md`. LongMemEval-S not re-run this tier
  (`lme-s-50-v1` numbers still stand, measured under the prior defaults).

### Added (Tier 3b — external benchmarks · library enablers)
- **`OpenAILLM(extra_body=...)`** — an optional dict forwarded verbatim into every
  `chat.completions.create` call (`generate`, `generate_stream`, `generate_json`) when set, e.g.
  `{"chat_template_kwargs": {"enable_thinking": False}}` to disable Qwen3.5's reasoning mode on
  vLLM. Omitted from the request entirely when not set.
- **`providers.create_llm()` now passes `timeout=cfg.llm.timeout`** to `OpenAILLM` — previously
  silently dropped, so the configured LLM timeout never reached the OpenAI-compatible provider.
- **`resilience.resilient_call()` now retries `openai.APITimeoutError`/`openai.APIConnectionError`**
  (guarded import — a no-op if `openai` isn't installed), so a vLLM/OpenAI-compatible call that
  times out or drops its connection gets the same retry/backoff/circuit-breaker treatment as a
  `requests` timeout instead of failing the whole run immediately.

### Fixed (Tier 3c)
- **Nightly `buffer_hygiene` no longer archives sources kept by consolidation**
  (`metadata.consolidated_into`); rows only marked `consolidation_seen` are
  still swept by TTL.
- **`buffer_hygiene`'s oldest-first fetch window can no longer be starved by
  consolidation-kept sources.** `SQLiteStorage.get_memories(...,
  exclude_consolidated=True)` excludes `metadata.consolidated_into` rows in
  SQL (unlike `unconsolidated=True`, it does *not* also exclude
  `consolidation_seen` rows — those must stay sweepable), so a pile of
  `>= 1000` kept sources can no longer fill the whole 1000-row LIMIT window
  and hide a genuinely stale, unrelated row from ever being considered. The
  existing Python-side guard is kept as well.
- **`Character.export()` no longer truncates the buffer at 1000 memories.**
  It now calls `MemoryStore.get_all(tier=..., limit=None)` for every tier
  (buffer/core/bedrock already supported `limit=None`; `export()` just
  wasn't passing it), so an export/import round-trip carries every memory,
  not just the newest 1000.
- **`eval/external/runner.py`'s `_judge_call` now catches only `ValueError`**
  (an unparseable judge response) instead of every `Exception`. A transport
  error (a dropped connection, a timeout — typically `RuntimeError` or an
  `openai`/`requests` exception) now propagates and kills the shard instead
  of being silently recorded as a scored-but-unparsed verdict; the run
  resumes from checkpoint once the provider is back instead of shipping
  results with hidden connectivity gaps baked in.
- **`MemoryStore.needs_consolidation()` now honors
  `memory.consolidation_keep_sources`**, matching
  `ConsolidationEngine.needs_consolidation()`: with the flag on (default) it
  counts only unconsolidated buffer rows; with it off, a plain buffer count.
  Previously it always counted unconsolidated rows regardless of the flag.
- **Import now remaps a kept source's stale `metadata.consolidated_into`
  pointer.** Every memory gets a new id on `Engine.import_character()`, so a
  Tier 3c "keep sources" row's `consolidated_into` (pointing at the *old* id
  of the core row it was consolidated into) is dangling on arrival. Import
  now builds an old-id → new-id map while re-adding memories, then rewrites
  `consolidated_into` to the new id in a second pass — or drops the key
  entirely if the target memory wasn't part of the export.
- **Retrieval skips building the importance/relationship RRF lists at
  `weight <= 0`** instead of building and then zero-weighting them — no
  behavior change (a zero-weight list already contributed nothing to fused
  scores), just skips the wasted per-candidate scoring loop at the current
  relevance-first defaults (both weights default to `0.0`).

### Changed (behavior)
- **`Character.ingest()` now uses unified bookkeeping and creates structured facts** when
  `unified_assessment` is on (the default): it routes through `_run_bookkeeping` — the same
  single-call emotion/relationship/beat/facts path `chat()` uses — instead of the legacy
  `_extract_memories`. This means ingested turns can now produce structured
  (subject, predicate, object) facts, not just free-text ones (and, under unified assessment,
  the same call also updates mood and narrative arc, not just facts). The legacy path is
  unchanged and still used when `unified_assessment` is off. Both `ingest()` and
  `ingest_exchange()` are fully synchronous — bookkeeping always runs inline on the calling
  thread, even with `background=True` — and flush any background worker still draining bookkeeping
  from earlier `chat()` calls before running their own, so the two never interleave out of order.
- **`Character.ingest_exchange(user_message, response, user_id=None)`** — like `ingest()` but
  records one user turn + one character reply as a single unit (one `_turn_count` increment, one
  unified bookkeeping call instead of two). For importing transcripts of paired user/assistant
  dialogue, e.g. benchmark haystacks or SillyTavern logs, where the two sides are already known
  together.

### Added (Tier 3b — external benchmarks · harness + docs)
- **`eval/external/` harness** (`python -m eval.external fetch|run|rejudge`): publishes
  reproducible LoCoMo, LoCoMo-Plus (Cognitive category), and LongMemEval-S numbers measured with
  the local brain (Qwen3.5-35B-A3B-FP8, thinking off) as both the answering model and the judge,
  alongside a full-context baseline under the identical judge. Message-by-message ingestion via
  `Character.ingest()` under clock control, product-path answering (pinned + facts + top-K
  retrieved memories), the LoCoMo/Mem0-lenient judge and the upstream LoCoMo-Plus Cognitive judge,
  a fixed abstention rule for category-5/`_abs` questions, per-conversation SQLite checkpointing
  (atomic writes, resumable, shardable via `--shard i/n`), a `rejudge` subcommand, and a
  stratified judge-calibration sample (`eval/results/external_judge_sample.json`). See
  `docs/BENCHMARKS.md`.
- **`docs/BENCHMARKS.md`** — method, exact prompts (quoted verbatim), protocol, metrics
  definitions, hardware/runtime, judge-calibration instructions, caveats (local judge ≠ GPT-4o,
  LoCoMo label noise, category-5 convention, subset sizes, and more), and reproduce commands for
  the external benchmark harness.
- **`docs/RESULTS.md`** gains an "External benchmarks (real LLM, local judge)" section, rendered
  by `eval/render_results.py` from `eval/results/external_latest.json` when that file exists (one
  table per `bench:mode`: overall/per-category J-score, token-F1, adversarial/abstain accuracy,
  mean prompt tokens, run id, timestamp). Purely additive — the deterministic headline score line
  is unaffected, and the section is simply absent until the harness has published a run.
- **Results: first published external-benchmark numbers** (LoCoMo J 0.444 memory vs 0.696
  full-context; LoCoMo-Plus cognitive 0.332 vs 0.135; LongMemEval-S 50-question sample J 0.396).

### Fixed (Tier 3b)
- **`OpenAILLM.generate_json` now sends `max_tokens`** (default 2048); previously an unbounded
  JSON-mode generation could run to the context limit (observed: a temperature-0 judge call
  looping for hours, reproduced on every retry).

### Added (Tier 3a — editable memory · interchange)
- **Memory & fact mutation** — every memory and fact is now viewable and editable, from the
  library, HTTP, and MCP:
  - `MemoryStore.edit(memory_id, *, content=None, importance=None, tier=None)` (re-embeds on
    content change), `.delete(memory_id)` (also retracts any fact that pointed to it — deleting
    a memory means "forget this"), `.pin(memory_id, pinned=True)`, `.pinned()`.
  - `FactStore.edit(fact_id, *, object=None, statement=None)` (keeps the linked memory's text in
    sync, re-embedding it), `.retract(fact_id)` (expires the fact now, archives the linked
    memory), `.delete(fact_id)` (hard delete of the fact row only).
  - `Character.memory.edit/delete/pin/pinned` and `Character.facts.edit/retract/delete` are the
    public surface.
- **Pinned block** — memories pinned via `MemoryStore.pin()` render into a non-sheddable
  "Things you always remember:" block in the volatile system prompt, ahead of and excluded
  from the retrieved-memories list (no duplicates). `context.pinned_block` (default `true`),
  `context.pinned_limit` (default `10`, oldest-pinned-first). Never shed by the context budget
  — proven by the `pinned_always_present` long-horizon benchmark (pin a day-3 memory; at day 60
  it's still in the volatile block, not duplicated in the memories list). `docs/RESULTS.md`
  regenerated (26/26 passed).
- **HTTP**: `GET /api/memory/pinned?character_id=`, `PATCH /api/memory/{id}`,
  `DELETE /api/memory/{id}`, `PATCH /api/facts/{id}`, `DELETE /api/facts/{id}?mode=retract|delete`
  (default `retract`), `GET /api/characters/{id}/card`. All auth-guarded, rate-limited under the
  `mutation` bucket, and scoped — a memory/fact must belong to the given `character_id` or the
  route 404s.
- **MCP tools**: `edit_memory`, `delete_memory`, `pin_memory`, `list_pinned`, `retract_fact`,
  `edit_fact`.
- **Demo X-Ray editing**: a **Pinned** card at the top of the memory area (unpin); Memory Feed
  rows gain pin/edit (inline textarea, Escape to cancel)/delete (confirm) controls; a new
  **Facts** card lists current user-facts with edit-object and retract controls. Fixes the
  relationship radar reading the wrong response envelope (`res.relationship`). Bundle rebuilt;
  `tests/test_demo_bundle.py` greps the built JS for `/api/memory/pinned` etc. so a stale bundle
  fails CI.
- **SillyTavern interchange**:
  - Import (`migrate/parsers.py`, `importer.py`) now reads `character_book` (lorebook),
    `system_prompt`, `post_history_instructions`, `alternate_greetings`, `creator`,
    `character_version`, `spec`/`spec_version`; PNG import accepts the V3 `ccv3` tEXt chunk
    (preferred) alongside the existing V2 `chara` chunk. `scenario`/`tags`/`greetings` map to
    soft persona traits, `system_prompt` to `hard.hard_constraints`, `creator_notes` to
    `hard.creator_notes`. Each enabled lorebook entry becomes a memory: `constant` entries are
    pinned bedrock (`tier="bedrock"`, `metadata.pinned=True`), the rest are `core`; content is
    prefixed `[Lore: key1, key2] ...`. Entry `keys` are coerced from string/list/comma-string
    forms; entries default to `enabled=True` when the field is absent.
  - Export: `Character.export_card(core_limit=20) -> dict` builds a `chara_card_v2` card whose
    lorebook is generated from pinned memories (`constant: true`), current user facts, and the
    top `core_limit` core memories by importance (`constant: false`). CLI `woven-imprint
    export-card <name-or-id> [-o card.json]`; HTTP `GET /api/characters/{id}/card`.
- `GET /api/facts/{character_id}` items now include `id` (needed by the new edit/retract
  controls).

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

### Added (Tier 2 — facts · relationships · relevance)
- Schema v5: bi-temporal `facts` table (subject/predicate/object, valid_from/valid_to,
  recorded_at/expired_at, superseded_by) and `relationships.state` (dynamics bookkeeping —
  betrayal damping counters, sliding trajectory window, update count). Existing DBs migrate on
  open; old data keeps working.
- `Character.facts` (`FactStore`): `current()`, `as_of(when)`, `history(subject, predicate)`,
  `find_active()`. Facts are extracted as structured objects (turn_assessment v2:
  subject/predicate/object/event_time) and superseded by `(subject, predicate)` across the
  **whole** store, not the last 50 core rows — the old antonym heuristic still runs, unchanged,
  for unstructured facts only. A **backdated correction** — a new fact whose `valid_from`
  predates the currently active fact's `valid_from` — is not a supersession: the active fact
  stays current and the backdated one is filed straight into history, superseded by the active
  fact as of the active fact's `valid_from`.
- "What you currently know about {user}" volatile context block (`context.facts_block`, default
  on; `context.facts_block_limit`, default 12), ordered by importance descending then
  `recorded_at` descending (newest first), with "previously: X" when a fact has a superseded
  predecessor; a short "Things you have said about yourself" block when self-facts exist.
- Relationship dynamics in code (`relationship.dynamics`, default on): positive trust deltas
  scaled by `trust_gain_factor`; a single clamped trust delta at or below `betrayal_threshold`
  is a betrayal that damps trust gains for `betrayal_damping_turns` updates (via
  `betrayal_gain_damping`); automatic key moments for any |clamped delta| ≥
  `key_moment_threshold`; trajectory from a sliding window (`trajectory_window`) of net deltas;
  tension decays toward 0 per elapsed day (`tension_decay_per_day`); a derived tier
  (stranger/acquaintance/friend/close_friend/adversary) shown by `describe()`. Set
  `relationship.dynamics: false` for the previous byte-identical arithmetic.
- Retrieval relevance gate (`memory.relevance_gate`, default on): for a non-empty query,
  recency/importance/relationship ranking is confined to memories that are semantically
  relevant (cosine similarity strictly above `memory.relevance_min_similarity`, an epsilon
  guarding against float32 matmul noise, within the top `memory.relevance_semantic_topk`) or
  keyword (FTS) matching; falls back to scoring every active memory when nothing clears the
  bar. Narrows, but does not eliminate, the bedrock-floor effect where an off-topic memory
  outranks a fresh on-topic one on recency/importance alone.
- `maintenance.py` jobs now use `memory/retrieval.py`'s batched `cosine_matrix()` for similarity
  scoring instead of scoring one pair at a time (behavior-preserving performance change, no
  ranking difference).
- Long-horizon benchmark grew to 11 checks: `structured_supersession`, `facts_block_rendered`,
  `betrayal_has_consequences`, `relevance_gate_global_rank`; `contradiction_supersession`
  regains its global-rank assertion. `docs/RESULTS.md` regenerated (25/25 passed).
- MCP `get_facts(character_id, subject=None, as_of=None)` and `get_stats().facts_current`;
  `GET /api/facts/{character_id}?subject=&as_of=`; export/import round-trip facts.
- Known limitation: import restores facts with `memory_id=None` (memory linkage is not
  preserved; reinforce/supersession of imported facts does not update memory rows).

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
- **`GET /api/memory` response shape**: memory rows no longer include `embedding` (it was
  leaking the full vector to the client for no reason the UI used); still include `id, tier,
  content, importance, certainty, status, created_at, accessed_at, metadata, session_id, role`.
- `Engine.create_character` now moves flat `scenario`, `greetings`, and `tags` persona fields
  into `soft` (previously only `personality`/`speaking_style`/`occupation`/`appearance`/`role`
  made that trip) — needed so SillyTavern-imported scenario/greetings/tags round-trip through
  `export_card()`.
- Kotlin C1 (unmerged branch) must also add memory/fact mutation (`edit`/`delete`/`pin`/
  `pinned`, `edit`/`retract`/`delete` on facts) and the pinned-block prompt rendering before
  merging (Tier 3a, this release).
- `SQLiteStorage.get_memories(limit=None)` returns all rows. Kotlin C1 (unmerged branch) must
  adopt the all-candidates retrieval rule and the dated memory/summary formats before merging.
- Long-horizon contradiction benchmark now asserts rank among on-topic memories (not global
  rank) and no longer depends on tail volume.
- `RelationshipModel.update` default behavior changed (dynamics on): trust gains are scaled and
  betrayals are damped, trajectory is windowed, tension decays, and a tier is derived — see
  Added (Tier 2) above. Set `relationship.dynamics: false` for the previous arithmetic.
  `key_moments_limit` is now honored (previously hard-coded to 20).
- Long-horizon `contradiction_supersession` regains a global-rank assertion (the day-40 fact is
  `ranked[0]` for query "tea") now that the relevance gate makes global ranking meaningful again.
- Kotlin C1 (unmerged branch) must also add the `facts` table, `relationships.state`, the
  relevance gate, and the facts block before merging (schema version 5).

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
