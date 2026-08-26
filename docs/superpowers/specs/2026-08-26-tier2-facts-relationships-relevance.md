# Tier 2 — Structured facts, relationship dynamics, retrieval relevance — Spec

**Date:** 2026-08-26  
**Owner:** Toni  
**Origin:** audit roadmap Tier 2 (artifact a52b92ee) + findings from Tier 1's long-horizon benchmark (bedrock floor outranks fresh on-topic facts; inline contradiction check sees only the 50 newest core rows).  
**Status:** approved ("proceed", 2026-08-26). Design choices below are the controller's stated assumptions; Toni can override any.

## Goal
1. **Facts are records, not strings.** Every extracted fact gets `subject / predicate / object / event_time` plus bi-temporal validity (`valid_from`, `valid_to`, `recorded_at`, `expired_at`). Supersession is a lookup on `(subject, predicate)` over the **whole** store — no more antonym heuristic on 50 rows. "She used to think X" is a query.
2. **Relationships are a state machine in code.** Trust rises slowly and falls fast; a betrayal damps trust gain for a while; key moments are recorded automatically; trajectory uses a sliding window; a relationship tier (stranger → acquaintance → friend → close friend / adversary) is derived and shown to the model; tension decays with elapsed time.
3. **Retrieval is relevance-first.** Recency/importance/relationship lists only re-rank memories that are semantically or lexically relevant to the query; off-topic bedrock seeds can no longer outrank a fresh on-topic fact.
4. All three are measured in the long-horizon benchmark and documented.

Out of scope: Ebbinghaus decay, drift instrumentation, LoRA distillation, social-graph propagation, Kotlin parity (CHANGELOG note only), editable-memory UI.

## T2.1 Structured bi-temporal facts

**Schema v5** (`_MIGRATIONS[5]`, additive):
```sql
CREATE TABLE IF NOT EXISTS facts (
    id TEXT PRIMARY KEY,
    character_id TEXT NOT NULL,
    subject TEXT NOT NULL,          -- "user" | "self" | free text (a named third party)
    predicate TEXT NOT NULL,        -- snake_case verb phrase, e.g. "has_cat_named", "lives_in", "likes"
    object TEXT NOT NULL,           -- value, e.g. "Pixel", "Oulu", "tea"
    statement TEXT NOT NULL,        -- the natural-language sentence as extracted
    event_time TEXT,                -- when the fact became true in the world (YYYY-MM-DD or timestamp), nullable
    valid_from TEXT NOT NULL,       -- world-time start of validity (event_time or recorded_at)
    valid_to TEXT,                  -- NULL = currently valid
    recorded_at TEXT NOT NULL,      -- when the character learned it (clock)
    expired_at TEXT,                -- when the character learned it was superseded (clock), NULL = never
    certainty REAL DEFAULT 1.0,
    importance REAL DEFAULT 0.75,
    memory_id TEXT,                 -- linked memories.id (the retrievable text row)
    superseded_by TEXT,             -- facts.id of the replacing fact
    session_id TEXT,
    user_id TEXT,
    metadata TEXT DEFAULT '{}',
    FOREIGN KEY (character_id) REFERENCES characters(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_facts_key ON facts(character_id, subject, predicate, valid_to);
CREATE INDEX IF NOT EXISTS idx_facts_recorded ON facts(character_id, recorded_at DESC);
ALTER TABLE relationships ADD COLUMN state TEXT DEFAULT '{}';
```
`schema_version` → 5; `docs/SCHEMA.md` gets the table + protocol bullet; `tests/test_storage.py` version assertion → 5; `meta.schema_semver` stays `0.6.0-dev`.

**`memory/facts.py` — `FactStore(storage, character_id)`**
- `add(*, subject, predicate, object, statement, event_time=None, certainty=1.0, importance=0.75, memory_id=None, session_id=None, user_id=None, metadata=None) -> dict` — stamps `recorded_at = clock.sqlite_ts()`, `valid_from = event_time or recorded_at`.
- `current(subject=None, predicate=None, limit=200) -> list[dict]` — `valid_to IS NULL AND expired_at IS NULL`, newest `recorded_at` first.
- `as_of(when, subject=None, predicate=None) -> list[dict]` — world-time query: `valid_from <= when AND (valid_to IS NULL OR valid_to > when)`.
- `history(subject, predicate) -> list[dict]` — all versions ordered by `valid_from`.
- `find_active(subject, predicate) -> dict | None`.
- `expire(fact_id, *, valid_to, superseded_by) -> None` — sets `valid_to`, `expired_at = clock`, `superseded_by`.
- `normalize_key(subject, predicate) -> tuple[str, str]` — lowercase, strip, collapse whitespace/underscores; `normalize_object(obj) -> str` for equality (casefold, strip trailing punctuation).

**Extraction (unified path only).** `TURN_ASSESSMENT_FACTS_SECTION` v2 asks for objects:
```
"facts": up to N objects {"statement": one sentence, "subject": "user"|"self"|a name, "predicate": snake_case verb phrase (e.g. has_cat_named, lives_in, works_as, likes, dislikes, plans_to), "object": the value, "event_time": "YYYY-MM-DD" or null}
```
`Character._parse_facts` accepts strings (legacy) and objects; returns `list[dict]` with `statement` always present and `subject/predicate/object/event_time` possibly `None`. Legacy prompt `fact_extraction` and the legacy `_extract_memories` path stay byte-identical (strings only, no facts rows).

**`_store_facts` (unified path) per fact:**
1. Structured (subject+predicate+object present):
   - `old = facts.find_active(subject, predicate)`.
   - Same normalized object → `belief.reinforce(old.memory_id)`; bump `old.certainty` (+0.15 clamp 1.0); no new rows.
   - Different object → new memory row (content = statement, `metadata={"source":"extraction","user_id":…,"fact_id":new_id,"contradicts":old.memory_id}`), new fact row with `memory_id`; `facts.expire(old.id, valid_to=new.valid_from, superseded_by=new.id)`; old memory → `storage.update_memory_status(old.memory_id, "contradicted", certainty=0.0)`.
   - No old → new memory row + fact row.
2. Unstructured (missing key): today's behavior (antonym heuristic over 50 newest core rows + `belief.contradict`) unchanged, and a fact row with `subject="unknown"`, `predicate="states"`, `object=statement` is **not** written (keeps the facts table clean).
3. Health counters unchanged (`extraction`).

**Prompt: "What I know" block.** `_build_context` adds a volatile section (after relationship, before memories; sheddable; gated by `context.facts_block: bool = True`, `context.facts_block_limit: int = 12`): 
```
What you currently know about toni (facts you learned; dates are when they became true):
- (since 2026-05-03) The visitor's cat is named Pixel.
- (since 2026-06-01, previously: Tampere) The visitor lives in Oulu.
```
Built from `facts.current(subject="user"…)` filtered by `user_id` when present, ordered by importance desc then recency; "previously" shows the immediately superseded object when one exists. Self-facts (`subject="self"`) go into a separate two-line block only if any exist ("Things you have said about yourself:").

**Exposure.** `Character.facts` (FactStore); `Character.export()["facts"]`; `engine.import_character` restores facts; MCP tool `get_facts(character_id, subject=None, as_of=None)`; demo `GET /api/facts/{character_id}?subject=&as_of=`; MCP `get_stats` adds `facts_current` count.

## T2.2 Relationship dynamics (code, not LLM)

`RelationshipConfig` additions (defaults):
```python
dynamics: bool = True                 # False = today's behavior byte-identical
trust_gain_factor: float = 0.5        # positive trust deltas are scaled; negative are not
betrayal_threshold: float = -0.10     # a single clamped trust delta <= this is a betrayal
betrayal_damping_turns: int = 10      # trust gain multiplied by betrayal_gain_damping for this many updates
betrayal_gain_damping: float = 0.25
key_moment_threshold: float = 0.08    # |clamped delta| >= this on any dimension records a key moment
trajectory_window: int = 5            # trajectory from the sum of the last N net deltas
tension_decay_per_day: float = 0.05   # tension moves toward 0 by this much per elapsed day (clock)
```
Tiers (derived on every update; stored in `type`):
- affinity = 0.5·trust + 0.3·affection + 0.2·respect
- `adversary` if affinity ≤ −0.3; else `close_friend` if affinity ≥ 0.5 and familiarity ≥ 0.6; else `friend` if affinity ≥ 0.25 and familiarity ≥ 0.3; else `acquaintance` if familiarity ≥ 0.1; else `stranger`.
- `type` set explicitly via `set_baseline`/`update(new_type=…)` is respected until the next dynamics update recomputes it (documented).

`update(target_id, deltas, new_type=None, *, note=None)`:
1. If dynamics: apply tension decay first: `tension = max(0, tension − tension_decay_per_day × elapsed_days)` where elapsed_days from `state.last_update_at` (clock); then per dimension clamp to ±max_delta as today; for `trust`: if delta > 0 → `delta × trust_gain_factor`, and additionally `× betrayal_gain_damping` while `state.damping_left > 0` (decrement per update); if delta ≤ betrayal_threshold → `state.damping_left = betrayal_damping_turns`, `state.betrayals += 1`, key moment `"{date}: betrayal — trust {delta:+.2f}{' — ' + note if note}"`.
2. Key moments for any |clamped delta| ≥ threshold: `"{date}: {dim} {delta:+.2f}{' — ' + note}"`, capped at `key_moments_limit` (config, replacing the hard-coded 20).
3. Trajectory: push `net = trust+affection+respect deltas` (post-scaling) and `tension_delta` to `state.recent` (window N); `warming` if Σnet > 0.1, `cooling` if < −0.1, `volatile` if Σ|tension| > 0.1, else `stable`.
4. Recompute tier → `type`; `state.updates += 1`, `state.last_update_at = clock`.
5. Persist `state` JSON (new column) with the row.

`describe()` gains `  tier: friend (affinity 0.31)` and up to 2 most recent key moments (`  recent: 2026-05-03: betrayal — trust -0.12`). Existing tests that pin legacy arithmetic (`test_update_dimensions`, `test_trajectory_*`, `bench_relationship_bounded_change`) set `dynamics=False` explicitly or are re-derived; new tests cover each rule.

`_run_bookkeeping` passes `note=message[:80]`; `_update_relationship_event` passes `note=event[:80]`.

## T2.3 Retrieval relevance gate

`MemoryConfig.relevance_gate: bool = True`, `relevance_semantic_topk: int = 100`.
When the query is non-empty and the gate is on: `eligible = set(semantic top-K ids) ∪ set(keyword ids)`; the recency, importance and relationship ranked lists are built from eligible memories only. Semantic and keyword lists unchanged. Empty query (or gate off) → today's behavior. `docs/ARCHITECTURE.md` documents the rationale (bedrock floor). Long-horizon `contradiction_supersession` regains the global assertion: the day-40 fact is `ranked[0]` for query "tea".

## Benchmarks (long-horizon additions)
- `structured_supersession`: day 12 fact `{user, has_cat_named, Pixel}`; day 45 `{user, has_cat_named, Moss}` → `facts.current(user, has_cat_named)` = Moss; `history` has both with Pixel `valid_to` = day-45 date; `as_of(day 30)` = Pixel; the Pixel memory row is `contradicted`.
- `facts_block_rendered`: the volatile context for a chat after day 45 contains "What you currently know" and "Moss" and "previously: Pixel".
- `betrayal_has_consequences`: trust(50) < trust(40); trust(60) < trust(40) (damped recovery); `type` at day 50 == "adversary" or affinity ≤ −0.3; at least one key moment containing "betrayal"; `trajectory` at day 45 == "cooling".
- `relevance_gate_global_rank`: `retrieve("tea")[0]` is the day-40 dislike fact.
- Existing 7 keep passing; `docs/RESULTS.md` regenerated.

## Acceptance
- pytest/ruff/pyright green; CI unchanged.
- Existing DBs migrate to v5 on open (facts table empty; relationships.state '{}'); old data keeps working.
- With `relationship.dynamics=False`, `memory.relevance_gate=False`, `context.facts_block=False` and the legacy bookkeeping path, behavior is byte-identical to master 9dfd13a.

## Global constraints
- Python 3.11+, no new required deps; public API additive only; commit per task on branch `feat/tier2-facts-relationships`.
- All timestamps via `woven_imprint.clock`; DB strings `"YYYY-MM-DD HH:MM:SS"` (event_time may be date-only `"YYYY-MM-DD"`; normalize to `"YYYY-MM-DD 00:00:00"` on write).
- Prompt registry: every new/changed prompt gets a version bump and a snapshot driver.
- Never modify `experiments/parametric_spike/` or `kotlin/`.
