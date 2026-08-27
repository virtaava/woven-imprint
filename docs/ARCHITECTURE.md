# Woven Imprint — Architecture

## Overview

Woven Imprint is persistent character infrastructure. It sits between the application
and the LLM, providing memory management, persona enforcement, relationship tracking,
and consistency verification.

```
Application ─→ Woven Imprint Engine ─→ LLM Provider
                    │
                    ├── Memory Store (SQLite)
                    ├── Persona Model
                    ├── Relationship Model
                    └── Consistency Checker
```

## Core Concepts

### Character
A persistent AI personality with:
- **Persona**: immutable identity (name, backstory, personality traits, speaking style)
- **Memory**: accumulated experiences organized in three tiers
- **Relationships**: tracked connections with users and other characters
- **State**: current emotional state, active goals, recent context

### Clock

`woven_imprint.clock` is the single source of truth for "now" across the library —
`retrieval.py`, `maintenance.py`, `callbacks.py`, `character.py`, `metrics.py`, and
`persona/model.py` all read time through it instead of calling `datetime.now()`
directly. Production uses real UTC time; tests and benchmarks call
`clock.override(fixed_datetime_or_callable)` (usable as a context manager or a
plain setter) and `clock.advance(timedelta)` to simulate days passing without a
real clock. `clock.sqlite_ts(dt)` / `clock.parse_ts(str)` handle the SQLite
timestamp format (assumes UTC if naive). Storage writes stamp explicitly —
`save_memory` uses `memory["created_at"]`/`["accessed_at"]` when the caller
supplies them, else `clock.now()`; SQL `DEFAULT (datetime('now'))` remains only
as a fallback for writes that bypass the Python layer. No schema migration was
needed.

### Memory Tiers

**Buffer** (working memory)
- Raw observations from the current and recent conversations
- Stored as-is with timestamps, embeddings, and importance scores
- Auto-consolidates when count exceeds threshold (default: 100)

**Core** (processed memory)
- Consolidated memories, session summaries, reflections
- Each entry has: content, embedding, importance, certainty, source_refs, created_at, accessed_at
- Formed by LLM-powered consolidation of Buffer entries
- Updated by belief revision (reinforce/contradict/invalidate)

**Bedrock** (deep memory)
- Fundamental character knowledge: backstory events, core beliefs, defining moments
- Rarely changes, highest retrieval weight
- Seeded from persona definition, enriched by significant interactions

### Pinned Memories

`MemoryStore.pin(memory_id, pinned=True)` sets `metadata.pinned` (no schema migration — pin
state lives inside the existing `metadata` JSON column, queried via `json_extract`). Pinned
memories get a placement no other memory has: `Character._build_context` renders them into a
"Things you always remember:" block in the **volatile-but-non-sheddable** part of the system
prompt (right after the "Today is …" line), via `context.pinned_block` (default `true`) /
`context.pinned_limit` (default `10`, oldest pinned memory first). That block counts toward
`base_size` — the one part of the context budget that is never trimmed when the total exceeds
`context.total_tokens` — so a pinned memory survives budget pressure that would otherwise drop
it. Pinned memories are excluded from the separately-retrieved memories list so nothing appears
twice. The `pinned_always_present` long-horizon benchmark checks exactly this: pin a day-3
memory, run to day 60, and confirm the pinned text is still in the volatile block and not
duplicated below it (see [docs/RESULTS.md](RESULTS.md)).

### Editable Memory & Facts

Every memory and fact is viewable and editable from the library, HTTP, and MCP — this is what
makes memory a user-facing artifact rather than an opaque store:

```
MemoryStore.edit(memory_id, *, content=None, importance=None, tier=None) -> dict
MemoryStore.delete(memory_id) -> None     # unlinks all + retracts only the still-active fact
MemoryStore.pin(memory_id, pinned=True) -> dict
MemoryStore.pinned() -> list[dict]

FactStore.edit(fact_id, *, object=None, statement=None) -> dict   # re-embeds the linked memory
FactStore.retract(fact_id) -> dict        # expires now; linked memory -> status="archived"
FactStore.delete(fact_id) -> None         # hard delete of the fact row only
```

`edit()` on a memory re-embeds when `content` changes (through the same
`guard_embedding_dimension` check every write goes through). Editing a memory's `content` is
one-directional, though: it does **not** rewrite a linked fact's `statement` — the fact keeps
its original text until someone edits the fact itself. Deleting a memory means "forget this":
it calls `unlink_fact_memory` (which clears `memory_id` on every fact pointing there, active or
historical) but only retracts the ones still active (`valid_to IS NULL`) — a superseded
historical fact keeps its own expiry/successor and isn't stamped `retracted` just because the
memory it once pointed at is gone. Retracting a fact is softer — it archives the linked memory
but the fact stays in `FactStore.history()`.

`FactStore.edit()` can be given `object` without `statement`: the new statement is then derived
from the old one — a case-insensitive replace of the old object text within the old statement
if it's found there, else the old statement gets an appended `" — now: <new_object>."` clause.
Either way the derived (or explicit) statement is what gets written back to the linked memory's
`content`, so the UI's object-only fact edit still keeps memory text in sync — see
`memory/facts.py::_derive_statement`.

`Character.memory.edit/delete/pin/pinned` and
`Character.facts.edit/retract/delete` are the public surface; no new `Character` methods were
needed. Every mutation is additionally exposed over HTTP
(`PATCH`/`DELETE /api/memory/{id}`, `GET /api/memory/pinned`, `PATCH`/`DELETE /api/facts/{id}`)
and MCP (`edit_memory`, `delete_memory`, `pin_memory`, `list_pinned`, `edit_fact`,
`retract_fact`) — both under the same character-ownership check other mutation routes use, so a
memory or fact that doesn't belong to the given `character_id` 404s rather than leaking across
characters.

### Retrieval Function

Multi-strategy retrieval via Reciprocal Rank Fusion (RRF):

```
final_score(memory, query) = RRF(
    semantic_rank(memory, query),    # cosine similarity of embeddings
    keyword_rank(memory, query),     # BM25 text match
    recency_rank(memory),            # exponential decay from last access
    importance_rank(memory),         # LLM-assigned importance score
    relationship_rank(memory, ctx),  # boost for memories involving current interlocutor
)
```

RRF formula: `score = Σ 1/(k + rank_i)` where k=60 (standard RRF constant)

### Retrieval: All-Candidate Scoring

`retrieve()` calls `SQLiteStorage.get_memories(character_id,
limit=mem_cfg.max_candidates)` — newest `memory.max_candidates` (default
5000) active memories for the character, not the full table (storage's
`get_memories(..., limit=None)` returns every row, but `retrieve()` never
passes that; see [CONFIGURATION.md](CONFIGURATION.md#memory-settings)). This
is unioned with up to 50 keyword hits from `fts_search()`, deduplicated by
memory id, so an old memory outside the `max_candidates` window can still
surface via FTS even though it isn't scored semantically — the FTS union
adds up to 50 keyword-only hits beyond the cap, not an unlimited reach. This
replaced a fixed 200-newest-core-rows window that made memories older than
the window unreachable by semantic or recency ranking — the gap the
long-horizon benchmark's `paraphrase_recall_day5` check exists to catch.

Semantic scoring builds one similarity matrix per `retrieve()` call: if
`numpy` is importable (`pip install woven-imprint[fast]`), `cosine_matrix()`
in `memory/retrieval.py` stacks the stored float32 embedding blobs and scores
them in a single matmul; otherwise it falls back to the pure-Python
`_cosine_similarity()` loop. Both paths produce the same ranking; `numpy` is
optional (CI installs it). `consolidation.py` imports the same
`_cosine_similarity()` helper instead of keeping its own copy. RRF fusion
and `_retrieval_score` semantics are unchanged.

### Retrieval: Relevance Gate

Equal-weight RRF fusion, on its own, lets a memory's permanent recency/
importance floor (e.g. a bedrock `[Self]` persona line, or a large flood of
old importance-boosted memories) outrank a fresh, directly query-relevant
fact — because recency and importance are scored over *every* active
memory, regardless of whether it has anything to do with the query.

`memory.relevance_gate` (default `true`) narrows that gap, but does not
eliminate it: for a non-empty query, `retrieve()` first computes `eligible =
top memory.relevance_semantic_topk of {ids with cosine similarity strictly
above memory.relevance_min_similarity} ∪ all keyword (FTS) hits`, then
builds the recency, importance, and relationship-boost ranked lists from
only that eligible subset (`gated`), not from every active memory.
`relevance_min_similarity` (default `1e-6`) is an epsilon, not a literal
zero floor — it exists so float32 matmul noise on real dense embeddings
(typically ~1e-8) isn't mistaken for a genuine positive match against a
`sim > 0.0` floor. A memory that doesn't clear that floor is never eligible
on semantic grounds alone, regardless of its rank slot — this closes the
case where fewer than top-K memories clear the floor and a noise-level-only
memory (e.g. an unrelated bedrock line) rode a rank slot into eligibility
anyway. Semantic and keyword ranking are unaffected — they still score
every candidate. The practical effect: most off-topic memories can no
longer win RRF fusion on recency/importance alone, since they never enter
those ranked lists — but an off-topic memory with genuine (if weak)
positive similarity that still lands in the semantic top-K remains eligible
and can still outrank a more relevant fact on recency or importance. If no
memory clears `relevance_min_similarity` or keyword-matches at all
(`eligible` is empty), the gate falls back to scoring every active memory,
matching pre-gate behavior.

Set `relevance_gate: false` (see
[CONFIGURATION.md](CONFIGURATION.md#memory-settings)) to restore the
pre-gate fusion, where recency/importance/relationship rank all active
memories unconditionally — useful for comparing behavior or if a workload
depends on the old semantics. An empty query always bypasses the gate (there
is nothing to be relevant to), matching prior behavior exactly.

### Persona Model

Four constraint levels (hard, temporal, soft, emergent):

1. **Hard constraints** — factual attributes that NEVER change
   - Name, core backstory, species, fundamental identity
   - Violation → regenerate response

2. **Temporal facts** — change on schedule or event, not through conversation
   - `age`: derived from `birthdate` + current time (auto-increments on birthday)
   - `location`: changes when character "moves" (event-driven)
   - `appearance`: can change through events (haircut, injury, aging)
   - Stored with a resolver function, not a static value
   - Character is aware of their own birthday and reacts naturally to it

3. **Soft constraints** — personality traits that evolve slowly
   - Speech patterns, opinions, preferences, behavioral tendencies
   - Violation → flag, allow if character growth context exists

4. **Emergent layer** — formed entirely through interaction
   - New opinions, reactions to events, relationship-driven changes
   - No constraint — this IS the character growth

### Consistency Verification

Post-generation NLI-inspired checking:

1. Generate response from LLM with persona + memory context
2. Extract claims from response (factual statements, opinions, emotional states)
3. Check claims against hard constraints → reject if contradiction
4. Check claims against soft constraints → flag if contradiction, check for growth justification
5. Conversation context (last 3 message pairs) is included for growth justification
6. If rejection: regenerate with explicit constraint reminder at configurable temperature
7. Max retries configurable (default 2), then return best-scoring response
8. On JSON parse failure: retry once at temperature=0.1 before fail-open
9. Fail-open score configurable (default 0.8) — returned when the check itself fails

### Relationship Model

```
Relationship:
    entities: (CharID, CharID)
    dimensions:
        trust:       float[-1, 1]  # suspicion ↔ trust
        affection:   float[-1, 1]  # dislike ↔ warmth
        respect:     float[-1, 1]  # contempt ↔ admiration
        familiarity: float[0, 1]   # stranger → intimate knowledge
        tension:     float[0, 1]   # calm → high unresolved conflict
    power_balance: float[-1, 1]    # who leads the dynamic
    type: stranger | acquaintance | friend | close_friend | adversary  # derived tier
    trajectory: warming | cooling | stable | volatile
    key_moments: str[]             # dated one-liners, capped at key_moments_limit
    state: {...}                   # dynamics bookkeeping (opaque JSON, see below)
    formed_at: datetime
    last_interaction: datetime
```

Per-turn dimension deltas are still LLM-assessed from conversation content (the unified
assessment's `relationship` section, or a caller passing `deltas` directly) — what changed in
Tier 2 is what happens to those deltas once `RelationshipModel.update(target_id, deltas, *,
note=None)` receives them. With `relationship.dynamics` (default `true` — see
[CONFIGURATION.md](CONFIGURATION.md#relationship-settings)) the formula is:

1. **Tension decay** — before applying anything, `tension` moves toward 0 by
   `tension_decay_per_day × elapsed_days` (elapsed since `state.last_update_at`, via the
   injectable clock).
2. **Clamp** each delta to ±`max_delta`, as before Tier 2.
3. **Trust asymmetry** — trust rises slowly and falls fast: a *positive* clamped trust delta is
   scaled by `trust_gain_factor`; a delta at or below `betrayal_threshold` is a **betrayal** —
   applied at full clamped magnitude, `state.damping_left` is (re)set to `betrayal_damping_turns`,
   `state.betrayals` increments, and a key moment is recorded
   (`"{date}: betrayal — trust {delta:+.2f}"`, plus `note` if given). While
   `state.damping_left > 0`, subsequent positive trust deltas are further scaled by
   `betrayal_gain_damping` (recovery is damped after a betrayal). `damping_left` decrements once
   per update, but only on an update that *began* with damping already active and did not itself
   contain a betrayal — the update that starts (or resets) the window never decrements in that
   same update.
4. **Key moments** — any dimension whose clamped, *pre-scaling* delta has `|value| ≥
   key_moment_threshold` (other than a betrayal, already recorded in step 3) gets its own dated
   one-liner (`"{date}: {dim} {delta:+.2f}"`, using that same pre-scaling value, plus `note`),
   capped at `key_moments_limit` (config — previously hard-coded to 20; `key_moments_limit ≤ 0`
   disables key moments entirely). The threshold is checked against the magnitude *before*
   `trust_gain_factor`/`betrayal_gain_damping` scaling is applied, so e.g. a +0.15 trust gain
   records a moment even though the dimension only moves by 0.075.
5. **Trajectory** — a sliding window of the last `trajectory_window` updates' `(net,
   tension_delta)` pairs lives in `state.recent`; `net = trust + affection + respect` (post-
   scaling). `warming` if the windowed sum of `net` > 0.1, `cooling` if < −0.1, `volatile` if the
   windowed sum of `|tension_delta|` > 0.1, else `stable`.
6. **Tier** — `type` is recomputed every update via `RelationshipModel.tier_for(dimensions)`
   unless `new_type=` is passed explicitly, which sticks only until the *next* dynamics update
   recomputes it: `affinity = 0.5·trust + 0.3·affection + 0.2·respect`; `adversary` if
   `affinity ≤ -0.3`; else `close_friend` if `affinity ≥ 0.5` and `familiarity ≥ 0.6`; else
   `friend` if `affinity ≥ 0.25` and `familiarity ≥ 0.3`; else `acquaintance` if
   `familiarity ≥ 0.1`; else `stranger`.

`describe()` renders the tier and affinity (`tier: friend (affinity +0.31)`) and the two most
recent key moments alongside the existing per-dimension description and trajectory. `state` is
persisted as a JSON column with the relationship row — see
[SCHEMA.md](SCHEMA.md#relationships) — and is opaque to storage, read/written
whole by `relationship/model.py`.

Set `relationship.dynamics: false` to run `_update_legacy()` instead: the original clamp-only
arithmetic with no trust scaling, betrayal damping, tension decay, or derived tier — byte-
identical to pre-Tier-2 behavior. Existing tests that pin that arithmetic set it explicitly.

### Facts (Bi-Temporal)

`Character.facts` (`FactStore`, `memory/facts.py`) stores extracted knowledge as
`(subject, predicate, object)` triples — e.g. `(user, lives_in, Oulu)` — with two independent
time axes, both stamped through the injectable clock:

- **World time** (`valid_from`/`valid_to`) — when the fact was true *in the world*, from the
  extracted `event_time` (normalized to `"YYYY-MM-DD 00:00:00"`) or, absent that, the moment it
  was recorded.
- **Record time** (`recorded_at`/`expired_at`) — when the character *learned* the fact, and when
  it learned that fact had been superseded.

```
FactStore.add(subject, predicate, object, statement, event_time=None, ...)
FactStore.current(subject=None, predicate=None)      # valid_to IS NULL AND expired_at IS NULL
FactStore.as_of(when, subject=None, predicate=None)  # valid_from <= when AND (valid_to IS NULL OR valid_to > when)
FactStore.history(subject, predicate)                # every version, ordered by valid_from
FactStore.find_active(subject, predicate)
```

**Supersession** is a lookup on `(subject, predicate)` over the **whole** facts table — not the
50-newest-core-rows antonym heuristic, which still runs, unchanged, only for facts extracted
*without* a subject/predicate/object (see [Unified Assessment](#unified-assessment-tier-1)).
`Character._store_structured_fact()` handles three cases when a new structured fact arrives:

1. **Same object** as the currently active fact (normalized casefold/strip comparison) — treated
   as a restatement: `belief.reinforce()` on the linked memory and `FactStore.bump_certainty()`
   on the fact row (`certainty += 0.15`, capped at 1.0); no new row is written. This is what
   makes repeated same-belief extractions (e.g. a fact re-emitted on consecutive turns) cheap
   and idempotent instead of piling up duplicate rows.
2. **Different object, later `valid_from`** — an ordinary update: the old fact is expired
   (`valid_to` = new fact's `valid_from`, `superseded_by` = new fact's id, `expired_at` = now),
   its linked memory row is marked `status="contradicted"`, and a new memory + fact row are
   written.
3. **Different object, earlier `valid_from` — a backdated correction.** A fact about the past
   that arrives *after* the character already believes something more recent is not a
   supersession: the currently active fact is left untouched (it stays current), and the new
   fact is written straight into history, itself marked `superseded_by` the active fact as of
   the active fact's `valid_from`. This keeps `current()` honest when information arrives out of
   chronological order — "she just learned a detail about what was true two months ago" never
   overwrites what she believes is true *now*.

The prompt's "What you currently know about {user}" volatile context block
(`context.facts_block`, `context.facts_block_limit` — see
[CONFIGURATION.md](CONFIGURATION.md#context-window-settings)) is built from `facts.current(subject=
"user")`, filtered by `user_id` when the fact has one, ordered by importance descending then
`recorded_at` descending (newest first, so the cap drops the oldest facts); each line shows
`(since <valid_from date>[, previously: <old object>]) <statement>`. A separate two-line "Things
you have said about yourself" block covers `subject="self"` facts when any exist. Facts are
exposed via `Character.facts`, `Character.export()["facts"]` / `engine.import_character()`, the
MCP `get_facts(character_id, subject=None, as_of=None)` tool and `get_stats().facts_current`, and
`GET /api/facts/{character_id}?subject=&as_of=`.

**Known limitation**: `import_character()` restores facts with `memory_id=None` — memory linkage
is not preserved, so reinforcing or superseding an imported fact does not update any memory row
(there is no linked memory to update).

### Belief Revision

Every Core/Bedrock memory carries a certainty score (0.0–1.0):

- **reinforce(memory_id)**: certainty += 0.15 (capped at 1.0)
- **contradict(old_id, new_content, source)**: old.certainty = 0, old.status = "contradicted", new memory created
- **invalidate(memory_id)**: removed from retrieval, preserved in archive
- **detect_contradictions(new_memory)**: pre-flight check before storage

Contradicted memories remain queryable for character growth ("I used to think X, but now I know Y").

### Consolidation Engine

Triggered when Buffer exceeds threshold (default: 100 entries):

1. Cluster related Buffer entries by semantic similarity
2. For each cluster, LLM generates a consolidated summary
3. Summary stored as Core memory with source_refs pointing to originals
4. Original Buffer entries archived (not deleted)
5. Importance scores aggregated (max of cluster)

### Session Management

Each conversation session produces:
- **Session summary**: key events, emotional beats, relationship changes
- **Memory extractions**: specific facts, opinions, commitments mentioned
- **Relationship updates**: dimension changes based on interaction quality
- **Growth events**: moments where soft constraints may shift

### Unified Assessment (Tier 1)

`persona/assessment.py`'s `TurnAssessor.assess(...)` replaces up to four
per-turn LLM calls (emotion, relationship, story beat, fact extraction) with
one `generate_json_robust` call returning
`{"emotion": {...}, "relationship": {...}, "beat": {...}|null, "facts": [...]}`.
Parsing is factored into pure functions that each engine's standalone method
now delegates to — `EmotionEngine.parse_assessment()`, `ArcTracker.parse_beat()`,
`Character._parse_relationship_deltas()`, `Character._parse_facts()` — so the
single-call and per-engine paths share exactly the same validation logic.

Config `character.unified_assessment` (default `true`,
`WOVEN_IMPRINT_UNIFIED_ASSESSMENT`) selects `Character._run_bookkeeping()`
(one call) over the legacy `_run_subsystems_sequential`/`_run_subsystems_parallel`
per-engine path, which is kept for A/B and is byte-identical to today's
behavior when the flag is `false`. `want_facts` still follows
`fact_extraction_interval`; beat detection still follows the every-2nd-turn
rule — the prompt tells the model which sections to emit, and absent
sections parse to `None`/`[]`. `health()` gained one new `"assessment"`
counter; per-section failures still count under their old keys. Consistency
checking remains a separate hot-path call (`character.enforce_consistency`,
default on) — see the [Performance](../README.md#performance) note in the
README.

### Offline Maintenance (Phase B)

Chat writes cheaply; a batch runner digests. `MaintenanceRunner`
(`maintenance.py`) owns all heavy multi-pass LLM work as budgeted,
idempotent, per-character jobs, callable headless via
`Engine.run_maintenance()` / `woven-imprint maintain`:

```
consolidate → buffer_hygiene → score_importance → dedup → reinforce
            → contradictions → reflect → evolve → callbacks
```

A `Budget` object caps LLM calls per run and is shared across jobs (and
across characters in a multi-character run). Jobs that would exceed it skip
with `budget exhausted`; a failing job is caught and reported, never
aborting the run. Non-LLM jobs (hygiene, dedup, reinforce) always complete.
This shape maps directly onto mobile schedulers (WorkManager: chunked
budgeted runs while charging + idle).

### Callbacks (Phase B)

The batch-generate / instant-read split, end to end:

```
chat()/observe()          nightly runner               session start
  writes buffer/core  ─→  callbacks job:           ─→  get_callbacks():
  memories cheaply        gather high-importance       plain DB read,
                          core observations,           salience-ordered,
                          key moments, arc beats       zero LLM calls
                          → 1 LLM call →
                          paraphrased hooks in
                          the callbacks table
                                                       compose_initiation():
                                                       weave top hook into a
                                                       character-first message,
                                                       mark it consumed
```

Invariants:
- Hooks are **paraphrased, in-character** — never verbatim memory quotes.
- The ready queue is capped; refreshes expire the lowest-salience overflow.
- `compose_initiation()` consumes its hook — scarcity by design, no repeats.
- `end_session()` also refreshes callbacks (config-gated,
  `maintenance.callbacks_refresh_on_session_end`), so the next session has
  fresh hooks even without a nightly run.

### Prompt Registry

`woven_imprint.prompts` holds every chat-path LLM prompt as a
`PromptSpec(id, version, system, user, expects)` in one `PROMPTS` dict,
rendered via `render(id, **kwargs) -> list[dict]` (`str.format_map`, so
literal `{`/`}` in a template — mostly JSON examples — is escaped as
`{{`/`}}`). 18 ids are covered: fact extraction, relationship (turn +
event), reflect, session summary, emotion (turn + event), consistency check
+ retry reminder, growth, arc beat, consolidation summary, callbacks hooks
+ compose, maintenance importance + contradiction, context compression, and
the unified turn assessment. Templates are byte-identical to the
pre-refactor inline strings (verified by snapshot tests that render fixed
kwargs and compare to captured fixtures). The `woven-imprint prompts` CLI
subcommand lists every id and version — cheap and useful when tuning
prompts against a smaller model. `character.py::_build_context`'s "Today
is …" header and memories preamble are intentionally NOT in the registry —
they stay literal strings at the call site.

### SillyTavern Interchange

**Import** (`migrate/parsers.py::parse_tavernai_card`, `migrate/importer.py`): a TavernAI/
SillyTavern V2 card (`chara_card_v2`), JSON or PNG (`tEXt` chunk — V3 `ccv3` preferred over V2
`chara` when both are present), maps `scenario`/`tags`/`alternate_greetings` (+ `first_mes`)
into soft persona traits, `system_prompt` into `hard.hard_constraints`, and `creator_notes` into
`hard.creator_notes`. Each **enabled** `character_book` (lorebook) entry becomes a memory:
`constant` entries import as pinned `bedrock` memories (`metadata.pinned=True`) — the closest
equivalent to SillyTavern's always-on lore — and keyed entries import as ordinary `core`
memories, both with content prefixed `[Lore: key1, key2, ...] `. `keys` is coerced from
string/comma-string/list-with-stray-entries into a clean `list[str]`; missing `enabled`
defaults to `True` (hand-edited cards sometimes omit it).

**Export** (`Character.export_card(core_limit=20) -> dict`): builds a `chara_card_v2` card —
`name`, `description` (backstory), `personality`, `scenario`, `first_mes`/`alternate_greetings`
(from persona greetings), `system_prompt` (`hard.hard_constraints`), `creator_notes`, `tags` —
with a generated `character_book` whose entries are, in order: every pinned memory
(`constant: true`), every current `subject="user"` fact (`statement` as content, `keys` built
from the fact's object + predicate words), then the top `core_limit` core memories by
importance (`constant: false`, excluding anything already pinned or fact-linked). CLI
`woven-imprint export-card <name-or-id> [-o card.json]`; HTTP `GET /api/characters/{id}/card`.

**Round trip is lossy, by design of the card format**: re-importing an exported card wraps
every lorebook entry's content as `[Lore: keys] content`, so exported memory text comes back
prefixed rather than verbatim; facts export as plain-text lorebook entries and come back as
memories (pinned bedrock if they were exported `constant`), not as `FactStore` rows — the card
format has no subject/predicate/object shape, so structured fact history and the fact ↔ memory
link do not survive an export → import cycle.

## Storage

### SQLite Schema (local-first default)

```sql
-- Character definition
CREATE TABLE characters (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    persona JSON NOT NULL,        -- hard + soft + temporal constraints
    birthdate DATE,               -- NULL if age is static/unknown
    state JSON DEFAULT '{}',      -- current emotional state, goals
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Memory entries
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    character_id TEXT NOT NULL REFERENCES characters(id),
    tier TEXT NOT NULL CHECK(tier IN ('buffer', 'core', 'bedrock')),
    content TEXT NOT NULL,
    embedding BLOB,               -- float32 vector, serialized
    importance REAL DEFAULT 0.5,
    certainty REAL DEFAULT 1.0,
    status TEXT DEFAULT 'active' CHECK(status IN ('active', 'contradicted', 'archived')),
    source_refs JSON DEFAULT '[]', -- IDs of source memories (for consolidated)
    session_id TEXT,
    role TEXT,                     -- 'user', 'character', 'system', 'observation'
    metadata JSON DEFAULT '{}',
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    accessed_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Relationships
CREATE TABLE relationships (
    id TEXT PRIMARY KEY,
    character_id TEXT NOT NULL REFERENCES characters(id),
    target_id TEXT NOT NULL,       -- user ID or other character ID
    dimensions JSON NOT NULL,      -- {trust, affection, respect, familiarity, tension}
    power_balance REAL DEFAULT 0.0,
    type TEXT DEFAULT 'stranger',
    trajectory TEXT DEFAULT 'stable',
    key_moments JSON DEFAULT '[]',
    formed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_interaction DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Sessions
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    character_id TEXT NOT NULL REFERENCES characters(id),
    summary TEXT,
    started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    ended_at DATETIME
);

-- Full-text search index
CREATE VIRTUAL TABLE memories_fts USING fts5(
    content, character_id UNINDEXED, tier UNINDEXED
);
```

Later migrations add `sessions.alias` (v2), the `session_turns` durable
conversation buffer (v3), the `callbacks` + `meta` tables (v4), and (v5) the
bi-temporal `facts` table plus `relationships.state` — see
[Facts](#facts-bi-temporal) and [Relationship Model](#relationship-model)
above. The authoritative, foreign-writer-facing specification of every
table, the embedding BLOB format, timestamp rules, FTS5 triggers, and the
migration protocol is **[SCHEMA.md](SCHEMA.md)**.

### Vector Index

For semantic search, embeddings stored in `memories.embedding` column.
At query time, compute cosine similarity in Python (for <10K memories per character, this is fast enough).
For production scale: optional Qdrant/Chroma backend via pluggable VectorStore interface.

## Module Structure

```
woven_imprint/
├── __init__.py           # Public API: Engine, Character
├── engine.py             # Engine class — entry point
├── character.py          # Character class — chat, reflect, export
├── providers.py          # Factory: create_llm(), create_embedding()
├── config.py             # Centralized configuration (60+ settings)
├── maintenance.py        # MaintenanceRunner + Budget — offline batch jobs
├── callbacks.py          # CallbackEngine — hooks + proactive initiation
├── memory/
│   ├── __init__.py
│   ├── store.py          # MemoryStore — CRUD operations
│   ├── retrieval.py      # Multi-strategy retrieval + RRF
│   ├── consolidation.py  # Buffer → Core compression
│   ├── belief.py         # Belief revision system
│   └── facts.py          # FactStore — bi-temporal structured facts
├── persona/
│   ├── __init__.py
│   ├── model.py          # PersonaModel — constraint management
│   └── consistency.py    # NLI-inspired consistency checker
├── relationship/
│   ├── __init__.py
│   └── model.py          # RelationshipModel — dimensional tracking
├── narrative/
│   ├── __init__.py
│   └── arc.py            # Narrative arc tracking + story beats
├── llm/
│   ├── __init__.py
│   ├── base.py           # Abstract LLM interface
│   ├── openai_llm.py     # OpenAI / compatible API
│   ├── anthropic_llm.py  # Anthropic Claude
│   └── ollama.py         # Ollama local models
├── embedding/
│   ├── __init__.py
│   ├── base.py           # Abstract embedding interface
│   ├── openai_embedding.py  # OpenAI embeddings
│   └── ollama.py         # Ollama embeddings (nomic-embed-text)
├── migrate/
│   ├── __init__.py
│   ├── parsers.py        # Format-specific parsers (ChatGPT, TavernAI, Claude, etc.)
│   └── importer.py       # CharacterImporter — chunked analysis, profile synthesis
├── server/
│   ├── __init__.py
│   └── api.py            # OpenAI-compatible HTTP proxy
├── storage/
│   ├── __init__.py
│   └── sqlite.py         # SQLite backend (default)
└── utils/
    ├── __init__.py
    ├── rrf.py            # Reciprocal Rank Fusion
    └── text.py           # Text processing utilities
```

`server/api.py` is the OpenAI-compatible endpoint behind `woven-imprint serve`; `server/demo.py`
is the demo UI behind `woven-imprint demo`. Both stay.

## API Surface

```python
# Core API
engine = Engine(db_path="characters.db", llm=OllamaLLM("qwen3"), embedding=OllamaEmbedding())

character = engine.create_character(name, persona, constraints)
character = engine.load_character(character_id)

response = character.chat(message, user_id=None)
character.reflect()  # generate higher-level reflections
character.consolidate()  # compress buffer → core

memories = character.recall(query, limit=10)
relationship = character.relationships.get(target_id)

character.facts.current("user", "lives_in")       # current belief
character.facts.as_of("2026-05-20", "user", ...)   # what was true then
character.facts.history("user", "lives_in")        # every version

character.memory.pin(memory_id)                    # always in the prompt, never dropped
character.memory.edit(memory_id, content="...")    # re-embeds on content change
character.memory.delete(memory_id)                 # also retracts any linked fact
character.facts.edit(fact_id, object="...")
character.facts.retract(fact_id)                    # no longer current; kept in history

character.export(path)  # full character state as JSON, including facts
character = engine.import_character(path)
card = character.export_card()  # SillyTavern V2 card + lorebook (pinned/facts/core)

# MCP Server
# Exposes: chat, recall, reflect, list_characters, get_relationship, get_facts, get_stats,
# edit_memory, delete_memory, pin_memory, list_pinned, edit_fact, retract_fact
```
