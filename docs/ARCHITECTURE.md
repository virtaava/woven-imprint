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
    type: friend | rival | mentor | protege | love_interest | family | colleague | stranger
    trajectory: warming | cooling | stable | volatile
    key_moments: Memory[]          # pivotal interaction memories
    formed_at: datetime
    last_interaction: datetime
```

Updates are LLM-assessed from conversation content, not formula-driven.
Change magnitude bounded to ±0.15 per interaction. Trajectory (warming/cooling/stable/volatile) computed from current interaction deltas.

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
conversation buffer (v3), and the `callbacks` + `meta` tables (v4). The
authoritative, foreign-writer-facing specification of every table, the
embedding BLOB format, timestamp rules, FTS5 triggers, and the migration
protocol is **[SCHEMA.md](SCHEMA.md)**.

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
│   └── belief.py         # Belief revision system
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

character.export(path)  # full character state as JSON
character = engine.import_character(path)

# MCP Server
# Exposes: chat, recall, reflect, list_characters, get_relationship
```
