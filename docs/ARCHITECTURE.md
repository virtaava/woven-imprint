# Woven Imprint — Architecture

## Overview

Woven Imprint is persistent character infrastructure. It sits between the application
and the LLM, providing memory management, persona enforcement, relationship tracking,
and consistency verification.

```
Application ─→ Woven Imprint Engine ─→ LLM Provider
                    │
                    ├── Memory Store (SQLite/Qdrant)
                    ├── Persona Model
                    ├── Relationship Graph
                    └── Consistency Checker
```

## Core Concepts

### Character
A persistent AI personality with:
- **Persona**: immutable identity (name, backstory, personality traits, speaking style)
- **Memory**: accumulated experiences organized in three tiers
- **Relationships**: tracked connections with users and other characters
- **State**: current emotional state, active goals, recent context

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

### Persona Model

Three constraint levels:

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
5. If rejection: regenerate with explicit constraint reminder
6. Max 2 regeneration attempts, then return best-scoring response

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
Change magnitude bounded to ±0.15 per interaction. Trajectory smoothing over 5-interaction window.

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
│   ├── model.py          # RelationshipModel — dimensional tracking
│   └── graph.py          # Multi-character relationship graph
├── llm/
│   ├── __init__.py
│   ├── base.py           # Abstract LLM interface
│   ├── openai.py         # OpenAI / compatible API
│   ├── anthropic.py      # Anthropic Claude
│   ├── ollama.py         # Ollama local models
│   └── vllm.py           # vLLM local models
├── embedding/
│   ├── __init__.py
│   ├── base.py           # Abstract embedding interface
│   ├── openai.py         # OpenAI embeddings
│   ├── ollama.py         # Ollama embeddings (nomic-embed-text)
│   └── sentence_transformers.py
├── storage/
│   ├── __init__.py
│   ├── sqlite.py         # SQLite backend (default)
│   └── qdrant.py         # Qdrant backend (optional)
└── utils/
    ├── __init__.py
    ├── rrf.py            # Reciprocal Rank Fusion
    └── text.py           # Text processing utilities
```

## API Surface

```python
# Core API
engine = Engine(db_path="characters.db", llm=OllamaLLM("qwen3"), embedding=OllamaEmbedding())

character = engine.create_character(name, persona, constraints)
character = engine.load_character(character_id)

response = character.chat(message, user_id=None, session_id=None)
character.reflect()  # generate higher-level reflections
character.consolidate()  # compress buffer → core

memories = character.recall(query, limit=10)
relationship = character.get_relationship(target_id)

character.export(path)  # full character state as JSON
character = engine.import_character(path)

# MCP Server
# Exposes: chat, recall, reflect, list_characters, get_relationship
```
