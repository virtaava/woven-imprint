# Woven Imprint — Research Synthesis

## Date: 2026-03-16
## Papers reviewed: 25 (from 471 collected across 6 research blocks)

---

## 1. Memory Architecture Patterns (What the Field Has Converged On)

### 1.1 The Three-Tier Memory Model

Every serious system converges on a variant of human cognitive memory:

| Tier | Stanford (2023) | Mem0 (2025) | Engram (2026) | Our Name |
|------|----------------|-------------|---------------|----------|
| Immediate | Observation stream | Working memory | Working tier | **Buffer** |
| Processed | Reflections | Factual + Episodic | (auto-consolidated) | **Core** |
| Deep | Plan archive | Semantic memory | Long-term tier | **Bedrock** |

**Key insight**: Nobody has nailed the transition mechanics. Stanford uses a fixed reflection threshold (importance score sum > N). Engram auto-consolidates at 100 working memories. Mem0 doesn't document their consolidation triggers. **This is our opportunity — adaptive consolidation based on emotional salience, narrative importance, and relationship context.**

### 1.2 Retrieval: The Retrieval Function Is Everything

**Stanford's formula** (the gold standard, 1176 citations):
```
score(memory) = α·recency(memory) + β·importance(memory) + γ·relevance(memory, query)
```
- Recency: exponential decay based on time since last access
- Importance: LLM-assigned score (1-10) at storage time
- Relevance: cosine similarity of embedding to current context

**Engram's improvement**: Reciprocal Rank Fusion across three independent rankers:
- Semantic embeddings (nomic-embed-text)
- BM25 keyword search
- Temporal decay

**Mem0's approach**: Dual vector + graph retrieval with reranking model. Sub-50ms.

**Our design decision**: We should combine Stanford's weighted scoring with Engram's multi-strategy RRF, PLUS add a **relationship-weighted** dimension that Stanford completely missed. A memory involving a close relationship partner should rank higher than one involving a stranger.

### 1.3 Memory Conflict Resolution

**Engram** is the only system with explicit belief revision:
- `reinforce(id)` — increases certainty by 0.15
- `contradict(oldId, newContent)` — marks old as contradicted, stores new
- `invalidate(id)` — removes from recall without deletion
- `detectContradictions()` — pre-flight check before storing

**Nobody else handles this properly.** Mem0 mentions "conflict resolution" but doesn't document it. Stanford doesn't address it at all — their agents can hold contradictory beliefs indefinitely.

**Our design decision**: Adopt Engram's belief-revision model as a foundation, but extend it with:
- **Narrative coherence checking** — contradictions in character backstory vs. opinions vs. relationship facts have different severities
- **Graceful contradiction** — characters can genuinely change their mind (not just overwrite)
- **Source attribution** — every belief tracks where it came from (which conversation, who said it)

---

## 2. Persona Consistency (The Hardest Problem)

### 2.1 The Consistency-Diversity Tradeoff

Every paper acknowledges this: strict persona adherence → repetitive responses. Diverse generation → persona drift.

**Post Persona Alignment (PPA)** has the most elegant solution:
1. Generate response based ONLY on dialogue context (natural, diverse)
2. Use the generated response as a query to retrieve relevant persona facts
3. Refine the response to align with retrieved persona

This is brilliant because it avoids the "persona tunnel vision" problem where characters only talk about their defined traits.

**NLI-based approach**: Trains a reward model using NLI (Natural Language Inference) to score whether a response contradicts persona facts. Used as RL reward during training.

**MoCoRP**: Explicitly extracts NLI relations between persona sentences and responses, making the model aware of which persona facts are being invoked.

**Our design decision**: Implement a **three-layer consistency system**:
1. **Hard constraints** — immutable identity (name, core backstory) — NEVER contradict
2. **Temporal facts** — change on schedule/events (age from birthdate, location, appearance) — auto-resolved, not static
3. **Soft constraints** — personality traits, speaking patterns — consistent but can evolve with justification
4. **Emergent layer** — new opinions, reactions, relationships — formed through interaction

Use NLI-style scoring at inference time (not training time, since we're LLM-agnostic) as a post-generation check. If a response contradicts a hard constraint, regenerate. If it contradicts a soft constraint, flag but allow if character growth context is present.

### 2.2 Multi-Session Persona Drift

PPA's key finding: persona consistency degrades significantly across sessions. Their "response-as-query" approach helps because it retrieves the MOST RELEVANT persona facts for each specific response, not a fixed persona prompt.

**Our design decision**: Each conversation session produces a **session summary** that captures:
- Key events and emotional beats
- Relationship changes
- New beliefs or opinions formed
- Character growth moments

These summaries become part of Core memory and inform future sessions. The character doesn't just remember WHAT happened — they remember HOW they felt about it.

---

## 3. Relationship Modeling (The Gap Nobody Fills)

### 3.1 Current State of the Art

**Stanford Generative Agents**: Relationships are implicit in the memory stream. No explicit relationship tracking. Agent A remembers interactions with Agent B, but there's no "relationship score" or "relationship type."

**Engram**: Has entity-relationship graph, but it's about FACTS (entity → relationship → entity), not emotional/social relationships.

**Mem0**: Graph memory captures entity relationships but is fact-oriented, not social.

**Social NPCs (Skyrim CiF-CK)**: The closest to what we need — explicit social architecture with reputation and relationship states. But it's rule-based, not learned.

**NOBODY has**: Persistent, evolving, emotionally-grounded character relationships that develop over time through actual interactions.

### 3.2 Our Relationship Model (Unique Differentiator)

```
Relationship {
    character_a: CharID
    character_b: CharID

    // Emotional dimensions (each -1.0 to 1.0)
    trust: float        // do they trust each other?
    affection: float    // warmth, liking
    respect: float      // admiration for competence
    familiarity: float  // how well they know each other (0 = strangers, 1 = intimate)
    tension: float      // unresolved conflict, suspicion

    // Dynamics
    power_balance: float  // -1 = B dominates, 0 = equal, 1 = A dominates
    shared_history: Memory[]  // key moments

    // Meta
    relationship_type: enum  // friend, rival, mentor, love_interest, etc.
    formed_at: timestamp
    last_interaction: timestamp

    // Evolution
    trajectory: "warming" | "cooling" | "stable" | "volatile"
}
```

**Each interaction updates relationship dimensions based on what happened.** Not arbitrary scores — LLM-assessed based on the actual conversation content.

---

## 4. Technical Architecture Decisions

### 4.1 Storage: Local-First, Like Engram

Engram proves SQLite is sufficient for consumer-grade character memory. No cloud dependency. Self-hostable infrastructure is what developers and studios want.

**Decision**: SQLite primary, with optional PostgreSQL/Qdrant for production deployments. All state in one portable file per character.

### 4.2 Embedding: Pluggable, Not Locked

Mem0 defaults to OpenAI (1536-dim). Engram uses nomic-embed-text (768-dim) via Ollama.

**Decision**: Abstract the embedding layer. Support:
- OpenAI text-embedding-3-small (default cloud)
- nomic-embed-text via Ollama (default local)
- Any sentence-transformers model
- Custom models via API endpoint

### 4.3 LLM: Completely Agnostic

Unlike Mem0 (OpenAI-centric) or Engram (Ollama-centric), we must work with ANY LLM.

**Decision**: All LLM calls go through a `CharacterLLM` interface:
- `generate(prompt, persona_context, memory_context) → response`
- `extract_memories(conversation) → Memory[]`
- `assess_consistency(response, persona) → ConsistencyReport`
- `consolidate(memories) → Memory[]`

Implementations for OpenAI, Anthropic, Ollama, vLLM, llama.cpp, etc.

### 4.4 API Design: Developer-First

```python
from woven_imprint import Character, Engine

engine = Engine(db_path="./characters.db")

# Create a character
alice = engine.create_character(
    name="Alice",
    birthdate="1998-03-15",  # age derived, auto-increments on birthday
    persona={
        "occupation": "detective",
        "personality": "sharp-witted, skeptical, secretly lonely",
        "backstory": "Left the force after her partner's death...",
        "speaking_style": "clipped sentences, dark humor, avoids emotional topics",
    },
    constraints={
        "hard": ["never reveals partner's name", "always carries a notebook"],
        "soft": ["distrusts authority", "drinks black coffee"],
    }
)

# Conversation with memory
response = alice.chat("Hey Alice, how's the case going?")
# → "Case is cold. Coffee's colder. What do you want?"

# Memory persists across sessions
response = alice.chat("Remember when we talked about the harbor case?")
# → retrieves relevant memories, responds in context

# Relationship tracking
alice.relationships["user_bob"]
# → Relationship(trust=0.6, affection=0.3, familiarity=0.7, ...)

# Character introspection
alice.reflect()
# → generates higher-level reflections from accumulated memories

# Export character (portability)
alice.export("alice_backup.json")
```

### 4.5 What Makes This Acquisition-Worthy

| Feature | Character.AI | Inworld | Convai | Mem0 | Engram | **Woven Imprint** |
|---------|-------------|---------|--------|------|--------|--------------------|
| Persistent memory | ❌ session-only | ⚠️ limited | ⚠️ limited | ✅ | ✅ | ✅ |
| Character persona | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ |
| Relationship tracking | ❌ | ⚠️ basic | ❌ | ❌ | ❌ | ✅ |
| Belief revision | ❌ | ❌ | ❌ | ⚠️ | ✅ | ✅ |
| Character growth | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| Consistency checking | ❌ | ⚠️ | ❌ | ❌ | ❌ | ✅ (NLI) |
| Local-first | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ |
| LLM-agnostic | ❌ | ❌ | ❌ | ⚠️ | ⚠️ Ollama | ✅ |
| Developer API | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Open source | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ (MIT) |

**The gap**: Nobody combines persistent memory + character persona + relationship tracking + belief revision + consistency checking in a single, LLM-agnostic, local-first package with a clean developer API.

---

## 5. Architecture Risks and Mitigations

### Risk 1: Memory bloat
Long-running characters accumulate thousands of memories. Retrieval slows.
**Mitigation**: Engram-style consolidation + importance-based pruning. Bedrock memories are compressed summaries. Buffer auto-consolidates.

### Risk 2: Persona drift over long conversations
Character slowly loses its personality as context fills with interaction history.
**Mitigation**: PPA-style "persona re-anchoring" — every N turns, re-inject core persona traits. NLI consistency scoring flags drift before it compounds.

### Risk 3: Hallucinated memories
LLM invents memories that never happened during memory extraction.
**Mitigation**: Every memory traces back to a specific conversation turn with exact quotes. Verification pass before storage.

### Risk 4: Relationship score gaming
Simple numeric dimensions can produce unrealistic relationship states.
**Mitigation**: Relationship changes are LLM-assessed from conversation content, not formula-driven. Change magnitude is bounded per interaction. Trajectory smoothing prevents whiplash.

### Risk 5: LLM quality floor
The system is only as good as the underlying LLM.
**Mitigation**: The memory/persona/consistency layers ADD value on top of any LLM. Even a weak LLM benefits from proper memory retrieval and consistency checking. The architecture is the product, not the model.

---

## 6. Implementation Priority

### Phase 1: Core Memory Engine (MVP)
1. Memory storage (SQLite) with embedding index
2. Three-tier memory (buffer → core → bedrock)
3. Multi-strategy retrieval (semantic + keyword + recency via RRF)
4. Basic persona enforcement (hard constraints)
5. Simple API: `create_character()`, `chat()`, `export()`

### Phase 2: Character Intelligence
6. Belief revision system (reinforce/contradict/invalidate)
7. Relationship model with dimensional tracking
8. Reflection generation (Stanford-style)
9. Session summaries and cross-session continuity
10. NLI-based consistency scoring

### Phase 3: Production Ready
11. Consolidation engine (working → long-term compression)
12. Pluggable LLM backends (OpenAI, Anthropic, Ollama, vLLM)
13. Pluggable embedding backends
14. Character import/export (JSON, YAML)
15. MCP server for IDE integration

### Phase 4: Differentiation
16. Character growth model (soft constraints evolve)
17. Multi-character interaction (relationship graph)
18. Emotional state tracking (mood affects responses)
19. Narrative arc awareness (story beats, tension curves)
20. Performance benchmarks and eval suite
