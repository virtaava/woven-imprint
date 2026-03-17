"""Multi-strategy memory retrieval with Reciprocal Rank Fusion."""

from __future__ import annotations

import math
from datetime import datetime, timezone

from ..embedding.base import EmbeddingProvider
from ..storage.sqlite import SQLiteStorage
from ..utils.rrf import reciprocal_rank_fusion


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# Decay rates per tier — bedrock and core decay much slower than buffer
_DECAY_RATES = {
    "bedrock": 0.9999,  # half-life ~290 days — nearly permanent
    "core": 0.999,  # half-life ~29 days — fades over months
    "buffer": 0.995,  # half-life ~5.8 days — fades in a week
}

# Importance floor per tier — minimum effective importance
_TIER_IMPORTANCE_BOOST = {
    "bedrock": 0.35,  # bedrock always gets a strong boost
    "core": 0.2,  # core gets a solid boost (session summaries, extracted facts)
    "buffer": 0.0,  # buffer gets no boost (ephemeral by design)
}


def _recency_score(accessed_at: str, tier: str = "buffer") -> float:
    """Exponential decay based on hours since last access.

    Different tiers decay at different rates:
    - bedrock: nearly permanent (you don't forget who you are)
    - core: slow decay (consolidated memories persist for months)
    - buffer: fast decay (raw observations fade in days)
    """
    decay_rate = _DECAY_RATES.get(tier, 0.995)
    try:
        accessed = datetime.fromisoformat(accessed_at.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return 0.5
    if accessed.tzinfo is None:
        accessed = accessed.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    hours = max(0, (now - accessed).total_seconds() / 3600)
    return decay_rate**hours


class MemoryRetriever:
    """Retrieve memories using multi-strategy RRF.

    Retrieval strategies:
    1. Semantic similarity (embedding cosine distance)
    2. Keyword match (FTS5 BM25)
    3. Recency (tier-aware exponential decay)
    4. Importance (score × certainty + tier boost)
    5. Relationship boost (if target specified)

    Tier-aware scoring:
    - Bedrock memories decay extremely slowly and get importance boosts
    - Core memories decay slowly
    - Buffer memories decay quickly (ephemeral by design)
    """

    def __init__(self, storage: SQLiteStorage, embedder: EmbeddingProvider, character_id: str):
        self.storage = storage
        self.embedder = embedder
        self.character_id = character_id

    def retrieve(
        self, query: str, limit: int = 10, relationship_target: str | None = None
    ) -> list[dict]:
        """Retrieve the most relevant memories using RRF across strategies."""
        # Two-phase retrieval (C5 fix):
        # Phase 1: Load recent memories per tier (recency window)
        bedrock = self.storage.get_memories(self.character_id, tier="bedrock", limit=50)
        core = self.storage.get_memories(self.character_id, tier="core", limit=200)
        buffer = self.storage.get_memories(self.character_id, tier="buffer", limit=100)

        # Phase 2: FTS pre-filter finds relevant OLD memories beyond the recency window
        # This ensures a memory from months ago can be found if the query matches
        try:
            fts_candidates = self.storage.fts_search(self.character_id, query, limit=50)
        except Exception:
            fts_candidates = []

        # Merge — deduplicate by ID
        memory_map: dict[str, dict] = {}
        for m in bedrock + core + buffer + fts_candidates:
            if m["id"] not in memory_map:
                memory_map[m["id"]] = m

        all_memories = list(memory_map.values())
        if not all_memories:
            return []

        # Strategy 1: Semantic ranking
        query_embedding = self.embedder.embed(query)
        semantic_scores = []
        for m in all_memories:
            if m.get("embedding"):
                sim = _cosine_similarity(query_embedding, m["embedding"])
                semantic_scores.append((m["id"], sim))
        semantic_scores.sort(key=lambda x: x[1], reverse=True)
        semantic_ranked = [mid for mid, _ in semantic_scores]

        # Strategy 2: Keyword ranking (BM25 via FTS5) — uses pre-fetched candidates
        keyword_ranked = [m["id"] for m in fts_candidates]

        # Strategy 3: Tier-aware recency ranking
        recency_scores = [
            (m["id"], _recency_score(m.get("accessed_at", ""), m.get("tier", "buffer")))
            for m in all_memories
        ]
        recency_scores.sort(key=lambda x: x[1], reverse=True)
        recency_ranked = [mid for mid, _ in recency_scores]

        # Strategy 4: Importance with tier boost
        importance_scores = []
        for m in all_memories:
            base = m.get("importance", 0.5) * m.get("certainty", 1.0)
            boost = _TIER_IMPORTANCE_BOOST.get(m.get("tier", "buffer"), 0.0)
            importance_scores.append((m["id"], base + boost))
        importance_scores.sort(key=lambda x: x[1], reverse=True)
        importance_ranked = [mid for mid, _ in importance_scores]

        # Strategy 5: Relationship boost (if target specified)
        ranked_lists = [semantic_ranked, keyword_ranked, recency_ranked, importance_ranked]

        if relationship_target:
            rel_scores = []
            target_lower = relationship_target.lower()
            for m in all_memories:
                content_lower = m["content"].lower()
                meta = m.get("metadata", {})
                involves_target = (
                    target_lower in content_lower or meta.get("target_id") == relationship_target
                )
                rel_scores.append((m["id"], 1.0 if involves_target else 0.0))
            rel_scores.sort(key=lambda x: x[1], reverse=True)
            ranked_lists.append([mid for mid, _ in rel_scores])

        # Fuse with RRF
        fused = reciprocal_rank_fusion(ranked_lists)

        # Return top-N memories
        results = []
        touch_ids = []
        for mem_id, score in fused[:limit]:
            if mem_id in memory_map:
                mem = memory_map[mem_id].copy()
                mem["_retrieval_score"] = score
                results.append(mem)
                touch_ids.append(mem_id)

        # Batch-update accessed_at (single transaction, no FTS reindex)
        self.storage.touch_memories_batch(touch_ids)

        return results
