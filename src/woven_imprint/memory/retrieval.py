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


def _recency_score(accessed_at: str, decay_rate: float = 0.995) -> float:
    """Exponential decay based on hours since last access."""
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
    """Retrieve memories using multi-strategy RRF."""

    def __init__(self, storage: SQLiteStorage, embedder: EmbeddingProvider, character_id: str):
        self.storage = storage
        self.embedder = embedder
        self.character_id = character_id

    def retrieve(
        self, query: str, limit: int = 10, relationship_target: str | None = None
    ) -> list[dict]:
        """Retrieve the most relevant memories using RRF across strategies.

        Strategies:
        1. Semantic similarity (embedding cosine distance)
        2. Keyword match (FTS5 BM25)
        3. Recency (exponential decay from last access)
        4. Importance (stored importance score)
        5. Relationship boost (if target specified, boost memories involving them)
        """
        # Get all active memories with embeddings for semantic search
        all_memories = self.storage.get_memories(self.character_id, status="active")
        if not all_memories:
            return []

        memory_map = {m["id"]: m for m in all_memories}

        # Strategy 1: Semantic ranking
        query_embedding = self.embedder.embed(query)
        semantic_scores = []
        for m in all_memories:
            if m.get("embedding"):
                sim = _cosine_similarity(query_embedding, m["embedding"])
                semantic_scores.append((m["id"], sim))
        semantic_scores.sort(key=lambda x: x[1], reverse=True)
        semantic_ranked = [mid for mid, _ in semantic_scores]

        # Strategy 2: Keyword ranking (BM25 via FTS5)
        try:
            fts_results = self.storage.fts_search(self.character_id, query, limit=100)
            keyword_ranked = [m["id"] for m in fts_results]
        except Exception:
            keyword_ranked = []

        # Strategy 3: Recency ranking
        recency_scores = [(m["id"], _recency_score(m.get("accessed_at", ""))) for m in all_memories]
        recency_scores.sort(key=lambda x: x[1], reverse=True)
        recency_ranked = [mid for mid, _ in recency_scores]

        # Strategy 4: Importance ranking
        importance_scores = [
            (m["id"], m.get("importance", 0.5) * m.get("certainty", 1.0)) for m in all_memories
        ]
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
        for mem_id, score in fused[:limit]:
            if mem_id in memory_map:
                mem = memory_map[mem_id].copy()
                mem["_retrieval_score"] = score
                # Touch accessed_at for recency tracking
                self.storage.touch_memory(mem_id)
                results.append(mem)

        return results
