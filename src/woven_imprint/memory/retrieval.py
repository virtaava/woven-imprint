"""Multi-strategy memory retrieval with Reciprocal Rank Fusion."""

from __future__ import annotations

import logging
import math
import re

from .. import clock
from ..embedding.base import EmbeddingProvider
from ..storage.sqlite import SQLiteStorage
from ..utils.rrf import reciprocal_rank_fusion

try:  # optional fast path
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None

logger = logging.getLogger(__name__)

# Small stopword set for `_salient_terms` (Tier 3f second-pass expansion) — common
# words that are long enough (>=5 chars) to otherwise pass the "salient" filter but
# carry no real topical signal. Not exhaustive; a heuristic gate, not an NLP pass.
_SALIENT_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "again",
        "along",
        "always",
        "another",
        "around",
        "because",
        "before",
        "being",
        "between",
        "could",
        "doesn't",
        "during",
        "either",
        "enough",
        "every",
        "first",
        "found",
        "great",
        "having",
        "however",
        "little",
        "maybe",
        "might",
        "never",
        "often",
        "other",
        "people",
        "perhaps",
        "place",
        "really",
        "should",
        "since",
        "still",
        "their",
        "there",
        "these",
        "thing",
        "think",
        "those",
        "through",
        "today",
        "under",
        "until",
        "using",
        "where",
        "which",
        "while",
        "whose",
        "would",
        "years",
    }
)


def _salient_terms(text: str, limit: int = 16) -> list[str]:
    """Extract salient terms from memory content for Tier 3f second-pass expansion.

    Pure heuristic, no LLM: capitalized tokens (proper nouns — naive, not
    sentence-position-aware, which is fine here), tokens containing a digit
    (dates/numbers/ids), and lowercase words of length >= 5 not in the small
    `_SALIENT_STOPWORDS` set. Deduped case-insensitively (first occurrence
    wins), capped at `limit` terms. Only `[A-Za-z0-9]` characters ever make it
    into a term, so FTS5 special characters (quotes, `*`, `:`, `-`, ...) can't
    leak into the query built from these terms.
    """
    terms: list[str] = []
    seen: set[str] = set()
    for word in re.findall(r"[A-Za-z0-9]+", text):
        key = word.lower()
        if key in seen:
            continue
        has_digit = any(c.isdigit() for c in word)
        is_capitalized = word[0].isupper()
        is_long_lowercase = not is_capitalized and len(word) >= 5 and key not in _SALIENT_STOPWORDS
        if has_digit or is_capitalized or is_long_lowercase:
            seen.add(key)
            terms.append(word)
            if len(terms) >= limit:
                break
    return terms


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Mismatched dimensions (e.g. a legacy embedding from a different model)
    score 0.0 rather than raising — they simply don't rank via semantics.
    """
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def cosine_matrix(query: list[float], rows: list[list[float]]) -> list[float]:
    """Cosine similarity of `query` against each row. numpy when available, else pure Python.

    Rows whose length differs from `query` (e.g. a legacy embedding from a
    prior model/dimension) score 0.0 instead of crashing the whole batch —
    `np.asarray` on a ragged list of rows raises ValueError, so those rows
    are excluded from the numpy call and filled in as 0.0 afterward.
    """
    if not rows:
        return []
    qlen = len(query)
    if _np is not None:
        same_len_idx = [i for i, r in enumerate(rows) if len(r) == qlen]
        skipped = len(rows) - len(same_len_idx)
        if skipped:
            logger.debug("cosine_matrix: skipping %d ragged-length row(s)", skipped)
        sims = [0.0] * len(rows)
        if same_len_idx:
            q = _np.asarray(query, dtype=_np.float32)
            m = _np.asarray([rows[i] for i in same_len_idx], dtype=_np.float32)
            qn = _np.linalg.norm(q)
            rn = _np.linalg.norm(m, axis=1)
            denom = rn * qn
            with _np.errstate(divide="ignore", invalid="ignore"):
                matched = _np.where(denom > 0, (m @ q) / denom, 0.0)
            for i, val in zip(same_len_idx, matched):
                sims[i] = float(val)
        return sims
    return [_cosine_similarity(query, r) for r in rows]


def _get_decay_rates() -> dict:
    from ..config import get_config

    cfg = get_config().memory
    return {"bedrock": cfg.decay_bedrock, "core": cfg.decay_core, "buffer": cfg.decay_buffer}


def _get_tier_boosts() -> dict:
    from ..config import get_config

    cfg = get_config().memory
    return {
        "bedrock": cfg.tier_boost_bedrock,
        "core": cfg.tier_boost_core,
        "buffer": cfg.tier_boost_buffer,
    }


def _recency_score(memory: dict, tier: str = "buffer") -> float:
    """Exponential decay based on hours since the anchor timestamp.

    Anchor is `created_at` by default (config memory.recency_anchor).
    Anchoring on `accessed_at` makes frequently-retrieved memories
    self-reinforcing — kept only as an opt-in legacy mode.
    """
    from ..config import get_config

    anchor_field = (
        "accessed_at" if get_config().memory.recency_anchor == "accessed" else "created_at"
    )
    decay_rate = _get_decay_rates().get(tier, 0.995)
    raw = memory.get(anchor_field) or memory.get("created_at") or ""
    try:
        anchored = clock.parse_ts(raw)
    except (ValueError, AttributeError):
        return 0.5
    now = clock.now()
    hours = max(0, (now - anchored).total_seconds() / 3600)
    return decay_rate**hours


class MemoryRetriever:
    """Retrieve memories using multi-strategy RRF.

    Retrieval strategies:
    1. Semantic similarity (embedding cosine distance)
    2. Keyword match (FTS5 BM25)
    3. Recency (tier-aware exponential decay)
    4. Importance (score × certainty + tier boost)
    5. Relationship boost (if target specified)

    Strategies are fused via weighted Reciprocal Rank Fusion (configurable
    per-strategy weights and k via MemoryConfig). Tier no longer contributes
    its own ranked list — it only influences decay rate (recency) and the
    importance tier boost.

    Relevance gate (MemoryConfig.relevance_gate, default on): for a non-empty
    query, recency/importance/relationship ranking is confined to memories
    that are semantically similar (similarity > relevance_min_similarity, an
    epsilon floor above float32 matmul noise, top relevance_semantic_topk
    of those) or keyword-matching (FTS) — this narrows, but does not
    eliminate, the case where a large off-topic bedrock/core flood outranks a
    fresh on-topic fact; an off-topic memory with genuine (if weak) positive
    similarity that still lands in the semantic top-k remains eligible and
    can still win on recency/importance. If nothing is relevant at all, the
    gate falls back to scoring every active memory. Semantic and keyword
    ranking are always computed over every candidate. Set
    relevance_gate=False to restore the pre-gate fusion.

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
        # Score all active memories (up to max_candidates, newest first).
        from ..config import get_config

        mem_cfg = get_config().memory
        candidates = self.storage.get_memories(self.character_id, limit=mem_cfg.max_candidates)
        # Memories beyond max_candidates (newest first) are reachable only via FTS.

        # FTS pre-filter finds relevant OLD memories beyond max_candidates
        # This ensures a memory from long ago can be found if the query matches
        try:
            fts_candidates = self.storage.fts_search(self.character_id, query, limit=50)
        except Exception:
            fts_candidates = []

        # Merge — deduplicate by ID
        memory_map: dict[str, dict] = {}
        for m in candidates + fts_candidates:
            if m["id"] not in memory_map:
                memory_map[m["id"]] = m

        all_memories = list(memory_map.values())
        if not all_memories:
            return []

        # Sort by rowid for stable input order (ensures tiebreakers in rankings are deterministic)
        all_memories.sort(key=lambda m: m.get("rowid", 0))

        # Strategy 1: Semantic ranking (skip if query empty)
        semantic_ranked = []
        semantic_scores: list[tuple[str, float]] = []
        if query.strip():
            query_embedding = self.embedder.embed(query)
            embedded = [m for m in all_memories if m.get("embedding")]
            sims = cosine_matrix(query_embedding, [m["embedding"] for m in embedded])
            semantic_scores = list(zip((m["id"] for m in embedded), sims))
            semantic_scores.sort(key=lambda x: x[1], reverse=True)
            semantic_ranked = [mid for mid, _ in semantic_scores]

        # Strategy 2: Keyword ranking (BM25 via FTS5) — uses pre-fetched candidates
        keyword_ranked = [m["id"] for m in fts_candidates]

        # Relevance gate: recency/importance/relationship ranking is confined to
        # memories that are semantically similar (cosine similarity strictly
        # above relevance_min_similarity — an epsilon floor so float32 matmul
        # noise on real dense embeddings, ~1e-8, doesn't count as "relevant" —
        # not merely a top-K rank slot) or lexically (FTS) matching the query. This
        # narrows, but does not eliminate, the case where a large off-topic
        # bedrock/core flood outranks a fresh on-topic fact — an off-topic
        # memory with genuine (if weak) positive similarity in the semantic
        # top-K stays eligible and can still win. Semantic and keyword
        # ranking themselves are unaffected — the gate only trims the
        # *other* two/three lists' input pool. Empty query, the flag off, or
        # an empty eligible set (no relevance signal at all) restores
        # pre-gate behavior (every list scores all_memories).
        gated = all_memories
        if mem_cfg.relevance_gate and query.strip():
            semantically_relevant = [
                mid for mid, sim in semantic_scores if sim > mem_cfg.relevance_min_similarity
            ]
            eligible = set(semantically_relevant[: mem_cfg.relevance_semantic_topk]) | set(
                keyword_ranked
            )
            # If nothing is semantically or lexically relevant (e.g. all memories
            # lack embeddings and FTS has no hit), there's no relevance signal to
            # gate on — fall back to all_memories rather than silently returning
            # nothing.
            if eligible:
                gated = [m for m in all_memories if m["id"] in eligible]

        # Strategy 3: Tier-aware recency ranking (with rowid tiebreaker for determinism)
        recency_scores = [
            (m["id"], _recency_score(m, m.get("tier", "buffer")), m.get("rowid", 0)) for m in gated
        ]
        # Sort by score descending, then by rowid ascending (newer=higher rowid comes last in tie)
        recency_scores.sort(key=lambda x: (-x[1], x[2]))
        recency_ranked = [mid for mid, _, _ in recency_scores]

        # Strategies 4 & 5 (importance, relationship) are skipped entirely — no list built,
        # no per-candidate scoring loop run — when their weight is <= 0: a 0-weight list
        # contributes nothing to weighted RRF (score * 0 == 0 from every id), so building it
        # is pure wasted work at the current relevance-first defaults (both default to 0.0).
        # `ranked_lists`/`weights` stay paired throughout — a list is appended if and only if
        # its weight is too.
        ranked_lists = [semantic_ranked, keyword_ranked, recency_ranked]
        weights = [mem_cfg.weight_semantic, mem_cfg.weight_keyword, mem_cfg.weight_recency]

        # Strategy 4: Importance with tier boost + user affinity (with rowid tiebreaker).
        # Importance is a real per-row score (base * certainty + tier boost), so exact
        # ties are rarer than for the boolean relationship signal below, but they do
        # happen (e.g. a batch of same-tier, same-importance, no-certainty facts).
        # Ties break newest-first (descending rowid) — NOT oldest-first: with a stable
        # sort over `gated` (which is ordered ascending-rowid, see above), an ascending
        # rowid tiebreak would silently prefer older memories among every tied group,
        # matching the same oldest-first bias found in the relationship strategy below.
        if mem_cfg.weight_importance > 0:
            importance_scores = []
            for m in gated:
                base = m.get("importance", 0.5) * m.get("certainty", 1.0)
                boost = _get_tier_boosts().get(m.get("tier", "buffer"), 0.0)
                # User affinity bonus
                if relationship_target:
                    meta = m.get("metadata", {})
                    if meta.get("user_id") == relationship_target:
                        base += 0.2
                importance_scores.append((m["id"], base + boost, m.get("rowid", 0)))
            # Sort by score descending, then by rowid descending (newest wins ties).
            importance_scores.sort(key=lambda x: (-x[1], -x[2]))
            ranked_lists.append([mid for mid, _, _ in importance_scores])
            weights.append(mem_cfg.weight_importance)

        # Strategy 5: Relationship boost (if target specified).
        #
        # Regression fix (ranking_experiments 2026-08-30): the relationship boost is
        # binary (a candidate either mentions relationship_target or it doesn't), so
        # the untouched majority all tie at 0.0. The old code ranked *every* gated
        # candidate — the tied majority then fell back to Python's stable sort, which
        # preserves `gated`'s ascending-rowid input order, i.e. oldest first. With
        # weight_relationship > 0 that silently injected an oldest-first bias into
        # RRF fusion for every candidate that doesn't mention the target — not just
        # the ones that do. Fix: only candidates with a positive boost enter the
        # relationship ranked list at all; RRF (`utils/rrf.py`) already scores an id
        # missing from a list as a 0 contribution from that list, so the untouched
        # majority now gets *no* signal from this strategy instead of an accidental
        # rank. Ties among the (typically few) matching candidates break newest-first
        # (descending rowid), matching the importance tiebreak above.
        if relationship_target and mem_cfg.weight_relationship > 0:
            rel_scores = []
            target_lower = relationship_target.lower()
            for m in gated:
                content_lower = m["content"].lower()
                meta = m.get("metadata", {})
                involves_target = (
                    target_lower in content_lower or meta.get("target_id") == relationship_target
                )
                if involves_target:
                    rel_scores.append((m["id"], 1.0, m.get("rowid", 0)))
            rel_scores.sort(key=lambda x: (-x[1], -x[2]))
            ranked_lists.append([mid for mid, _, _ in rel_scores])
            weights.append(mem_cfg.weight_relationship)

        # Fuse with weighted RRF
        fused = reciprocal_rank_fusion(ranked_lists, k=mem_cfg.rrf_k, weights=weights)

        # ── Tier 3f: multi-hop second-pass expansion (off by default) ──────────
        # When `retrieval_second_pass` (N) > 0, take the top-N fused hits as
        # "seeds" and look for co-dependent evidence that the first pass missed
        # because it shares terms with a SEED, not with the original query
        # (e.g. "Rocket the beagle" showing up only once the seed "Caroline
        # adopted a beagle named Rocket" narrows the search). Cost bound: at
        # most one extra `fts_search` call and zero embedder calls — the
        # "semantic" side of the widening reuses vectors already stored on the
        # seeds and candidates, meaned in pure Python/numpy.
        #
        # This block runs entirely after the original fusion above and never
        # touches its inputs, so `retrieval_second_pass=0` (the default)
        # leaves `fused` byte-identical to today's ranking.
        if mem_cfg.retrieval_second_pass > 0 and query.strip() and fused:
            n = mem_cfg.retrieval_second_pass
            seeds = [memory_map[mid] for mid, _ in fused[:n] if mid in memory_map]
            if seeds:
                combined_text = " ".join(s.get("content", "") for s in seeds)
                terms = _salient_terms(combined_text)
                try:
                    # fts_search tokenizes on \w+ and ORs the tokens itself;
                    # pre-joining with a literal " OR " made "OR" a search term
                    # matching the word "or" corpus-wide (review 2026-09-01).
                    second_fts = self.storage.fts_search(
                        self.character_id, " ".join(terms), limit=50
                    )
                except Exception:
                    second_fts = []

                # Widen the candidate pool with any genuinely new rows found via
                # the seed-term FTS pull (their vectors arrive on the row dicts —
                # no separate embedding fetch needed).
                new_rows = [m for m in second_fts if m["id"] not in memory_map]
                for m in new_rows:
                    memory_map[m["id"]] = m
                if new_rows:
                    all_memories = all_memories + new_rows
                    all_memories.sort(key=lambda m: m.get("rowid", 0))

                # Strategy 1 (semantic), recomputed against the ORIGINAL query
                # embedding over the widened pool — no new embed call; the new
                # rows' vectors came back on the fts_search rows themselves.
                embedded2 = [m for m in all_memories if m.get("embedding")]
                sims2 = cosine_matrix(query_embedding, [m["embedding"] for m in embedded2])
                semantic_scores2 = list(zip((m["id"] for m in embedded2), sims2))
                semantic_scores2.sort(key=lambda x: x[1], reverse=True)
                semantic_ranked2 = [mid for mid, _ in semantic_scores2]

                # Strategy 2 (keyword): extend with the second-pass FTS hits so
                # they carry a real (if weak) keyword-fusion contribution — this
                # is how co-dependent evidence found only via seed terms (not
                # the original query) earns rank credit and gate eligibility.
                already_keyword = set(keyword_ranked)
                keyword_ranked2 = keyword_ranked + [
                    m["id"] for m in second_fts if m["id"] not in already_keyword
                ]

                # Mean-of-seeds vector widens gate eligibility only — it is NOT
                # a new RRF-weighted list, so ranking itself stays anchored to
                # the original query's semantic/keyword signals above. Pure
                # Python mean over already-stored vectors; no embedder call.
                seed_vectors = [s["embedding"] for s in seeds if s.get("embedding")]
                mean_eligible: set[str] = set()
                if seed_vectors:
                    dims = len(seed_vectors[0])
                    if all(len(v) == dims for v in seed_vectors):
                        mean_vector = [
                            sum(v[i] for v in seed_vectors) / len(seed_vectors) for i in range(dims)
                        ]
                        mean_sims = cosine_matrix(mean_vector, [m["embedding"] for m in embedded2])
                        mean_scores = list(zip((m["id"] for m in embedded2), mean_sims))
                        mean_scores.sort(key=lambda x: x[1], reverse=True)
                        mean_eligible = {
                            mid
                            for mid, sim in mean_scores[: mem_cfg.relevance_semantic_topk]
                            if sim > mem_cfg.relevance_min_similarity
                        }

                # Re-derive the gate-eligible pool (widened) and re-run
                # strategies 3-5 over it — same weights/gate semantics as the
                # first pass, just over the widened candidates.
                gated2 = all_memories
                if mem_cfg.relevance_gate:
                    semantically_relevant2 = [
                        mid
                        for mid, sim in semantic_scores2
                        if sim > mem_cfg.relevance_min_similarity
                    ]
                    eligible2 = (
                        set(semantically_relevant2[: mem_cfg.relevance_semantic_topk])
                        | set(keyword_ranked2)
                        | mean_eligible
                    )
                    if eligible2:
                        gated2 = [m for m in all_memories if m["id"] in eligible2]

                recency_scores2 = [
                    (m["id"], _recency_score(m, m.get("tier", "buffer")), m.get("rowid", 0))
                    for m in gated2
                ]
                recency_scores2.sort(key=lambda x: (-x[1], x[2]))
                recency_ranked2 = [mid for mid, _, _ in recency_scores2]

                ranked_lists2 = [semantic_ranked2, keyword_ranked2, recency_ranked2]
                weights2 = [
                    mem_cfg.weight_semantic,
                    mem_cfg.weight_keyword,
                    mem_cfg.weight_recency,
                ]

                if mem_cfg.weight_importance > 0:
                    importance_scores2 = []
                    for m in gated2:
                        base = m.get("importance", 0.5) * m.get("certainty", 1.0)
                        boost = _get_tier_boosts().get(m.get("tier", "buffer"), 0.0)
                        if relationship_target:
                            meta = m.get("metadata", {})
                            if meta.get("user_id") == relationship_target:
                                base += 0.2
                        importance_scores2.append((m["id"], base + boost, m.get("rowid", 0)))
                    importance_scores2.sort(key=lambda x: (-x[1], -x[2]))
                    ranked_lists2.append([mid for mid, _, _ in importance_scores2])
                    weights2.append(mem_cfg.weight_importance)

                if relationship_target and mem_cfg.weight_relationship > 0:
                    rel_scores2 = []
                    target_lower = relationship_target.lower()
                    for m in gated2:
                        content_lower = m["content"].lower()
                        meta = m.get("metadata", {})
                        involves_target = (
                            target_lower in content_lower
                            or meta.get("target_id") == relationship_target
                        )
                        if involves_target:
                            rel_scores2.append((m["id"], 1.0, m.get("rowid", 0)))
                    rel_scores2.sort(key=lambda x: (-x[1], -x[2]))
                    ranked_lists2.append([mid for mid, _, _ in rel_scores2])
                    weights2.append(mem_cfg.weight_relationship)

                fused = reciprocal_rank_fusion(ranked_lists2, k=mem_cfg.rrf_k, weights=weights2)

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
