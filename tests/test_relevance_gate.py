"""Retrieval relevance gate: recency/importance/relationship re-rank only
memories that are semantically or lexically relevant to the query — an
off-topic bedrock/core flood must not drown out an on-topic fresh fact."""

from tests.helpers import make_test_engine
from woven_imprint.config import get_config


def _setup():
    engine = make_test_engine()
    char = engine.create_character(
        "Ada", persona={"personality": "patient and wise", "speaking_style": "warm"}
    )
    for i in range(30):  # bedrock flood, off-topic
        char.memory.add(
            f"[Self] I value {i} things about the archive and its silence.",
            tier="bedrock",
            importance=0.9,
        )
    fresh = char.memory.add(
        "The visitor dislikes tea and prefers coffee.", tier="core", importance=0.75
    )
    return engine, char, fresh


def test_gate_puts_on_topic_fact_first():
    engine, char, fresh = _setup()
    top = char.retriever.retrieve("tea", limit=5)
    assert top[0]["id"] == fresh["id"]


def test_gate_off_restores_legacy_flood():
    engine, char, fresh = _setup()
    get_config().memory.relevance_gate = False
    try:
        top = char.retriever.retrieve("tea", limit=5)
        # legacy: bedrock floor may outrank; we only assert the fresh fact is still retrievable
        assert any(m["id"] == fresh["id"] for m in top)
    finally:
        get_config().memory.relevance_gate = True


def test_empty_query_ignores_gate():
    engine, char, fresh = _setup()
    assert len(char.retriever.retrieve("", limit=5)) == 5


def test_gate_falls_back_to_all_candidates_when_eligible_is_empty():
    """No relevance signal at all (no embeddings, no FTS hit) must not collapse to []."""
    engine = make_test_engine()
    # No persona -> _seed_bedrock adds nothing, so the candidate pool is
    # exactly the 5 memories below (no stray embedded seed to keep `eligible`
    # from being truly empty).
    char = engine.create_character("Ada")
    contents = [
        "The lantern glowed near the old harbor at dusk.",
        "A violin was left on the windowsill overnight.",
        "The orchard gate creaked in the evening wind.",
        "Someone left a compass on the kitchen table.",
        "The kettle whistled twice before anyone answered.",
    ]
    for content in contents:
        char.memory.add_without_embedding(content, tier="core", importance=0.6)

    top = char.retriever.retrieve("zzqx unknown", limit=5)
    assert len(top) == min(5, len(contents))


def test_gate_excludes_zero_similarity_from_semantic_eligibility():
    """Regression: semantic eligibility used to be `top relevance_semantic_topk
    BY RANK`, which admits zero-similarity memories whenever fewer than topk
    memories have any similarity at all. With `relevance_semantic_topk`
    defaulting to 100, 150 zero-similarity bedrock lines exceed that cap, so
    the old rank-based slice let 99 of them ride a rank slot into the gated
    recency/importance eligibility set purely by tie order — none of them
    have anything to do with the query "tea". Their permanent
    importance/recency floor (importance=0.95, bedrock tier boost) then
    outranked the single on-topic fact (importance=0.75) for first place.

    Embeddings are set explicitly (bypassing FakeEmbedder's incremental
    per-instance vocab table) so similarity is deterministic: the fact's
    embedding exactly equals the query's embedding (cosine == 1.0), and
    every bedrock line's embedding is a unit vector on a dimension the query
    never touches (cosine == 0.0 with the query, guaranteed, not just "no
    shared words by luck").

    Eligibility now requires cosine similarity > 0, so the 150
    zero-similarity bedrock lines never enter the gated recency/importance
    lists at all (regardless of topk), and the on-topic fact — the only
    memory eligible on either semantic or keyword grounds — wins outright.
    """
    engine = make_test_engine()
    char = engine.create_character("Ada", persona={"personality": "patient"})

    query_vec = char.retriever.embedder.embed("tea")
    dims = len(query_vec)
    used = {i for i, v in enumerate(query_vec) if v}
    orthogonal_idx = next(i for i in range(dims) if i not in used)
    orthogonal_vec = [0.0] * dims
    orthogonal_vec[orthogonal_idx] = 1.0

    for i in range(150):
        m = char.memory.add(
            f"[Self] entry {i}: quiet mornings and old maps.",
            tier="bedrock",
            importance=0.95,
        )
        m["embedding"] = orthogonal_vec
        char.storage.save_memory(m)

    fact = char.memory.add("The visitor dislikes tea.", tier="core", importance=0.75)
    fact["embedding"] = query_vec
    char.storage.save_memory(fact)

    results = char.retriever.retrieve("tea", limit=20)
    assert results[0]["id"] == fact["id"], (
        f"expected the on-topic fact first, got: {results[0]['content']!r}"
    )


def test_relevance_min_similarity_excludes_float32_noise(monkeypatch):
    """Regression for `MemoryConfig.relevance_min_similarity`: on real dense
    embeddings, float32 matmul (numpy's fast path in `cosine_matrix`) can
    yield a ~1e-8 "similarity" between vectors that are mathematically
    orthogonal — noise, not a genuine semantic match. A bare `sim > 0.0`
    floor treats that noise as eligible; requiring `sim >
    relevance_min_similarity` (default 1e-6) does not.

    Approach: monkeypatch `retrieval.cosine_matrix` to return exactly
    `[1e-8, 0.9]` for two memories, so the test is deterministic regardless
    of the embedder or numpy's actual float32 behavior. `noisy` is off-topic
    content given a huge recency/importance floor (bedrock tier, high
    importance) — if it slipped into the gated recency/importance
    eligibility set (old `sim > 0.0` floor) it would outrank `ontopic`
    (genuinely relevant at 0.9 similarity, but modest importance and no
    keyword overlap with the query) despite trailing badly on the semantic
    signal itself. With the epsilon floor, `noisy`'s 1e-8 no longer clears
    eligibility, so it gets zero contribution from the recency/importance
    strategies and `ontopic` wins the fused ranking — verified by hand: with
    the old `sim > 0.0` gate this assertion fails (noisy's recency+importance
    dominance wins the RRF fusion despite second-place semantics).
    """
    from woven_imprint.memory import retrieval as retrieval_mod

    engine = make_test_engine()
    char = engine.create_character("Ada")

    noisy = char.memory.add_without_embedding(
        "The kettle whistled twice before anyone answered.",
        tier="bedrock",
        importance=0.95,
    )
    ontopic = char.memory.add_without_embedding(
        "The visitor's favorite beverage arrived cold.",
        tier="buffer",
        importance=0.3,
    )
    # Placeholder embeddings so the semantic strategy includes both rows —
    # the values don't matter since cosine_matrix itself is patched below.
    for m in (noisy, ontopic):
        m["embedding"] = [1.0]
        char.storage.save_memory(m)

    monkeypatch.setattr(retrieval_mod, "cosine_matrix", lambda query, rows: [1e-8, 0.9])

    results = char.retriever.retrieve("tea", limit=10)
    ids = [m["id"] for m in results]
    assert ids.index(ontopic["id"]) < ids.index(noisy["id"]), (
        "the 1e-8-similarity memory must not outrank the genuinely on-topic "
        "0.9-similarity memory — it should be excluded from the gated "
        "recency/importance eligibility set (sim > relevance_min_similarity)"
    )
