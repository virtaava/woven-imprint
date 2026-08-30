"""Tests for memory retrieval — the core of the persistence system."""

import pytest

from woven_imprint.storage.sqlite import SQLiteStorage
from woven_imprint.memory.retrieval import MemoryRetriever, _recency_score


class WordEmbedder:
    """Embedder that creates vectors based on word overlap — deterministic."""

    def __init__(self, dims=50):
        self._vocab = {}
        self._next = 0
        self._dims = dims

    def embed(self, text):
        vec = [0.0] * self._dims
        for word in text.lower().split():
            if word not in self._vocab:
                self._vocab[word] = self._next % self._dims
                self._next += 1
            vec[self._vocab[word]] += 1.0
        mag = sum(x * x for x in vec) ** 0.5
        if mag > 0:
            vec = [x / mag for x in vec]
        return vec

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]

    def dimensions(self):
        return self._dims


@pytest.fixture
def setup():
    storage = SQLiteStorage(":memory:")
    storage.save_character("c1", "Alice", {})
    embedder = WordEmbedder()
    retriever = MemoryRetriever(storage, embedder, "c1")
    yield storage, embedder, retriever
    storage.close()


class TestRetrieval:
    def test_empty_returns_empty(self, setup):
        _, _, retriever = setup
        results = retriever.retrieve("anything")
        assert results == []

    def test_finds_relevant_memory(self, setup):
        storage, embedder, retriever = setup
        vec = embedder.embed("The harbor case is solved")
        storage.save_memory(
            {
                "id": "m1",
                "character_id": "c1",
                "tier": "core",
                "content": "The harbor case is solved",
                "embedding": vec,
                "importance": 0.8,
            }
        )
        results = retriever.retrieve("harbor case", limit=5)
        assert len(results) >= 1
        assert "harbor" in results[0]["content"].lower()

    def test_semantic_ranking(self, setup):
        storage, embedder, retriever = setup
        # Relevant memory
        storage.save_memory(
            {
                "id": "m1",
                "character_id": "c1",
                "tier": "core",
                "content": "Marcus went missing near the Thames river",
                "embedding": embedder.embed("Marcus went missing near the Thames river"),
                "importance": 0.5,
            }
        )
        # Irrelevant memory
        storage.save_memory(
            {
                "id": "m2",
                "character_id": "c1",
                "tier": "core",
                "content": "Had lunch at the cafe today nice weather",
                "embedding": embedder.embed("Had lunch at the cafe today nice weather"),
                "importance": 0.5,
            }
        )

        results = retriever.retrieve("missing person Thames", limit=2)
        assert results[0]["id"] == "m1"

    def test_keyword_ranking_via_fts(self, setup):
        storage, embedder, retriever = setup
        storage.save_memory(
            {
                "id": "m1",
                "character_id": "c1",
                "tier": "core",
                "content": "Detective work on Blackwood murder investigation",
                "embedding": embedder.embed("detective blackwood murder"),
                "importance": 0.5,
            }
        )
        storage.save_memory(
            {
                "id": "m2",
                "character_id": "c1",
                "tier": "core",
                "content": "Weather was nice at the park",
                "embedding": embedder.embed("weather nice park"),
                "importance": 0.5,
            }
        )

        results = retriever.retrieve("Blackwood murder", limit=2)
        assert results[0]["id"] == "m1"

    def test_importance_affects_ranking(self, setup):
        """weight_importance defaults to 0.0 (relevance-first RRF, 2026-08-30) —
        the importance strategy's machinery must still work when a caller opts
        back in by raising the weight, so this test exercises it explicitly."""
        storage, embedder, retriever = setup
        vec = embedder.embed("some event happened")
        storage.save_memory(
            {
                "id": "m1",
                "character_id": "c1",
                "tier": "core",
                "content": "Important event happened",
                "embedding": vec,
                "importance": 0.9,
            }
        )
        storage.save_memory(
            {
                "id": "m2",
                "character_id": "c1",
                "tier": "core",
                "content": "Trivial event happened",
                "embedding": vec,
                "importance": 0.1,
            }
        )

        from woven_imprint.config import get_config

        cfg = get_config().memory
        orig = cfg.weight_importance
        cfg.weight_importance = 1.0
        try:
            results = retriever.retrieve("event happened", limit=2)
        finally:
            cfg.weight_importance = orig
        # Higher importance should rank higher
        ids = [r["id"] for r in results]
        assert ids.index("m1") < ids.index("m2")

    def test_tier_boost(self, setup):
        """weight_importance defaults to 0.0 (relevance-first RRF, 2026-08-30) —
        the tier-boost machinery it feeds must still work when a caller opts
        back in by raising the weight, so this test exercises it explicitly."""
        storage, embedder, retriever = setup
        vec = embedder.embed("my identity")
        storage.save_memory(
            {
                "id": "m1",
                "character_id": "c1",
                "tier": "bedrock",
                "content": "My identity is detective",
                "embedding": vec,
                "importance": 0.5,
            }
        )
        storage.save_memory(
            {
                "id": "m2",
                "character_id": "c1",
                "tier": "buffer",
                "content": "My identity mentioned in passing",
                "embedding": vec,
                "importance": 0.5,
            }
        )

        from woven_imprint.config import get_config

        cfg = get_config().memory
        orig = cfg.weight_importance
        cfg.weight_importance = 1.0
        try:
            results = retriever.retrieve("identity", limit=2)
        finally:
            cfg.weight_importance = orig
        # Bedrock should rank higher due to tier boost
        assert results[0]["tier"] == "bedrock"

    def test_relationship_boost(self, setup):
        storage, embedder, retriever = setup
        storage.save_memory(
            {
                "id": "m1",
                "character_id": "c1",
                "tier": "core",
                "content": "Talked with player_bob about the case",
                "embedding": embedder.embed("talked player_bob case"),
                "importance": 0.5,
            }
        )
        storage.save_memory(
            {
                "id": "m2",
                "character_id": "c1",
                "tier": "core",
                "content": "Thought about the case alone",
                "embedding": embedder.embed("thought about case alone"),
                "importance": 0.5,
            }
        )

        results = retriever.retrieve("the case", limit=2, relationship_target="player_bob")
        # Memory involving player_bob should rank higher
        assert "player_bob" in results[0]["content"]

    def test_relationship_weight_no_hidden_bias_when_nothing_matches(self, setup, monkeypatch):
        """Regression: the relationship strategy must not inject an oldest-first
        bias into RRF fusion for candidates that don't mention relationship_target.

        Before the fix, the relationship ranked list ranked EVERY gated candidate
        on a binary boost (1.0 if it mentions the target, else 0.0). The untouched
        majority all tied at 0.0, and Python's stable sort fell back to `gated`'s
        ascending-rowid input order (oldest first) — so with weight_relationship > 0,
        older memories were silently favored over newer, more relevant ones for
        every candidate that doesn't mention the target, not just the ones that do.

        Construction: monkeypatch semantic similarity so a NEWER memory (highest
        rowid, added last) is genuinely the most relevant, and an OLDER memory
        (lowest rowid, added first) is a close second — a realistic "a fresh fact
        slightly edges out an older one" scenario, with 4 filler memories in
        between by insertion order. Neither "old" nor "new" mentions
        relationship_target. Under the pre-fix code, the relationship list's
        oldest-first bias (old at rank 0, new at the last rank in that list) is
        large enough to flip the fused winner from "new" to "old" once
        weight_relationship=1. Under the fix, the relationship list is empty
        (nothing matches it) and contributes nothing, so fusion order at
        weight_relationship=1 must be identical to weight_relationship=0.
        """
        from woven_imprint.config import get_config
        from woven_imprint.memory import retrieval as retrieval_mod

        storage, _embedder, retriever = setup
        insertion_order = ["old", "f1", "f2", "f3", "f4", "new"]
        for mid in insertion_order:
            storage.save_memory(
                {
                    "id": mid,
                    "character_id": "c1",
                    "tier": "core",
                    "content": f"memory {mid} about the lighthouse",
                    "embedding": [1.0],  # placeholder — cosine_matrix is monkeypatched below
                    "importance": 0.5,
                }
            )

        # `embedded` preserves all_memories' ascending-rowid order == insertion_order
        # above, so index i of this list corresponds to insertion_order[i]. "new"
        # (last inserted, highest rowid) gets the top similarity; "old" (first
        # inserted, lowest rowid) is a close second; fillers trail well behind.
        sims_by_insertion_order = [0.59, 0.1, 0.09, 0.08, 0.07, 0.6]
        monkeypatch.setattr(
            retrieval_mod, "cosine_matrix", lambda query, rows: sims_by_insertion_order
        )

        cfg = get_config().memory
        orig_rel = cfg.weight_relationship
        orig_kw = cfg.weight_keyword
        cfg.weight_keyword = 0.0  # isolate the semantic-vs-relationship interaction
        try:
            cfg.weight_relationship = 0.0
            baseline = [
                m["id"]
                for m in retriever.retrieve(
                    "lighthouse", limit=10, relationship_target="carol_not_mentioned"
                )
            ]
            cfg.weight_relationship = 1.0
            with_weight = [
                m["id"]
                for m in retriever.retrieve(
                    "lighthouse", limit=10, relationship_target="carol_not_mentioned"
                )
            ]
        finally:
            cfg.weight_relationship = orig_rel
            cfg.weight_keyword = orig_kw

        assert baseline[0] == "new", baseline  # sanity: semantics alone favor the newer memory
        assert with_weight == baseline, (
            "weight_relationship=1 changed fusion order with no relationship match "
            f"(hidden oldest-first bias): baseline={baseline} with_weight={with_weight}"
        )

    def test_character_isolation(self, setup):
        storage, embedder, retriever = setup
        storage.save_character("c2", "Bob", {})
        vec = embedder.embed("secret information")
        storage.save_memory(
            {
                "id": "m1",
                "character_id": "c1",
                "tier": "core",
                "content": "Alice secret info",
                "embedding": vec,
            }
        )
        storage.save_memory(
            {
                "id": "m2",
                "character_id": "c2",
                "tier": "core",
                "content": "Bob secret info",
                "embedding": vec,
            }
        )

        results = retriever.retrieve("secret", limit=10)
        ids = [r["id"] for r in results]
        assert "m1" in ids
        assert "m2" not in ids  # Bob's memory shouldn't appear

    def test_fts_prefetch_finds_old_memories(self, setup):
        storage, embedder, retriever = setup
        # This memory would normally be outside the LIMIT 200 recency window
        storage.save_memory(
            {
                "id": "old1",
                "character_id": "c1",
                "tier": "core",
                "content": "The ancient Blackwood manuscript was found in the cellar",
                "embedding": embedder.embed("ancient Blackwood manuscript cellar"),
                "importance": 0.9,
            }
        )
        # FTS should find it by keyword even if recency is low
        results = retriever.retrieve("Blackwood manuscript", limit=5)
        assert any("Blackwood" in r["content"] for r in results)

    def test_retrieval_score_included(self, setup):
        storage, embedder, retriever = setup
        storage.save_memory(
            {
                "id": "m1",
                "character_id": "c1",
                "tier": "core",
                "content": "test memory",
                "embedding": embedder.embed("test memory"),
            }
        )
        results = retriever.retrieve("test", limit=1)
        assert "_retrieval_score" in results[0]
        assert results[0]["_retrieval_score"] > 0


class TestRecencyScore:
    def test_recency_anchor_created_ignores_touches(self):
        """Retrieval touching accessed_at must not make old memories look fresh."""
        old_created = "2020-01-01 00:00:00"
        fresh_touch = "2099-01-01 00:00:00"
        mem = {"created_at": old_created, "accessed_at": fresh_touch}
        score = _recency_score(mem, "buffer")
        assert score < 0.01  # decayed to ~nothing despite the fresh touch

    def test_bedrock_decays_slowly(self):
        # 1 week ago
        from datetime import datetime, timezone, timedelta

        one_week_ago = (datetime.now(timezone.utc) - timedelta(hours=168)).isoformat()
        mem_old = {"created_at": one_week_ago, "accessed_at": one_week_ago}
        bedrock = _recency_score(mem_old, "bedrock")
        buffer = _recency_score(mem_old, "buffer")
        assert bedrock > buffer  # bedrock should retain more

    def test_recent_scores_high(self):
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        mem_now = {"created_at": now, "accessed_at": now}
        score = _recency_score(mem_now, "core")
        assert score > 0.99

    def test_invalid_timestamp(self):
        mem = {"created_at": "not-a-date", "accessed_at": "not-a-date"}
        score = _recency_score(mem, "core")
        assert score == 0.5  # default fallback


def test_personal_core_fact_beats_bedrock_seed_flood():
    """Regression: many bedrock seeds must not drown a query-relevant core fact."""
    from tests.helpers import make_test_engine

    engine = make_test_engine()
    char = engine.create_character("Meridian", persona={"backstory": "A wizard of the old tower."})
    # Flood bedrock with irrelevant seeds
    for i in range(30):
        char.memory.add(
            content=f"Ancient tower lore volume {i}: the stones hum at dusk.",
            tier="bedrock",
            role="observation",
            importance=0.9,
        )
    # One personal core fact
    char.memory.add(
        content="The user's sister is named Anna and she loves rowing.",
        tier="core",
        role="observation",
        importance=0.75,
    )
    results = char.retriever.retrieve("what is my sister's name", limit=5)
    contents = [m["content"] for m in results]
    assert any("Anna" in c for c in contents), contents


def test_fts_search_carries_rowid():
    """Regression: fts_search must include rowid so retrieval tiebreakers are sound.

    When a memory is found via Phase-2 FTS (older than recency window),
    it must carry rowid so that tiebreakers in retrieval._recency_score()
    and retrieval.importance_scores work correctly.

    This test creates >100 buffer memories to push an early distinctive memory
    out of the Phase-1 recency window, then retrieves by FTS and verifies rowid.
    """
    from tests.helpers import make_test_engine

    engine = make_test_engine()
    char = engine.create_character("Zephyr", persona={"role": "scout"})

    # Add the distinctive old memory that will be found via FTS
    char.memory.add(
        content="I saw zephyrblossom flowers blooming near the ancient grove.",
        tier="buffer",
        role="observation",
        importance=0.6,
    )

    # Insert >100 buffer memories to push the old one out of Phase-1 window (limit=100)
    for i in range(105):
        char.memory.add(
            content=f"Daily observation {i}: encountered unrelated event number {i}.",
            tier="buffer",
            role="observation",
            importance=0.5,
        )

    # Retrieve by the distinctive keyword — must use FTS Phase-2
    results = char.retriever.retrieve("zephyrblossom", limit=5)

    # Assert: (a) old memory is found (proves FTS Phase-2 works)
    assert any("zephyrblossom" in r["content"] for r in results), (
        "FTS Phase-2 should find memory outside recency window"
    )

    # Assert: (b) rowid is in dict and is positive int (proves fts_search carries rowid)
    old_mem = next(r for r in results if "zephyrblossom" in r["content"])
    assert "rowid" in old_mem, "fts_search result must include rowid"
    assert isinstance(old_mem["rowid"], int), "rowid must be an integer"
    assert old_mem["rowid"] > 0, "rowid must be positive"
