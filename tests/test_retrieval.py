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

        results = retriever.retrieve("event happened", limit=2)
        # Higher importance should rank higher
        ids = [r["id"] for r in results]
        assert ids.index("m1") < ids.index("m2")

    def test_tier_boost(self, setup):
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

        results = retriever.retrieve("identity", limit=2)
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
    def test_bedrock_decays_slowly(self):
        # 1 week ago
        from datetime import datetime, timezone, timedelta

        one_week_ago = (datetime.now(timezone.utc) - timedelta(hours=168)).isoformat()
        bedrock = _recency_score(one_week_ago, "bedrock")
        buffer = _recency_score(one_week_ago, "buffer")
        assert bedrock > buffer  # bedrock should retain more

    def test_recent_scores_high(self):
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        score = _recency_score(now, "core")
        assert score > 0.99

    def test_invalid_timestamp(self):
        score = _recency_score("not-a-date", "core")
        assert score == 0.5  # default fallback
