from tests.helpers import FakeEmbedder
from woven_imprint.embedding.cache import CachedEmbedder


def test_repeat_embed_hits_cache():
    inner = FakeEmbedder()
    cached = CachedEmbedder(inner)
    v1 = cached.embed("hello world")
    v2 = cached.embed("hello world")
    assert v1 == v2
    assert cached.hits == 1
    assert cached.misses == 1


def test_lru_eviction():
    cached = CachedEmbedder(FakeEmbedder(), maxsize=2)
    cached.embed("a")
    cached.embed("b")
    cached.embed("c")  # evicts "a"
    cached.embed("a")
    assert cached.misses == 4


def test_engine_wraps_embedder():
    from tests.helpers import make_test_engine

    engine = make_test_engine()
    assert isinstance(engine.embedding, CachedEmbedder)


def test_indexes_exist():
    from tests.helpers import make_test_engine

    engine = make_test_engine()
    names = {
        r[0]
        for r in engine.storage._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
    }
    assert "idx_memories_accessed" in names
    assert "idx_memories_created" in names
