"""Retrieval must scan all active memories (up to max_candidates), not just a 200-row window."""

from tests.helpers import make_test_engine


def test_old_core_memory_found_beyond_200_window():
    engine = make_test_engine()
    char = engine.create_character("Rex")
    needle = char.memory.add("The visitor's cat is named Pixel and likes windowsills.", tier="core")
    for i in range(260):
        char.memory.add(f"Routine note number {i} about weather and errands.", tier="core")
    ids = [m["id"] for m in char.retriever.retrieve("what is the cat named", limit=10)]
    assert needle["id"] in ids


def test_get_memories_limit_none_returns_all():
    engine = make_test_engine()
    char = engine.create_character("Rex")
    for i in range(1500):
        char.memory.add_without_embedding(f"m{i}", tier="core")
    assert len(engine.storage.get_memories(char.id, limit=None)) == 1500


def test_cosine_matrix_matches_pairwise():
    from woven_imprint.memory.retrieval import _cosine_similarity, cosine_matrix

    q = [1.0, 0.0, 0.5]
    rows = [[1.0, 0.0, 0.5], [0.0, 1.0, 0.0], [0.2, 0.2, 0.2]]
    got = cosine_matrix(q, rows)
    for g, r in zip(got, rows):
        assert abs(g - _cosine_similarity(q, r)) < 1e-6


def test_cosine_matrix_matches_pairwise_pure_python(monkeypatch):
    """Pure-Python fallback path (numpy unavailable) must match the pairwise result too."""
    from woven_imprint.memory import retrieval

    monkeypatch.setattr(retrieval, "_np", None)

    q = [1.0, 0.0, 0.5]
    rows = [[1.0, 0.0, 0.5], [0.0, 1.0, 0.0], [0.2, 0.2, 0.2]]
    got = retrieval.cosine_matrix(q, rows)
    for g, r in zip(got, rows):
        assert abs(g - retrieval._cosine_similarity(q, r)) < 1e-6


def test_cosine_matrix_skips_ragged_row_numpy():
    from woven_imprint.memory.retrieval import _cosine_similarity, cosine_matrix

    q = [1.0, 0.0, 0.5]
    rows = [[1.0, 0.0, 0.5], [0.2, 0.2], [0.2, 0.2, 0.2]]  # middle row is ragged (2-dim)
    got = cosine_matrix(q, rows)
    assert got[1] == 0.0
    assert abs(got[0] - _cosine_similarity(q, rows[0])) < 1e-6
    assert abs(got[2] - _cosine_similarity(q, rows[2])) < 1e-6


def test_cosine_matrix_skips_ragged_row_pure_python(monkeypatch):
    from woven_imprint.memory import retrieval

    monkeypatch.setattr(retrieval, "_np", None)

    q = [1.0, 0.0, 0.5]
    rows = [[1.0, 0.0, 0.5], [0.2, 0.2], [0.2, 0.2, 0.2]]  # middle row is ragged (2-dim)
    got = retrieval.cosine_matrix(q, rows)
    assert got[1] == 0.0
    assert abs(got[0] - retrieval._cosine_similarity(q, rows[0])) < 1e-6
    assert abs(got[2] - retrieval._cosine_similarity(q, rows[2])) < 1e-6


def test_retrieve_survives_ragged_embedding_dimension():
    """A legacy memory saved with a different embedding dimension must not crash retrieve()."""
    engine = make_test_engine()
    char = engine.create_character("Rex")
    good = char.memory.add("The visitor's cat is named Pixel and likes windowsills.", tier="core")
    for i in range(5):
        char.memory.add(f"Routine note number {i} about weather and errands.", tier="core")

    # Bypass the normal add() dimension guard to simulate a legacy row from a
    # different embedding model/dimension living alongside current-dimension rows.
    engine.storage.save_memory(
        {
            "id": "legacy-ragged-embedding",
            "character_id": char.id,
            "tier": "core",
            "content": "An old memory embedded with a different model.",
            "embedding": [0.1, 0.2, 0.3],  # 3-dim, while FakeEmbedder produces 50-dim
        }
    )

    ids = [m["id"] for m in char.retriever.retrieve("what is the cat named", limit=10)]
    assert good["id"] in ids
