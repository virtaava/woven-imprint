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
