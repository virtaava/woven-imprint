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
