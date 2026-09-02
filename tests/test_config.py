def test_query_expansion_defaults():
    """Tier 3i (docs/superpowers/specs/2026-09-02-tier3i-llm-query-expansion.md):
    LLM query expansion ships opt-in — 0 queries by default, weight 0.5."""
    from woven_imprint.config import MemoryConfig

    cfg = MemoryConfig()
    assert cfg.query_expansion == 0
    assert cfg.query_expansion_weight == 0.5


def test_query_expansion_negative_rejected():
    """Tier 3i spec: `query_expansion >= 0`."""
    import pytest

    from woven_imprint.config import MemoryConfig

    with pytest.raises(ValueError):
        MemoryConfig(query_expansion=-1)


def test_query_expansion_weight_non_positive_rejected():
    """Tier 3i spec: `query_expansion_weight > 0`."""
    import pytest

    from woven_imprint.config import MemoryConfig

    with pytest.raises(ValueError):
        MemoryConfig(query_expansion_weight=0.0)


def test_query_expansion_valid_custom_pair_constructs():
    from woven_imprint.config import MemoryConfig

    cfg = MemoryConfig(query_expansion=3, query_expansion_weight=0.5)
    assert cfg.query_expansion == 3
    assert cfg.query_expansion_weight == 0.5
