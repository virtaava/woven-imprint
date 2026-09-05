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


def test_second_pass_entity_max_df_default():
    """Tier 3p (docs/superpowers/specs/2026-09-05-tier3p-df-entity-pivot.md):
    default DF ceiling is 0.05, off (no effect) unless second_pass_entities=True."""
    from woven_imprint.config import MemoryConfig

    cfg = MemoryConfig()
    assert cfg.second_pass_entity_max_df == 0.05
    assert cfg.second_pass_entities is False


def test_second_pass_entity_max_df_zero_rejected():
    """Tier 3p spec: `0 < second_pass_entity_max_df <= 1`."""
    import pytest

    from woven_imprint.config import MemoryConfig

    with pytest.raises(ValueError):
        MemoryConfig(second_pass_entity_max_df=0.0)


def test_second_pass_entity_max_df_negative_rejected():
    import pytest

    from woven_imprint.config import MemoryConfig

    with pytest.raises(ValueError):
        MemoryConfig(second_pass_entity_max_df=-0.1)


def test_second_pass_entity_max_df_above_one_rejected():
    import pytest

    from woven_imprint.config import MemoryConfig

    with pytest.raises(ValueError):
        MemoryConfig(second_pass_entity_max_df=1.1)


def test_second_pass_entity_max_df_boundary_one_accepted():
    """Upper bound is inclusive: 1.0 means "no filtering" (every entity is
    pivot-eligible), not an invalid value."""
    from woven_imprint.config import MemoryConfig

    cfg = MemoryConfig(second_pass_entity_max_df=1.0)
    assert cfg.second_pass_entity_max_df == 1.0


def test_second_pass_entity_max_df_valid_custom_value_constructs():
    from woven_imprint.config import MemoryConfig

    cfg = MemoryConfig(second_pass_entity_max_df=0.2, second_pass_entities=True)
    assert cfg.second_pass_entity_max_df == 0.2
    assert cfg.second_pass_entities is True
