def test_query_expansion_defaults():
    """Tier 3i (docs/superpowers/specs/2026-09-02-tier3i-llm-query-expansion.md):
    LLM query expansion ships opt-in — 0 queries by default, weight 0.5."""
    from woven_imprint.config import MemoryConfig

    cfg = MemoryConfig()
    assert cfg.query_expansion == 0
    assert cfg.query_expansion_weight == 0.5
