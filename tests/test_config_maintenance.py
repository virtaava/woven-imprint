"""Tests for MaintenanceConfig."""

from woven_imprint.config import WovenConfig, reload_config


def test_maintenance_defaults():
    cfg = WovenConfig()
    m = cfg.maintenance
    assert m.max_llm_calls_per_run == 50
    assert m.consolidate_chunk_size == 500
    assert m.buffer_ttl_days == 14
    assert m.buffer_hygiene_max_importance == 0.55
    assert m.importance_scoring_batch == 30
    assert m.dedup_scan_limit == 200
    assert m.dedup_similarity == 0.92
    assert m.reinforce_similarity == 0.85
    assert m.contradiction_candidate_similarity == 0.70
    assert m.contradiction_max_pairs == 10
    assert m.reflect_importance_sum == 12.0
    assert m.callbacks_refresh_limit == 5
    assert m.callbacks_ready_cap == 10
    assert m.callbacks_refresh_on_session_end is True


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("WOVEN_IMPRINT_MAINTENANCE_BUDGET", "7")
    monkeypatch.setenv("WOVEN_IMPRINT_EMBEDDING_BASE_URL", "http://127.0.0.1:11801/v1")
    try:
        cfg = reload_config()
        assert cfg.maintenance.max_llm_calls_per_run == 7
        assert cfg.llm.embedding_base_url == "http://127.0.0.1:11801/v1"
    finally:
        monkeypatch.delenv("WOVEN_IMPRINT_MAINTENANCE_BUDGET")
        monkeypatch.delenv("WOVEN_IMPRINT_EMBEDDING_BASE_URL")
        reload_config()
