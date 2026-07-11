import json

from woven_imprint import metrics
from woven_imprint.config import reload_config
from tests.helpers import make_test_engine


def _configure_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("WOVEN_IMPRINT_METRICS_PATH", str(tmp_path / "metrics.jsonl"))
    reload_config()
    metrics.reset_sink()


def test_sink_disabled_by_default(monkeypatch):
    monkeypatch.delenv("WOVEN_IMPRINT_METRICS_PATH", raising=False)
    reload_config()
    metrics.reset_sink()
    assert metrics.get_sink() is None


def test_chat_writes_metrics_line(tmp_path, monkeypatch):
    _configure_metrics(tmp_path, monkeypatch)
    engine = make_test_engine()
    char = engine.create_character("Metra")
    char.chat("Hello there")
    lines = (tmp_path / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["character_id"] == char.id
    assert "total_ms" in rec["metrics"]
    assert "generate_ms" in rec["metrics"]
    # cleanup so other tests see defaults
    monkeypatch.delenv("WOVEN_IMPRINT_METRICS_PATH", raising=False)
    reload_config()
    metrics.reset_sink()
