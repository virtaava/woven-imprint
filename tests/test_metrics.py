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
    try:
        engine = make_test_engine()
        char = engine.create_character("Metra")
        char.chat("Hello there")
        lines = (tmp_path / "metrics.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["character_id"] == char.id
        assert "total_ms" in rec["metrics"]
        assert "generate_ms" in rec["metrics"]
    finally:
        # cleanup so other tests see defaults
        monkeypatch.delenv("WOVEN_IMPRINT_METRICS_PATH", raising=False)
        reload_config()
        metrics.reset_sink()


def test_chat_with_unwritable_metrics_path_does_not_crash(tmp_path, monkeypatch):
    """Test that chat() does not crash when metrics_path is unwritable."""
    # Create a file, then try to use a path under it as a directory
    file_path = tmp_path / "file.txt"
    file_path.write_text("content")
    impossible_metrics_path = file_path / "x.jsonl"

    # Set metrics path to impossible location
    monkeypatch.setenv("WOVEN_IMPRINT_METRICS_PATH", str(impossible_metrics_path))
    reload_config()
    metrics.reset_sink()

    try:
        # Should not raise, even though metrics sink creation fails
        engine = make_test_engine()
        char = engine.create_character("TestChar")
        response = char.chat("Hello")

        # Should return a response despite unwritable metrics path
        assert response is not None
        assert isinstance(response, str)
    finally:
        # cleanup
        monkeypatch.delenv("WOVEN_IMPRINT_METRICS_PATH", raising=False)
        reload_config()
        metrics.reset_sink()
