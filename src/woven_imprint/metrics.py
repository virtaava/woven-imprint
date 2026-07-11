"""Opt-in JSONL metrics sink for per-turn chat metrics.

Enabled by setting `character.metrics_path` in config or the
WOVEN_IMPRINT_METRICS_PATH env var. Each chat() appends one JSON line:
{"ts": ..., "character_id": ..., "metrics": {...}}.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from .log import logger

_UNSET = object()
_sink: object = _UNSET
_lock = threading.Lock()


class MetricsSink:
    def __init__(self, path: str):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()

    def write(self, character_id: str, metrics: dict) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "character_id": character_id,
            "metrics": metrics,
        }
        try:
            with self._write_lock, open(self.path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.debug("Metrics write failed: %s", e)


def get_sink() -> MetricsSink | None:
    """Return the configured sink, or None if metrics are disabled."""
    global _sink
    with _lock:
        if _sink is _UNSET:
            from .config import get_config

            path = get_config().character.metrics_path
            if path:
                try:
                    _sink = MetricsSink(path)
                except Exception as e:
                    logger.debug("Failed to create metrics sink at %s: %s", path, e)
                    _sink = None
            else:
                _sink = None
        return _sink  # type: ignore[return-value]


def reset_sink() -> None:
    """Forget the cached sink (used after config reload / in tests)."""
    global _sink
    with _lock:
        _sink = _UNSET
