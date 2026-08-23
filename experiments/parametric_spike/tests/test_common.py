import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import common  # noqa: E402


def test_paths_and_constants():
    assert common.SPIKE_PORT == 11810
    assert common.SPIKE_URL.endswith(":11810/v1")
    assert common.OUT_DIR.name == "out"
    assert common.MINIMAL_SYSTEM == "You are Meridian."


def test_full_system_prompt_mentions_meridian_and_constraints():
    p = common.full_system_prompt()
    assert "Meridian" in p
    assert "Never" in p  # hard constraints flattened in


def test_jsonl_roundtrip(tmp_path):
    path = tmp_path / "x.jsonl"
    rows = [{"a": 1}, {"b": "ä"}]
    common.write_jsonl(path, rows)
    assert common.read_jsonl(path) == rows
