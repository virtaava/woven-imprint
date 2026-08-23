import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE = "http://127.0.0.1:11810"


def _up():
    try:
        return requests.get(BASE + "/health", timeout=2).ok
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _up(), reason="spike server not running (serve.sh)")


def test_adapters_listed_and_toggle():
    import lora

    names = {Path(a["path"]).stem for a in lora.list_adapters(BASE)}
    assert {"persona", "facts"} <= names
    state = {Path(a["path"]).stem: a["scale"] for a in lora.set_adapters({"persona": 1.0}, BASE)}
    assert state["persona"] == 1.0 and state["facts"] == 0.0
    state = {Path(a["path"]).stem: a["scale"] for a in lora.set_adapters({}, BASE)}
    assert state["persona"] == 0.0 and state["facts"] == 0.0
