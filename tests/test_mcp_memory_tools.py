"""Tests for the editable-memory/fact MCP tools.

Requires the [mcp] extra. Skipped automatically if not installed.
Calls the tool functions directly (not through the MCP transport) with
`mcp_server._engine` monkeypatched to a test engine.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

sys.path.insert(0, str(Path(__file__).parent))

from helpers import make_test_engine

import woven_imprint.mcp_server as mcp_server


def _tool(fn):
    """Unwrap a FastMCP-decorated tool if it isn't directly callable."""
    return getattr(fn, "fn", fn)


@pytest.fixture()
def test_engine(monkeypatch):
    engine = make_test_engine()
    monkeypatch.setattr(mcp_server, "_engine", engine)
    mcp_server._char_cache.clear()
    yield engine
    mcp_server._char_cache.clear()


def _setup(test_engine):
    char = test_engine.create_character(name="EditChar")
    char.parallel = False
    char.background = False
    m = char.memory.add("The visitor likes tea.", tier="core", importance=0.6)
    return char, m


def test_edit_memory(test_engine):
    char, m = _setup(test_engine)
    out = json.loads(
        _tool(mcp_server.edit_memory)(
            character_id=char.id,
            memory_id=m["id"],
            content="The visitor likes coffee.",
            importance=0.9,
            tier="bedrock",
        )
    )
    assert out["content"] == "The visitor likes coffee."
    assert out["tier"] == "bedrock"
    assert "embedding" not in out


def test_pin_memory_and_list_pinned(test_engine):
    char, m = _setup(test_engine)
    out = json.loads(
        _tool(mcp_server.pin_memory)(character_id=char.id, memory_id=m["id"], pinned=True)
    )
    assert out["metadata"]["pinned"] is True

    pinned = json.loads(_tool(mcp_server.list_pinned)(character_id=char.id))
    assert [x["id"] for x in pinned] == [m["id"]]


def test_delete_memory(test_engine):
    char, m = _setup(test_engine)
    out = json.loads(_tool(mcp_server.delete_memory)(character_id=char.id, memory_id=m["id"]))
    assert out == {"deleted": True}
    out2 = json.loads(_tool(mcp_server.delete_memory)(character_id=char.id, memory_id=m["id"]))
    assert "error" in out2


def test_retract_and_edit_fact(test_engine):
    char, m = _setup(test_engine)
    f = char.facts.add(
        subject="user", predicate="likes", object="tea", statement=m["content"], memory_id=m["id"]
    )

    out = json.loads(
        _tool(mcp_server.edit_fact)(character_id=char.id, fact_id=f["id"], object="coffee")
    )
    assert out["object"] == "coffee"

    out = json.loads(_tool(mcp_server.retract_fact)(character_id=char.id, fact_id=f["id"]))
    assert out["metadata"]["retracted"] is True


def test_recall_exposes_id_usable_by_edit_memory(test_engine):
    char, m = _setup(test_engine)
    recalled = json.loads(_tool(mcp_server.recall)(character_id=char.id, query="tea", limit=5))
    memories = recalled["memories"]
    assert memories and all("id" in r for r in memories)
    recalled_id = next(r["id"] for r in memories if r["id"] == m["id"])

    out = json.loads(
        _tool(mcp_server.edit_memory)(
            character_id=char.id, memory_id=recalled_id, content="The visitor likes coffee."
        )
    )
    assert out["content"] == "The visitor likes coffee."


def test_get_facts_exposes_id_usable_by_retract_fact(test_engine):
    char, m = _setup(test_engine)
    char.facts.add(
        subject="user", predicate="likes", object="tea", statement=m["content"], memory_id=m["id"]
    )

    facts = json.loads(_tool(mcp_server.get_facts)(character_id=char.id))
    assert facts and all(k in facts[0] for k in ("id", "superseded_by", "memory_id"))
    fact_id = facts[0]["id"]

    out = json.loads(_tool(mcp_server.retract_fact)(character_id=char.id, fact_id=fact_id))
    assert out["metadata"]["retracted"] is True


def test_bad_id_error_shape(test_engine):
    char, _m = _setup(test_engine)
    out = json.loads(
        _tool(mcp_server.edit_memory)(character_id=char.id, memory_id="mem-nope", content="x")
    )
    assert "error" in out

    out = json.loads(
        _tool(mcp_server.edit_fact)(character_id=char.id, fact_id="fact-nope", object="x")
    )
    assert "error" in out

    out = json.loads(_tool(mcp_server.retract_fact)(character_id=char.id, fact_id="fact-nope"))
    assert "error" in out

    out = json.loads(_tool(mcp_server.delete_memory)(character_id=char.id, memory_id="mem-nope"))
    assert "error" in out

    out = json.loads(_tool(mcp_server.pin_memory)(character_id=char.id, memory_id="mem-nope"))
    assert "error" in out

    out = json.loads(_tool(mcp_server.list_pinned)(character_id="char-nope"))
    assert "error" in out
