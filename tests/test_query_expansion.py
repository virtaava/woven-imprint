"""Tier 3i: LLM-guided query expansion in `MemoryRetriever.retrieve`.

Spec: docs/superpowers/specs/2026-09-02-tier3i-llm-query-expansion.md

`query_expansion` (default 0 = off) rewrites the query into instance-level
search queries via one LLM JSON call; each expansion adds one embedding +
one fts_search and two extra weighted-RRF ranked lists. Off path is
byte-identical; every failure mode degrades silently to the unexpanded
ranking.
"""

from __future__ import annotations

import pytest

from woven_imprint.memory.retrieval import _generate_expansions


class FakeExpansionLLM:
    """Returns a canned generate_json_robust payload; counts calls."""

    def __init__(self, payload=None, exc: Exception | None = None):
        self.payload = payload if payload is not None else []
        self.exc = exc
        self.calls = 0
        self.last_messages = None

    def generate_json_robust(self, messages, temperature=0.3):
        self.calls += 1
        self.last_messages = messages
        if self.exc is not None:
            raise self.exc
        return self.payload


def test_generate_expansions_happy_path():
    llm = FakeExpansionLLM(["gallery opening attended", "museum visit", "art exhibit"])
    out = _generate_expansions(llm, "How many art events did I attend?", 3)
    assert out == ["gallery opening attended", "museum visit", "art exhibit"]
    assert llm.calls == 1


def test_generate_expansions_caps_dedups_and_drops_originals():
    llm = FakeExpansionLLM(
        [
            "  Museum Visit ",
            "museum visit",          # case-fold duplicate after strip
            "How many art events did I attend?",  # equals original -> dropped
            "",                       # empty -> dropped
            "concert attended",
            "play attended",          # beyond n=2 cap once dups removed
        ]
    )
    out = _generate_expansions(llm, "How many art events did I attend?", 2)
    assert out == ["Museum Visit", "concert attended"]


@pytest.mark.parametrize(
    "bad",
    [
        {"queries": ["a"]},   # dict, not list
        [1, 2, 3],             # non-string items
        "not json-list",       # plain string
    ],
)
def test_generate_expansions_garbage_payload_returns_empty(bad):
    llm = FakeExpansionLLM(bad)
    assert _generate_expansions(llm, "question", 3) == []


def test_generate_expansions_llm_error_returns_empty():
    llm = FakeExpansionLLM(exc=ValueError("no JSON"))
    assert _generate_expansions(llm, "question", 3) == []
    llm2 = FakeExpansionLLM(exc=RuntimeError("connection refused"))
    assert _generate_expansions(llm2, "question", 3) == []
