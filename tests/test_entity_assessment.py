"""Tier 3o: `entities` field in the unified turn assessment.

Spec: docs/superpowers/specs/2026-09-04-tier3o-entity-linking.md

The unified JSON gains an optional `entities` list (<=8 short canonical
names). Parsing is fault-tolerant like every other section: absent,
malformed, or non-list payloads yield [] and never raise.
"""

from __future__ import annotations

from woven_imprint.persona.assessment import TurnAssessor


def test_parse_entities_happy_path():
    out = TurnAssessor._parse_entities(["Rocket", "Caroline", "San Diego"])
    assert out == ["Rocket", "Caroline", "San Diego"]


def test_parse_entities_cleans_dedups_caps():
    raw = ["  Rocket ", "rocket", "", 42, None, "A", "B", "C", "D", "E", "F", "G", "H"]
    out = TurnAssessor._parse_entities(raw)
    assert out[0] == "Rocket"          # first-seen casing kept, stripped
    assert "rocket" not in out[1:]     # case-fold dedup
    assert 42 not in out and None not in out and "" not in out
    assert len(out) <= 8


def test_parse_entities_non_list_returns_empty():
    assert TurnAssessor._parse_entities("Rocket") == []
    assert TurnAssessor._parse_entities({"e": 1}) == []
    assert TurnAssessor._parse_entities(None) == []
