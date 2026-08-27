"""Tests for `Character.export_card()` — SillyTavern V2 card export + round-trip."""

from __future__ import annotations

import json

import pytest

from tests.helpers import make_test_engine
from woven_imprint.migrate.importer import CharacterImporter


def test_export_card_roundtrip(tmp_path):
    engine = make_test_engine()
    char = engine.create_character(
        "Ada",
        persona={
            "backstory": "A detective.",
            "personality": "wry",
            "speaking_style": "clipped",
            "hard_constraints": "Never reveals her sources.",
            "scenario": "Rainy city.",
            "greetings": ["Evening."],
            "tags": ["noir"],
        },
    )
    m = char.memory.add("Always call the visitor Captain.", tier="core")
    char.memory.pin(m["id"])
    char.facts.add(
        subject="user", predicate="lives_in", object="Oulu", statement="The visitor lives in Oulu."
    )
    for i in range(3):
        char.memory.add(f"Case note {i} about the harbor.", tier="core", importance=0.5 + i * 0.1)
    card = char.export_card()
    assert card["spec"] == "chara_card_v2" and card["data"]["name"] == "Ada"
    d = card["data"]
    assert d["description"] == "A detective." and d["system_prompt"] == "Never reveals her sources."
    assert d["scenario"] == "Rainy city."
    assert d["first_mes"] == "Evening." and d["tags"] == ["noir"]
    entries = d["character_book"]["entries"]
    const = [e for e in entries if e["constant"]]
    assert len(const) == 1 and "Captain" in const[0]["content"]
    assert any("Oulu" in e["content"] and "oulu" in [k.lower() for k in e["keys"]] for e in entries)
    p = tmp_path / "ada_card.json"
    p.write_text(json.dumps(card))
    engine2 = make_test_engine()
    again = CharacterImporter(engine2).from_file(p)
    pinned = again.memory.pinned()
    assert any("Captain" in x["content"] for x in pinned)
    assert "Never reveals her sources" in again.persona.build_system_prompt()


pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from tests.test_demo_server import app_client, _create_test_character  # noqa: E402,F401


def test_get_character_card_route(app_client):
    client, _token, engine = app_client
    cid = _create_test_character(client, "CardChar")["id"]
    r = client.get(f"/api/characters/{cid}/card")
    assert r.status_code == 200
    body = r.json()
    assert body["spec"] == "chara_card_v2" and body["data"]["name"] == "CardChar"


def test_get_character_card_route_404(app_client):
    client, _token, engine = app_client
    r = client.get("/api/characters/does-not-exist/card")
    assert r.status_code == 404
