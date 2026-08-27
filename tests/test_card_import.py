"""Tests for SillyTavern character card import — lorebook, scenario, greetings, PNG V3."""

import base64
import json
import struct
import zlib

from tests.helpers import make_test_engine
from woven_imprint.migrate.importer import CharacterImporter
from woven_imprint.migrate.parsers import parse_tavernai_card

CARD = {
    "spec": "chara_card_v2",
    "spec_version": "2.0",
    "data": {
        "name": "Vesper",
        "description": "A lighthouse keeper on a cold coast.",
        "personality": "gruff, loyal",
        "scenario": "The visitor arrives during a storm.",
        "first_mes": "You made it through the storm, then.",
        "alternate_greetings": ["Shut the door behind you."],
        "mes_example": "<START>\n{{user}}: Hello\n{{char}}: Aye.",
        "system_prompt": "Never leave the lighthouse.",
        "creator_notes": "test card",
        "tags": ["fantasy", "sea"],
        "character_book": {
            "name": "Coast lore",
            "entries": [
                {
                    "keys": ["lantern", "lamp"],
                    "content": "The great lantern was forged in Oulu.",
                    "enabled": True,
                    "constant": False,
                    "insertion_order": 1,
                },
                {
                    "keys": ["oath"],
                    "content": "Vesper swore never to abandon the light.",
                    "enabled": True,
                    "constant": True,
                    "insertion_order": 0,
                },
                {
                    "keys": ["disabled"],
                    "content": "should not import",
                    "enabled": False,
                    "constant": False,
                    "insertion_order": 2,
                },
            ],
        },
    },
}


def _png_with_chara(card: dict, keyword: bytes = b"chara") -> bytes:
    def chunk(t: bytes, d: bytes) -> bytes:
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00\x00\x00")
    payload = base64.b64encode(json.dumps(card).encode())
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"tEXt", keyword + b"\x00" + payload)
        + chunk(b"IDAT", idat)
        + chunk(b"IEND", b"")
    )


def test_parse_json_card_reads_book_and_extra_fields(tmp_path):
    p = tmp_path / "vesper.json"
    p.write_text(json.dumps(CARD))
    parsed = parse_tavernai_card(p)
    card = parsed["card"]
    assert (
        card["scenario"].startswith("The visitor")
        and card["system_prompt"] == "Never leave the lighthouse."
    )
    assert card["alternate_greetings"] == ["Shut the door behind you."]
    assert len(card["character_book"]["entries"]) == 3 and parsed["spec"] == "chara_card_v2"


def test_parse_png_card_v2_and_v3(tmp_path):
    p2 = tmp_path / "v2.png"
    p2.write_bytes(_png_with_chara(CARD))
    assert parse_tavernai_card(p2)["card"]["name"] == "Vesper"
    v3 = {
        "spec": "chara_card_v3",
        "spec_version": "3.0",
        "data": {**CARD["data"], "name": "Vesper3"},
    }
    p3 = tmp_path / "v3.png"
    p3.write_bytes(_png_with_chara(v3, keyword=b"ccv3"))
    assert parse_tavernai_card(p3)["card"]["name"] == "Vesper3"


def test_import_seeds_lorebook_and_persona(tmp_path):
    p = tmp_path / "vesper.json"
    p.write_text(json.dumps(CARD))
    engine = make_test_engine()
    char = CharacterImporter(engine).from_file(p)
    assert char.persona.soft.get("scenario", "").startswith("The visitor")
    assert "Never leave the lighthouse" in char.persona.build_system_prompt()
    mems = char.memory.get_all(limit=None)
    lore = [m for m in mems if m["metadata"].get("source") == "lorebook"]
    assert len(lore) == 2
    constant = next(m for m in lore if "oath" in m["content"])
    assert (
        constant["tier"] == "bedrock"
        and constant["metadata"]["pinned"] is True
        and constant["metadata"]["lorebook_keys"] == ["oath"]
    )
    keyed = next(m for m in lore if "lantern" in m["content"])
    assert keyed["tier"] == "core" and keyed["content"].startswith("[Lore: lantern, lamp]")
    assert char.persona.soft.get("greetings") == [
        "You made it through the storm, then.",
        "Shut the door behind you.",
    ]


EDGE_CASE_CARD = {
    "spec": "chara_card_v2",
    "spec_version": "2.0",
    "data": {
        "name": "Vesper",
        "description": "A lighthouse keeper on a cold coast.",
        "personality": "gruff, loyal",
        "first_mes": "You made it through the storm, then.",
        "character_book": {
            "name": "Coast lore",
            "entries": [
                {
                    # No "enabled" key at all — SillyTavern convention: default enabled.
                    "keys": ["default-enabled"],
                    "content": "Import me anyway.",
                    "constant": False,
                    "insertion_order": 0,
                },
                {
                    # Hand-edited card: keys as a comma-separated string.
                    "keys": "lamp, lantern",
                    "content": "Two keys as a string.",
                    "enabled": True,
                    "constant": False,
                    "insertion_order": 1,
                },
                {
                    # No keys at all — should render without an empty bracket.
                    "keys": [],
                    "content": "No specific keyword.",
                    "enabled": True,
                    "constant": False,
                    "insertion_order": 2,
                },
            ],
        },
    },
}


def test_import_lorebook_entry_without_enabled_key_defaults_to_enabled(tmp_path):
    p = tmp_path / "edge.json"
    p.write_text(json.dumps(EDGE_CASE_CARD))
    engine = make_test_engine()
    char = CharacterImporter(engine).from_file(p)
    mems = char.memory.get_all(limit=None)
    lore = [m for m in mems if m["metadata"].get("source") == "lorebook"]
    assert any("Import me anyway." in m["content"] for m in lore)


def test_import_lorebook_entry_with_string_keys_is_split_on_comma(tmp_path):
    p = tmp_path / "edge.json"
    p.write_text(json.dumps(EDGE_CASE_CARD))
    engine = make_test_engine()
    char = CharacterImporter(engine).from_file(p)
    mems = char.memory.get_all(limit=None)
    lore = [m for m in mems if m["metadata"].get("source") == "lorebook"]
    entry = next(m for m in lore if "Two keys as a string." in m["content"])
    assert entry["metadata"]["lorebook_keys"] == ["lamp", "lantern"]
    assert entry["content"].startswith("[Lore: lamp, lantern]")


def test_import_lorebook_entry_with_no_keys_renders_without_empty_brackets(tmp_path):
    p = tmp_path / "edge.json"
    p.write_text(json.dumps(EDGE_CASE_CARD))
    engine = make_test_engine()
    char = CharacterImporter(engine).from_file(p)
    mems = char.memory.get_all(limit=None)
    lore = [m for m in mems if m["metadata"].get("source") == "lorebook"]
    entry = next(m for m in lore if "No specific keyword." in m["content"])
    assert entry["content"] == "[Lore] No specific keyword."
    assert entry["metadata"]["lorebook_keys"] == []
