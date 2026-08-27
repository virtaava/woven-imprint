import pytest

from tests.helpers import make_test_engine


def _char():
    engine = make_test_engine()
    return engine, engine.create_character("Ada")


def test_edit_content_reembeds_and_keeps_created_at():
    engine, char = _char()
    m = char.memory.add("The visitor likes tea.", tier="core", importance=0.6)
    before = engine.storage.get_memory(m["id"])
    updated = char.memory.edit(m["id"], content="The visitor likes coffee.")
    row = engine.storage.get_memory(m["id"])
    assert row["content"] == "The visitor likes coffee." and updated["content"] == row["content"]
    assert row["created_at"] == before["created_at"]
    assert row["embedding"] != before["embedding"]
    assert [x["id"] for x in engine.storage.fts_search(char.id, "coffee")] == [m["id"]]
    assert engine.storage.fts_search(char.id, "tea") == []


def test_edit_importance_and_tier():
    engine, char = _char()
    m = char.memory.add("A routine note.", tier="buffer")
    char.memory.edit(m["id"], importance=0.9, tier="core")
    row = engine.storage.get_memory(m["id"])
    assert row["importance"] == 0.9 and row["tier"] == "core"
    with pytest.raises(ValueError):
        char.memory.edit(m["id"], tier="granite")
    with pytest.raises(KeyError):
        char.memory.edit("mem-nope", importance=0.1)


def test_pin_unpin_and_list():
    engine, char = _char()
    a = char.memory.add("Brother died in March.", tier="core")
    b = char.memory.add("Prefers window seats.", tier="core")
    char.memory.pin(a["id"])
    char.memory.pin(b["id"])
    char.memory.pin(b["id"], pinned=False)
    pinned = char.memory.pinned()
    assert [p["id"] for p in pinned] == [a["id"]]
    assert engine.storage.get_memory(a["id"])["metadata"]["pinned"] is True
    assert engine.storage.get_memory(b["id"])["metadata"].get("pinned") is False


def test_delete_removes_row_fts_and_retracts_linked_fact():
    engine, char = _char()
    m = char.memory.add("The visitor's cat is named Pixel.", tier="core")
    f = char.facts.add(
        subject="user",
        predicate="has_cat_named",
        object="Pixel",
        statement=m["content"],
        memory_id=m["id"],
    )
    char.memory.delete(m["id"])
    assert engine.storage.get_memory(m["id"]) is None
    assert engine.storage.fts_search(char.id, "Pixel") == []
    fact = char.facts.get(f["id"])
    assert (
        fact["memory_id"] is None
        and fact["expired_at"] is not None
        and fact["metadata"].get("retracted") is True
    )
    assert char.facts.find_active("user", "has_cat_named") is None
