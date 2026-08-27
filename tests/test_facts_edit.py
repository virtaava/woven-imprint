from tests.helpers import make_test_engine


def _char():
    engine = make_test_engine()
    return engine, engine.create_character("Ada")


def test_edit_fact_updates_record_and_linked_memory():
    engine, char = _char()
    m = char.memory.add("The visitor lives in Tampere.", tier="core")
    f = char.facts.add(
        subject="user",
        predicate="lives_in",
        object="Tampere",
        statement=m["content"],
        memory_id=m["id"],
    )
    out = char.facts.edit(f["id"], object="Oulu", statement="The visitor lives in Oulu.")
    assert out["object"] == "Oulu"
    assert engine.storage.get_memory(m["id"])["content"] == "The visitor lives in Oulu."
    assert engine.storage.fts_search(char.id, "Oulu")


def test_retract_expires_without_successor_and_archives_memory():
    engine, char = _char()
    m = char.memory.add("The visitor plays chess.", tier="core")
    f = char.facts.add(
        subject="user", predicate="plays", object="chess", statement=m["content"], memory_id=m["id"]
    )
    out = char.facts.retract(f["id"])
    assert (
        out["valid_to"] is not None
        and out["superseded_by"] is None
        and out["metadata"]["retracted"] is True
    )
    assert char.facts.current("user", "plays") == []
    assert engine.storage.get_memory(m["id"])["status"] == "archived"


def test_delete_fact_only():
    engine, char = _char()
    m = char.memory.add("Likes rye bread.", tier="core")
    f = char.facts.add(
        subject="user",
        predicate="likes",
        object="rye bread",
        statement=m["content"],
        memory_id=m["id"],
    )
    char.facts.delete(f["id"])
    assert char.facts.get(f["id"]) is None
    assert engine.storage.get_memory(m["id"]) is not None
