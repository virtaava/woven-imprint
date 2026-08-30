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


def test_edit_object_only_derives_statement_by_replacement():
    engine, char = _char()
    m = char.memory.add("The visitor lives in Tampere.", tier="core")
    f = char.facts.add(
        subject="user",
        predicate="lives_in",
        object="Tampere",
        statement=m["content"],
        memory_id=m["id"],
    )
    out = char.facts.edit(f["id"], object="Oulu")
    assert out["object"] == "Oulu"
    assert out["statement"] == "The visitor lives in Oulu."
    assert engine.storage.get_memory(m["id"])["content"] == "The visitor lives in Oulu."
    assert engine.storage.fts_search(char.id, "Oulu")


def test_edit_object_only_falls_back_when_object_not_in_statement():
    engine, char = _char()
    m = char.memory.add("The visitor mentioned their favorite color once.", tier="core")
    f = char.facts.add(
        subject="user",
        predicate="favorite_color",
        object="blue",
        statement=m["content"],
        memory_id=m["id"],
    )
    out = char.facts.edit(f["id"], object="green")
    assert out["object"] == "green"
    assert out["statement"] == ("The visitor mentioned their favorite color once — now: green.")
    assert engine.storage.get_memory(m["id"])["content"] == out["statement"]


def test_edit_object_only_uses_word_boundaries_not_substring():
    engine, char = _char()
    m = char.memory.add("The kettle lets off steam; the visitor likes tea.", tier="core")
    f = char.facts.add(
        subject="user",
        predicate="likes",
        object="tea",
        statement=m["content"],
        memory_id=m["id"],
    )
    out = char.facts.edit(f["id"], object="coffee")
    assert out["object"] == "coffee"
    assert out["statement"] == "The kettle lets off steam; the visitor likes coffee."
    assert "scoffeem" not in out["statement"]
    assert engine.storage.get_memory(m["id"])["content"] == out["statement"]


def test_edit_object_only_falls_back_when_no_bounded_match():
    engine, char = _char()
    m = char.memory.add("The kettle lets off steam.", tier="core")
    f = char.facts.add(
        subject="user",
        predicate="likes",
        object="tea",
        statement=m["content"],
        memory_id=m["id"],
    )
    out = char.facts.edit(f["id"], object="coffee")
    assert out["object"] == "coffee"
    assert out["statement"] == "The kettle lets off steam — now: coffee."
    assert engine.storage.get_memory(m["id"])["content"] == out["statement"]


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


def test_retract_leaves_memory_active_while_another_fact_still_references_it():
    """Two facts can end up pointing at the same memory row (semantic dedup —
    see MemoryStore.add's dedup_similarity path). Retracting one of them must
    not archive a memory the other one still actively references."""
    engine, char = _char()
    m = char.memory.add("The visitor plays chess and also enjoys go.", tier="core")
    f1 = char.facts.add(
        subject="user", predicate="plays", object="chess", statement=m["content"], memory_id=m["id"]
    )
    f2 = char.facts.add(
        subject="user", predicate="enjoys", object="go", statement=m["content"], memory_id=m["id"]
    )

    out1 = char.facts.retract(f1["id"])
    assert out1["metadata"]["retracted"] is True
    assert engine.storage.get_memory(m["id"])["status"] == "active"
    # The still-active fact's link is untouched.
    assert char.facts.get(f2["id"])["memory_id"] == m["id"]

    out2 = char.facts.retract(f2["id"])
    assert out2["metadata"]["retracted"] is True
    assert engine.storage.get_memory(m["id"])["status"] == "archived"


def test_edit_content_change_on_shared_memory_relinks_instead_of_rewriting():
    """Two facts share one memory row (semantic dedup — see MemoryStore.add's
    dedup_similarity path). Editing ONE fact's statement must not rewrite the
    shared row out from under the other still-active fact: it creates a fresh
    memory for the edited fact's new statement and relinks that fact to it."""
    engine, char = _char()
    m = char.memory.add("The visitor plays chess and also enjoys go.", tier="core")
    f1 = char.facts.add(
        subject="user", predicate="plays", object="chess", statement=m["content"], memory_id=m["id"]
    )
    f2 = char.facts.add(
        subject="user", predicate="enjoys", object="go", statement=m["content"], memory_id=m["id"]
    )

    out1 = char.facts.edit(f1["id"], object="shogi", statement="The visitor plays shogi.")

    assert out1["object"] == "shogi"
    assert out1["memory_id"] != m["id"]  # relinked to a fresh row
    new_mem = engine.storage.get_memory(out1["memory_id"])
    assert new_mem is not None and new_mem["content"] == "The visitor plays shogi."

    # The shared row itself is untouched — f2's link and the memory's own
    # content still read exactly as they did before f1's edit.
    shared_mem = engine.storage.get_memory(m["id"])
    assert shared_mem["content"] == "The visitor plays chess and also enjoys go."
    assert char.facts.get(f2["id"])["memory_id"] == m["id"]


def test_edit_content_change_on_unshared_memory_still_rewrites_in_place():
    """Control: when the linked memory is referenced by only the fact being
    edited (the common case), `edit()` keeps rewriting the row in place —
    the shared-row guard must not change behavior for the non-shared case."""
    engine, char = _char()
    m = char.memory.add("The visitor plays chess.", tier="core")
    f = char.facts.add(
        subject="user", predicate="plays", object="chess", statement=m["content"], memory_id=m["id"]
    )

    out = char.facts.edit(f["id"], object="shogi", statement="The visitor plays shogi.")

    assert out["memory_id"] == m["id"]  # same row, rewritten in place
    assert engine.storage.get_memory(m["id"])["content"] == "The visitor plays shogi."


def test_memory_delete_cascade_retracts_all_shared_facts():
    """MemoryStore.delete() hard-deletes a memory and retracts every fact still
    pointing at it — including when more than one fact shares the row."""
    engine, char = _char()
    m = char.memory.add("The visitor plays chess and also enjoys go.", tier="core")
    f1 = char.facts.add(
        subject="user", predicate="plays", object="chess", statement=m["content"], memory_id=m["id"]
    )
    f2 = char.facts.add(
        subject="user", predicate="enjoys", object="go", statement=m["content"], memory_id=m["id"]
    )

    char.memory.delete(m["id"])

    assert engine.storage.get_memory(m["id"]) is None
    r1 = char.facts.get(f1["id"])
    r2 = char.facts.get(f2["id"])
    assert r1["metadata"]["retracted"] is True and r1["memory_id"] is None
    assert r2["metadata"]["retracted"] is True and r2["memory_id"] is None
