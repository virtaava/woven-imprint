from datetime import datetime, timedelta, timezone

import pytest

from woven_imprint import clock
from woven_imprint.memory.facts import FactStore, _norm_time, normalize_key, normalize_object
from woven_imprint.storage.sqlite import SQLiteStorage

T0 = datetime(2026, 5, 3, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def store():
    s = SQLiteStorage(":memory:")
    s.save_character("c1", "Ada", {})
    yield FactStore(s, "c1"), s
    s.close()


def test_schema_version_is_5():
    s = SQLiteStorage(":memory:")
    v = s._conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]
    assert v == 5
    cols = {r[1] for r in s._conn.execute("PRAGMA table_info(facts)")}
    assert {
        "subject",
        "predicate",
        "object",
        "valid_from",
        "valid_to",
        "recorded_at",
        "expired_at",
        "memory_id",
    } <= cols
    rcols = {r[1] for r in s._conn.execute("PRAGMA table_info(relationships)")}
    assert "state" in rcols


def test_normalizers():
    # normalize_key(subject, predicate) -> (norm(subject), norm(predicate)) — order
    # matches the signature and how FactStore.add()/_key() consume the result.
    assert normalize_key(" Has_Cat_Named ", "USER") == ("has_cat_named", "user")
    assert normalize_key("has cat named", "user") == ("has_cat_named", "user")
    assert normalize_object("Pixel.") == "pixel"


def test_add_current_and_supersede(store):
    facts, _ = store
    with clock.override(T0):
        a = facts.add(
            subject="user",
            predicate="has_cat_named",
            object="Pixel",
            statement="The visitor's cat is named Pixel.",
        )
    assert a["valid_from"] == "2026-05-03 12:00:00" and a["valid_to"] is None
    with clock.override(T0 + timedelta(days=40)):
        b = facts.add(
            subject="user",
            predicate="has_cat_named",
            object="Moss",
            statement="The visitor's cat is named Moss.",
        )
        facts.expire(a["id"], valid_to=b["valid_from"], superseded_by=b["id"])
    cur = facts.current(subject="user", predicate="has_cat_named")
    assert [f["object"] for f in cur] == ["Moss"]
    hist = facts.history("user", "has_cat_named")
    assert [f["object"] for f in hist] == ["Pixel", "Moss"]
    assert hist[0]["valid_to"] == b["valid_from"] and hist[0]["superseded_by"] == b["id"]
    assert hist[0]["expired_at"] == "2026-06-12 12:00:00"


def test_as_of_world_time(store):
    facts, _ = store
    with clock.override(T0):
        a = facts.add(
            subject="user", predicate="lives_in", object="Tampere", statement="Lives in Tampere."
        )
    with clock.override(T0 + timedelta(days=30)):
        b = facts.add(
            subject="user",
            predicate="lives_in",
            object="Oulu",
            statement="Lives in Oulu.",
            event_time="2026-06-01",
        )
        facts.expire(a["id"], valid_to=b["valid_from"], superseded_by=b["id"])
    assert b["valid_from"] == "2026-06-01 00:00:00"
    assert [f["object"] for f in facts.as_of("2026-05-20 00:00:00", subject="user")] == ["Tampere"]
    assert [f["object"] for f in facts.as_of("2026-06-15 00:00:00", subject="user")] == ["Oulu"]


def test_find_active_uses_normalized_key(store):
    facts, _ = store
    facts.add(
        subject="User", predicate="Works As", object="a luthier", statement="Works as a luthier."
    )
    assert facts.find_active("user", "works_as")["object"] == "a luthier"
    assert facts.find_active("user", "plays") is None


def test_relationship_state_roundtrip(store):
    _, s = store
    s.save_relationship(
        {
            "id": "rel-1",
            "character_id": "c1",
            "target_id": "t",
            "dimensions": {"trust": 0.1},
            "state": {"updates": 3, "recent": [0.1]},
        }
    )
    rel = s.get_relationship("c1", "t")
    assert rel["state"] == {"updates": 3, "recent": [0.1]}
    assert s.get_relationships("c1")[0]["state"]["updates"] == 3


def test_legacy_relationship_row_has_empty_state(store):
    _, s = store
    s._conn.execute(
        "INSERT INTO relationships (id, character_id, target_id, dimensions) VALUES ('rel-x','c1','u','{}')"
    )
    assert s.get_relationship("c1", "u")["state"] == {}


def test_norm_time_rejects_non_date_10_char_strings():
    # A 10-char string that isn't a YYYY-MM-DD date must fall through to
    # parse_ts and return None on failure, not be treated as a bare date.
    assert _norm_time("yesterday!") is None
    assert _norm_time("2026-05-03") == "2026-05-03 00:00:00"
