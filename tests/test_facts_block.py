from datetime import datetime, timedelta, timezone

from tests.helpers import make_test_engine
from woven_imprint import clock

T0 = datetime(2026, 5, 3, 12, 0, tzinfo=timezone.utc)


def _char():
    engine = make_test_engine()
    char = engine.create_character("Ada")
    char.background = False
    char.parallel = False
    char.enforce_consistency = False
    return engine, char


def test_facts_block_lists_current_facts_with_previous():
    engine, char = _char()
    with clock.override(T0):
        a = char.facts.add(
            subject="user",
            predicate="has_cat_named",
            object="Pixel",
            statement="The visitor's cat is named Pixel.",
            user_id="toni",
        )
        char.facts.add(
            subject="self",
            predicate="fears",
            object="forgetting",
            statement="I fear forgetting.",
        )
    with clock.override(T0 + timedelta(days=40)):
        b = char.facts.add(
            subject="user",
            predicate="has_cat_named",
            object="Moss",
            statement="The visitor's cat is named Moss.",
            user_id="toni",
        )
        char.facts.expire(a["id"], valid_to=b["valid_from"], superseded_by=b["id"])
        text = char._format_facts_block("toni")
    assert "What you currently know about toni" in text
    assert "(since 2026-06-12, previously: Pixel) The visitor's cat is named Moss." in text
    assert "Things you have said about yourself" in text and "I fear forgetting." in text


def test_facts_block_in_volatile_context_and_gated():
    from woven_imprint.config import get_config

    engine, char = _char()
    char.facts.add(
        subject="user",
        predicate="lives_in",
        object="Oulu",
        statement="The visitor lives in Oulu.",
        user_id="toni",
    )
    char.chat("hi", user_id="toni")
    msgs = char.last_chat_messages
    volatile = msgs[1]["content"]
    assert "What you currently know about toni" in volatile and "Oulu" in volatile
    get_config().context.facts_block = False
    try:
        char.chat("again", user_id="toni")
        assert "What you currently know" not in char.last_chat_messages[1]["content"]
    finally:
        get_config().context.facts_block = True


def test_facts_block_keeps_newest_when_over_the_cap():
    # 14 same-importance facts recorded on 14 successive days: the block (cap
    # 12) must keep the 12 newest and drop the 2 oldest, not the reverse.
    engine, char = _char()
    for day in range(14):
        with clock.override(T0 + timedelta(days=day)):
            char.facts.add(
                subject="user",
                predicate=f"fact_{day}",
                object=f"value{day}",
                statement=f"Fact number {day}.",
                user_id="toni",
                importance=0.75,
            )
    text = char._format_facts_block("toni")
    for day in range(2):
        assert f"Fact number {day}." not in text
    for day in range(2, 14):
        assert f"Fact number {day}." in text


def test_export_import_roundtrip_facts(tmp_path):
    engine, char = _char()
    char.facts.add(
        subject="user",
        predicate="lives_in",
        object="Oulu",
        statement="The visitor lives in Oulu.",
    )
    data = char.export(str(tmp_path / "c.json"))
    assert data["facts"][0]["predicate"] == "lives_in"
    engine2 = make_test_engine()
    char2 = engine2.import_character(str(tmp_path / "c.json"))
    assert char2.facts.find_active("user", "lives_in")["object"] == "Oulu"
