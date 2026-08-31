from datetime import datetime, timedelta, timezone

from tests.helpers import make_test_engine
from woven_imprint import clock

T0 = datetime(2026, 8, 25, 10, 0, tzinfo=timezone.utc)


def _char():
    engine = make_test_engine()
    return engine, engine.create_character("Ada")


def test_format_memories_has_date_and_relative():
    engine, char = _char()
    with clock.override(T0 - timedelta(days=21)):
        char.memory.add("The visitor fixed the harbor lamp.", tier="core")
    with clock.override(T0):
        mems = char.retriever.retrieve("harbor lamp", limit=5)
        text = char._format_memories(mems)
    assert "(2026-08-04 Tue, 3 weeks ago)" in text
    assert "harbor lamp" in text


def test_volatile_block_has_today_line():
    engine, char = _char()
    with clock.override(T0):
        char.chat("hello")
        messages = char._build_context("hi again", [], "")
    volatile = (
        messages[1]["content"] if len(messages) > 1 and messages[1]["role"] == "system" else ""
    )
    assert "Today is Tuesday, 2026-08-25." in (volatile or messages[0]["content"])


def test_session_summary_is_dated():
    engine, char = _char()
    with clock.override(T0):
        char.start_session()
        char.chat("I moved to Oulu last spring.")
        char.end_session()
    summaries = [
        m for m in char.memory.get_all(tier="core") if m["content"].startswith("[Session Summary")
    ]
    assert summaries, "no summary stored"
    s = summaries[0]
    assert s["content"].startswith("[Session Summary 2026-08-25]")
    assert s["metadata"]["type"] == "session_summary"
    assert s["metadata"]["started_at"].startswith("2026-08-25")


def test_consolidation_keeps_date_range():
    from woven_imprint.memory.consolidation import ConsolidationEngine

    engine, char = _char()
    # ConsolidationEngine.consolidate() requires >= 10 buffer rows to run at
    # all — 3 identical-content entries per day clears that gate while
    # keeping the date range at exactly days 30..27 (identical FakeEmbedder
    # vectors also guarantees they all land in one cluster).
    for d in (30, 29, 28, 27):
        for _ in range(3):
            with clock.override(T0 - timedelta(days=d)):
                char.memory.add(
                    "The visitor talked about sailing boats on the lake.", tier="buffer"
                )
    with clock.override(T0):
        ConsolidationEngine(engine.storage, engine.llm, engine.embedding, char.id).consolidate()
    cons = [
        m for m in char.memory.get_all(tier="core") if m["content"].startswith("[Consolidated]")
    ]
    assert cons
    meta = cons[0]["metadata"]
    assert meta["date_range"][0].startswith("2026-07-26") and meta["date_range"][1].startswith(
        "2026-07-29"
    )
    assert cons[0]["created_at"].startswith("2026-07-29")
