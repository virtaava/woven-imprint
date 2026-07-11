"""Tests for durable conversation buffer (session_turns, migration v3)."""

from tests.helpers import make_test_engine


def test_turns_persisted_and_rehydrated():
    engine = make_test_engine(db_path=":memory:")
    char = engine.create_character("Memo")
    sid = char.start_session()
    char.chat("first message")
    char.chat("second message")

    turns = engine.storage.get_session_turns(sid)
    assert [t["role"] for t in turns] == ["user", "assistant", "user", "assistant"]
    assert turns[0]["content"] == "first message"

    # Simulate cold start: fresh Character object, resume session
    char2 = engine.get_character(char.id)
    assert char2._context.turn_count == 0
    char2.resume_session(sid)
    assert char2._context.turn_count == 4
    msgs = char2._context.get_messages()
    assert msgs[0]["content"] == "first message"


def test_tail_limit():
    engine = make_test_engine(db_path=":memory:")
    char = engine.create_character("Taily")
    sid = char.start_session()
    for i in range(6):
        char.chat(f"msg {i}")
    tail = engine.storage.get_session_turns(sid, tail=4)
    assert len(tail) == 4
    assert tail[-1]["role"] == "assistant"
