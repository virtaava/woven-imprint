from datetime import datetime, timedelta, timezone

from woven_imprint import clock


def test_now_is_aware_utc():
    n = clock.now()
    assert n.tzinfo is not None and n.utcoffset() == timedelta(0)


def test_override_and_advance():
    fixed = datetime(2026, 5, 3, 12, 0, tzinfo=timezone.utc)
    with clock.override(fixed):
        assert clock.now() == fixed
        clock.advance(timedelta(days=2, hours=1))
        assert clock.now() == fixed + timedelta(days=2, hours=1)
        assert clock.today().isoformat() == "2026-05-05"
    assert clock.now() != fixed


def test_advance_on_callable_override_freezes_then_advances():
    fixed = datetime(2026, 6, 1, 8, 0, tzinfo=timezone.utc)
    with clock.override(lambda: fixed):
        assert clock.now() == fixed
        clock.advance(timedelta(hours=5))
        assert clock.now() == fixed + timedelta(hours=5)
        clock.advance(timedelta(hours=1))
        assert clock.now() == fixed + timedelta(hours=6)
    assert clock.now() != fixed


def test_sqlite_roundtrip():
    fixed = datetime(2026, 5, 3, 12, 34, 56, tzinfo=timezone.utc)
    s = clock.sqlite_ts(fixed)
    assert s == "2026-05-03 12:34:56"
    assert clock.parse_ts(s) == fixed
    assert clock.parse_ts("2026-05-03T12:34:56Z") == fixed
    assert clock.parse_ts("2026-05-03 12:34:56.123456") == fixed.replace(microsecond=123456)


def test_relative_phrases():
    now = datetime(2026, 8, 25, tzinfo=timezone.utc)
    rel = lambda **kw: clock.relative(now - timedelta(**kw), now)  # noqa: E731
    assert rel(hours=3) == "today"
    assert rel(days=1) == "yesterday"
    assert rel(days=5) == "5 days ago"
    assert rel(days=21) == "3 weeks ago"
    assert rel(days=70) == "2 months ago"
    assert rel(days=400) == "1 year ago"
    assert rel(days=800) == "2 years ago"


def test_storage_stamps_from_clock():
    from tests.helpers import make_test_engine

    fixed = datetime(2026, 1, 10, 9, 0, tzinfo=timezone.utc)
    engine = make_test_engine()
    char = engine.create_character("T")
    with clock.override(fixed):
        m = char.memory.add("Fixed-time memory", tier="core")
    row = engine.storage.get_memory(m["id"])
    assert row["created_at"] == "2026-01-10 09:00:00"
    assert row["accessed_at"] == "2026-01-10 09:00:00"
    with clock.override(fixed + timedelta(days=3)):
        engine.storage.touch_memory(m["id"])
    assert engine.storage.get_memory(m["id"])["accessed_at"] == "2026-01-13 09:00:00"
