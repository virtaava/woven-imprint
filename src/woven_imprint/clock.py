"""Injectable wall clock. Every timestamp the library writes or compares comes from here.

Production: real UTC time. Tests/benchmarks: `override()` a fixed datetime (or a callable)
and `advance()` it to simulate days passing.
"""

from __future__ import annotations

import contextlib
import threading
from datetime import date, datetime, timedelta, timezone
from typing import Callable

_lock = threading.Lock()
_override: datetime | Callable[[], datetime] | None = None

SQLITE_FMT = "%Y-%m-%d %H:%M:%S"


def now() -> datetime:
    """Current time, timezone-aware UTC."""
    with _lock:
        ov = _override
    if ov is None:
        return datetime.now(timezone.utc)
    value = ov() if callable(ov) else ov
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def today() -> date:
    return now().date()


class _Override(contextlib.AbstractContextManager):
    def __init__(self, previous):
        self._previous = previous

    def __exit__(self, *exc):
        global _override
        with _lock:
            _override = self._previous
        return False


def override(value: datetime | Callable[[], datetime] | None):
    """Set the clock. Usable as a context manager or a plain call (pass None to clear)."""
    global _override
    with _lock:
        previous = _override
        _override = value
    return _Override(previous)


def advance(delta: timedelta) -> datetime:
    """Move an overridden clock forward. Raises if the clock is not overridden."""
    global _override
    with _lock:
        if _override is None:
            raise RuntimeError("clock.advance() requires an active override")
        current = _override() if callable(_override) else _override
        current = current if current.tzinfo else current.replace(tzinfo=timezone.utc)
        _override = current + delta
        return _override


def sqlite_ts(dt: datetime | None = None) -> str:
    dt = dt or now()
    if dt.tzinfo:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime(SQLITE_FMT)


def parse_ts(raw: str | datetime) -> datetime:
    """Parse SQLite/ISO timestamps; naive values are UTC."""
    if isinstance(raw, datetime):
        dt = raw
    else:
        s = str(raw).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def relative(dt: datetime | str, now_: datetime | None = None) -> str:
    """Human phrase for how long ago `dt` was: today, yesterday, N days/weeks/months/years ago.

    Based on elapsed wall-clock time (not calendar-date subtraction), so "3 hours
    ago" reads as "today" even when the reference moment is just past local midnight.
    """
    then = parse_ts(dt)
    ref = now_ or now()
    if ref.tzinfo is None:
        ref = ref.replace(tzinfo=timezone.utc)
    days = (ref - then).days
    if days <= 0:
        return "today"
    if days == 1:
        return "yesterday"
    if days < 14:
        return f"{days} days ago"
    if days < 56:
        return f"{days // 7} weeks ago"
    if days < 365:
        months = max(2, round(days / 30.44))
        return f"{months} months ago"
    years = days // 365
    return f"{years} year ago" if years == 1 else f"{years} years ago"
