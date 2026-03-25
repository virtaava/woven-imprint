"""Persistence regression tests for the demo path."""

import os
import tempfile
import pytest

# Import from shared helpers
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from helpers import make_test_engine, FakeLLM, FakeEmbedder

from woven_imprint.engine import Engine
from woven_imprint.server.services import (
    create_character_service,
    start_session_service,
    end_session_service,
    record_message_service,
    recall_memories_service,
)


def _make_engine_with_path(db_path):
    """Create engine with specific db path (for restart testing)."""
    from helpers import FakeLLM, FakeEmbedder
    llm = FakeLLM()
    embedder = FakeEmbedder()
    engine = Engine(db_path=db_path, llm=llm, embedding=embedder)
    orig = engine.create_character
    def _create_seq(*a, **kw):
        c = orig(*a, **kw)
        c.parallel = False
        return c
    engine.create_character = _create_seq
    return engine


class TestCrossSessionPersistence:
    """Verify that facts survive across sessions and engine restarts."""

    def test_memory_survives_session_restart(self):
        """Fact from session 1 is recalled in session 2."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            # Session 1: create character and record a fact
            engine1 = _make_engine_with_path(db_path)
            r = create_character_service(engine1, "TestChar", None, None)
            char_id = r["id"]
            start_session_service(engine1, char_id)
            record_message_service(engine1, char_id, "user", "My favorite color is blue", None)
            end_session_service(engine1, char_id)
            engine1.close()

            # Session 2: new engine instance, recall the fact
            engine2 = _make_engine_with_path(db_path)
            result = recall_memories_service(engine2, char_id, "favorite color", 10, None)
            contents = [m.get("content", "") for m in result["memories"]]
            assert any("blue" in c.lower() for c in contents), \
                f"Expected 'blue' in recalled memories, got: {contents}"
            engine2.close()
        finally:
            os.unlink(db_path)

    def test_character_survives_restart(self):
        """Character created in session 1 exists in session 2."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        try:
            engine1 = _make_engine_with_path(db_path)
            r = create_character_service(engine1, "Persistent", None, None)
            char_id = r["id"]
            engine1.close()

            engine2 = _make_engine_with_path(db_path)
            chars = engine2.list_characters()
            assert any(c["id"] == char_id for c in chars)
            engine2.close()
        finally:
            os.unlink(db_path)


class TestSeedIdempotency:
    """Verify Meridian seeding is safe to run multiple times."""

    def test_seed_does_not_duplicate(self):
        from woven_imprint.server.demo import _seed_meridian_if_needed

        engine = make_test_engine()
        _seed_meridian_if_needed(engine)
        _seed_meridian_if_needed(engine)  # Second call should be a no-op

        chars = engine.list_characters()
        meridians = [c for c in chars if c["name"].lower() == "meridian"]
        assert len(meridians) == 1
        engine.close()
