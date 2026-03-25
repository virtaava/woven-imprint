"""Tests for service-layer functions (woven_imprint.server.services)."""

import pytest

from tests.helpers import make_test_engine
from woven_imprint.server.services import (
    create_character_service,
    delete_character_service,
    export_character_service,
    import_character_service,
    list_characters_service,
    get_character_state_service,
    recall_memories_service,
    record_message_service,
    reflect_character_service,
    start_session_service,
    end_session_service,
    get_relationship_service,
    find_character_by_name_or_id,
    extract_last_user_message,
    extract_user_id_from_messages,
)


@pytest.fixture
def engine():
    return make_test_engine()


# ── create_character_service ──────────────────────────────────────────


class TestCreateCharacter:
    def test_creates_new_character(self, engine):
        result = create_character_service(engine, "Alice", {"personality": "kind"}, None)
        assert result["created"] is True
        assert result["name"] == "Alice"
        assert result["id"]

    def test_deduplicates_by_name_case_insensitive(self, engine):
        first = create_character_service(engine, "Alice", {}, None)
        second = create_character_service(engine, "alice", {}, None)
        assert second["created"] is False
        assert second["id"] == first["id"]

    def test_string_persona_normalized_to_personality(self, engine):
        result = create_character_service(engine, "Bob", "cheerful and loud", None)
        assert result["created"] is True
        # Verify the character was actually created with the persona
        chars = list_characters_service(engine)
        assert any(c["id"] == result["id"] for c in chars)

    def test_dict_persona_passed_through(self, engine):
        persona = {"personality": "shy", "backstory": "Grew up in a forest"}
        result = create_character_service(engine, "Carol", persona, None)
        assert result["created"] is True

    def test_none_persona_uses_empty_dict(self, engine):
        result = create_character_service(engine, "Dave", None, None)
        assert result["created"] is True


# ── list_characters_service ───────────────────────────────────────────


class TestListCharacters:
    def test_empty_initially(self, engine):
        assert list_characters_service(engine) == []

    def test_lists_created_characters(self, engine):
        create_character_service(engine, "Alice", {}, None)
        create_character_service(engine, "Bob", {}, None)
        chars = list_characters_service(engine)
        names = {c["name"] for c in chars}
        assert names == {"Alice", "Bob"}


# ── get_character_state_service ───────────────────────────────────────


class TestGetCharacterState:
    def test_returns_state_dict(self, engine):
        created = create_character_service(engine, "Alice", {}, None)
        state = get_character_state_service(engine, created["id"])
        assert state["id"] == created["id"]
        assert state["name"] == "Alice"
        assert "emotion" in state
        assert "arc" in state

    def test_raises_keyerror_for_unknown_id(self, engine):
        with pytest.raises(KeyError):
            get_character_state_service(engine, "nonexistent-id")


# ── start_session / end_session ───────────────────────────────────────


class TestSessionLifecycle:
    def test_start_session_returns_session_id(self, engine):
        created = create_character_service(engine, "Alice", {}, None)
        result = start_session_service(engine, created["id"])
        assert "session_id" in result
        assert result["session_id"]

    def test_end_session_returns_summary(self, engine):
        created = create_character_service(engine, "Alice", {}, None)
        start_session_service(engine, created["id"])
        result = end_session_service(engine, created["id"])
        assert "summary" in result

    def test_start_raises_for_unknown(self, engine):
        with pytest.raises(KeyError):
            start_session_service(engine, "no-such-char")

    def test_end_raises_for_unknown(self, engine):
        with pytest.raises(KeyError):
            end_session_service(engine, "no-such-char")


# ── record_message_service ────────────────────────────────────────────


class TestRecordMessage:
    def test_records_user_message(self, engine):
        created = create_character_service(engine, "Alice", {}, None)
        start_session_service(engine, created["id"])
        # Should not raise
        record_message_service(engine, created["id"], "user", "Hello!", "toni")

    def test_records_assistant_message(self, engine):
        created = create_character_service(engine, "Alice", {}, None)
        start_session_service(engine, created["id"])
        record_message_service(engine, created["id"], "assistant", "Hi there!", None)

    def test_rejects_invalid_role_strict(self, engine):
        created = create_character_service(engine, "Alice", {}, None)
        with pytest.raises(ValueError, match="Invalid role"):
            record_message_service(engine, created["id"], "system", "test", None)

    def test_rejects_character_role_strict(self, engine):
        """strict_roles=True (sidecar compat) rejects 'character' role."""
        created = create_character_service(engine, "Alice", {}, None)
        with pytest.raises(ValueError, match="Invalid role"):
            record_message_service(engine, created["id"], "character", "test", None)

    def test_non_strict_passes_service_validation_for_system(self, engine):
        """strict_roles=False allows broader roles at the service layer.
        The underlying Character.ingest() still only accepts user/assistant,
        so this verifies the service-layer check passes (ValueError from
        Character, not from the service).
        """
        created = create_character_service(engine, "Alice", {}, None)
        start_session_service(engine, created["id"])
        # Service layer allows 'system' in non-strict mode, but
        # Character.ingest() still raises — the error message differs
        with pytest.raises(ValueError, match="role must be"):
            record_message_service(
                engine, created["id"], "system", "test", None, strict_roles=False
            )

    def test_raises_for_unknown_character(self, engine):
        with pytest.raises(KeyError):
            record_message_service(engine, "no-such", "user", "hi", None)


# ── recall_memories_service ───────────────────────────────────────────


class TestRecallMemories:
    def test_recall_after_record(self, engine):
        created = create_character_service(engine, "Alice", {}, None)
        start_session_service(engine, created["id"])
        record_message_service(engine, created["id"], "user", "I love cats", "toni")
        result = recall_memories_service(engine, created["id"], "cats", limit=5)
        assert "memories" in result
        assert "context" in result

    def test_empty_recall_on_fresh_character(self, engine):
        created = create_character_service(engine, "Alice", {}, None)
        result = recall_memories_service(engine, created["id"], "random query")
        assert "memories" in result

    def test_raises_for_unknown_character(self, engine):
        with pytest.raises(KeyError):
            recall_memories_service(engine, "no-such", "query")


# ── get_relationship_service ──────────────────────────────────────────


class TestGetRelationship:
    def test_returns_none_for_no_relationship(self, engine):
        created = create_character_service(engine, "Alice", {}, None)
        rel = get_relationship_service(engine, created["id"], "unknown-user")
        assert rel is None

    def test_raises_for_unknown_character(self, engine):
        with pytest.raises(KeyError):
            get_relationship_service(engine, "no-such", "target")


# ── find_character_by_name_or_id ──────────────────────────────────────


class TestFindCharacter:
    def test_find_by_exact_name(self, engine):
        created = create_character_service(engine, "Alice", {}, None)
        found = find_character_by_name_or_id(engine, "Alice")
        assert found is not None
        assert found["id"] == created["id"]

    def test_find_by_name_case_insensitive(self, engine):
        created = create_character_service(engine, "Alice", {}, None)
        found = find_character_by_name_or_id(engine, "alice")
        assert found is not None
        assert found["id"] == created["id"]

    def test_find_by_id(self, engine):
        created = create_character_service(engine, "Alice", {}, None)
        found = find_character_by_name_or_id(engine, created["id"])
        assert found is not None
        assert found["id"] == created["id"]

    def test_find_with_hyphen_for_space(self, engine):
        created = create_character_service(engine, "Mary Jane", {}, None)
        found = find_character_by_name_or_id(engine, "mary-jane")
        assert found is not None
        assert found["id"] == created["id"]

    def test_find_with_underscore_for_space(self, engine):
        created = create_character_service(engine, "Mary Jane", {}, None)
        found = find_character_by_name_or_id(engine, "mary_jane")
        assert found is not None
        assert found["id"] == created["id"]

    def test_returns_none_for_no_match(self, engine):
        assert find_character_by_name_or_id(engine, "nobody") is None


# ── extract helpers ───────────────────────────────────────────────────


class TestExtractHelpers:
    def test_extract_last_user_message(self):
        msgs = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Reply"},
            {"role": "user", "content": "Second"},
        ]
        assert extract_last_user_message(msgs) == "Second"

    def test_extract_last_user_message_empty(self):
        assert extract_last_user_message([]) == ""

    def test_extract_last_user_message_no_user(self):
        msgs = [{"role": "assistant", "content": "Hello"}]
        assert extract_last_user_message(msgs) == ""

    def test_extract_user_id_from_messages(self):
        msgs = [
            {"role": "system", "content": "config\nuser_id: toni42\nother"},
            {"role": "user", "content": "hi"},
        ]
        assert extract_user_id_from_messages(msgs) == "toni42"

    def test_extract_user_id_default(self):
        msgs = [{"role": "user", "content": "hi"}]
        assert extract_user_id_from_messages(msgs) == "api_user"

    def test_extract_user_id_custom_default(self):
        msgs = []
        assert extract_user_id_from_messages(msgs, default="anon") == "anon"


# ── delete_character_service ─────────────────────────────────────────


class TestDeleteCharacter:
    def test_deletes_existing_character(self, engine):
        created = create_character_service(engine, "Alice", {}, None)
        delete_character_service(engine, created["id"])
        assert list_characters_service(engine) == []

    def test_raises_keyerror_for_unknown_id(self, engine):
        with pytest.raises(KeyError):
            delete_character_service(engine, "nonexistent-id")


# ── export_character_service ─────────────────────────────────────────


class TestExportCharacter:
    def test_exports_existing_character(self, engine):
        created = create_character_service(engine, "Alice", {"personality": "kind"}, None)
        data = export_character_service(engine, created["id"])
        assert isinstance(data, dict)
        assert "persona" in data or "id" in data

    def test_raises_keyerror_for_unknown_id(self, engine):
        with pytest.raises(KeyError):
            export_character_service(engine, "nonexistent-id")


# ── import_character_service ─────────────────────────────────────────


class TestImportCharacter:
    def test_imports_from_exported_data(self, engine):
        created = create_character_service(engine, "Alice", {"personality": "kind"}, None)
        exported = export_character_service(engine, created["id"])
        # Delete original so we can re-import
        delete_character_service(engine, created["id"])
        result = import_character_service(engine, exported)
        assert result["imported"] is True
        assert result["id"]


# ── reflect_character_service ────────────────────────────────────────


class TestReflectCharacter:
    def test_reflect_returns_reflection(self, engine):
        created = create_character_service(engine, "Alice", {}, None)
        result = reflect_character_service(engine, created["id"])
        assert "reflection" in result
        assert isinstance(result["reflection"], str)

    def test_raises_keyerror_for_unknown_id(self, engine):
        with pytest.raises(KeyError):
            reflect_character_service(engine, "nonexistent-id")
