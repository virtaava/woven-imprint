"""Service-layer functions for woven-imprint server endpoints.

Business logic extracted from sidecar.py and api.py handlers.
Both the existing stdlib servers and the new FastAPI demo server
call these same functions — no logic duplication.
"""

from __future__ import annotations


def create_character_service(engine, name, persona, birthdate):
    """Create a character, deduplicating by name (case-insensitive).
    Returns dict with keys: id, name, created (bool).
    String persona normalized to {"personality": persona} to match existing sidecar behavior.
    """
    existing = engine.list_characters()
    for char_info in existing:
        if char_info["name"].lower() == name.lower():
            return {"id": char_info["id"], "name": char_info["name"], "created": False}
    if isinstance(persona, str):
        persona = {"personality": persona}
    char = engine.create_character(name=name, persona=persona, birthdate=birthdate)
    return {"id": char.id, "name": name, "created": True}


def list_characters_service(engine):
    return engine.list_characters()


def get_character_state_service(engine, character_id):
    """Raises KeyError if not found. Uses char.name property."""
    char = engine.get_character(character_id)
    emotion = char.emotion.to_dict() if hasattr(char, "emotion") and char.emotion else {}
    arc_phase = ""
    arc_tension = 0.0
    if hasattr(char, "arc") and char.arc:
        arc_phase = (
            char.arc.current_phase.value
            if hasattr(char.arc.current_phase, "value")
            else str(char.arc.current_phase)
        )
        arc_tension = getattr(char.arc, "tension", 0.0)
    return {
        "id": char.id,
        "name": getattr(char, "name", character_id),
        "emotion": emotion,
        "arc": {"phase": arc_phase, "tension": arc_tension},
    }


def start_session_service(engine, character_id):
    char = engine.get_character(character_id)
    session_id = char.start_session()
    return {"session_id": session_id}


def end_session_service(engine, character_id):
    char = engine.get_character(character_id)
    summary = char.end_session()
    return {"summary": summary}


_SIDECAR_ROLES = {"user", "assistant"}
_ALL_ROLES = {"user", "assistant", "character", "system"}


def record_message_service(engine, character_id, role, content, user_id, *, strict_roles=True):
    """strict_roles=True (sidecar compat) only allows user/assistant."""
    allowed = _SIDECAR_ROLES if strict_roles else _ALL_ROLES
    if role not in allowed:
        raise ValueError(f"Invalid role '{role}'. Must be one of: {allowed}")
    char = engine.get_character(character_id)
    char.ingest(role=role, content=content, user_id=user_id)


def recall_memories_service(engine, character_id, query, limit=10, user_id=None):
    char = engine.get_character(character_id)
    memories = char.recall(query, limit=limit)
    context_parts = []
    if user_id:
        rel = char.get_relationship(user_id)
        if rel:
            desc = char.relationships.describe(user_id)
            context_parts.append(f"[Relationship with {user_id}: {desc}]")
    if memories:
        context_parts.append("[Relevant memories:]")
        for m in memories[:limit]:
            content = m.get("content", "")[:200]
            context_parts.append(f"- {content}")
    context = "\n".join(context_parts)
    return {"memories": memories, "context": context}


def get_relationship_service(engine, character_id, target_id):
    char = engine.get_character(character_id)
    return char.get_relationship(target_id)


def find_character_by_name_or_id(engine, name_or_id):
    chars = engine.list_characters()
    model_name = name_or_id.lower().strip()
    return next(
        (
            c
            for c in chars
            if c["name"].lower().replace(" ", "-") == model_name
            or c["name"].lower().replace(" ", "_") == model_name
            or c["name"].lower() == model_name
            or c["id"] == name_or_id
        ),
        None,
    )


def extract_last_user_message(messages):
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def extract_user_id_from_messages(messages, default="api_user"):
    for msg in messages:
        if msg.get("role") == "system" and "user_id:" in msg.get("content", ""):
            for line in msg["content"].split("\n"):
                if line.strip().startswith("user_id:"):
                    return line.split(":", 1)[1].strip()
    return default


def delete_character_service(engine, character_id):
    """Delete a character. Raises KeyError if not found."""
    engine.get_character(character_id)  # verify exists
    engine.delete_character(character_id)


def export_character_service(engine, character_id):
    """Export character as JSON dict. Raises KeyError if not found."""
    char = engine.get_character(character_id)
    return char.export()


def import_character_service(engine, data: dict):
    """Import character from JSON dict. Returns character info."""
    import tempfile
    import json
    import os

    # Write to temp file since engine.import_character expects a path
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        tmp_path = f.name
    try:
        char = engine.import_character(tmp_path)
        return {"id": char.id, "name": getattr(char, "name", ""), "imported": True}
    finally:
        os.unlink(tmp_path)


def migrate_character_service(engine, name, text=None, file_path=None):
    """Migrate/create character from text or file. Returns character info."""
    from woven_imprint.migrate import CharacterImporter

    importer = CharacterImporter(engine)
    if text:
        char = importer.from_text(text, name=name)
    elif file_path:
        char = importer.from_file(file_path, name=name)
    else:
        raise ValueError("Either text or file_path must be provided")
    return {"id": char.id, "name": getattr(char, "name", ""), "migrated": True}


def reflect_character_service(engine, character_id):
    """Trigger character reflection. Returns reflection text."""
    char = engine.get_character(character_id)
    reflection = char.reflect()
    return {"reflection": reflection}
