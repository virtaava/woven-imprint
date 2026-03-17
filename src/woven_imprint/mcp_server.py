"""MCP server — exposes Woven Imprint as tools for IDE integration."""

from __future__ import annotations

import json
from pathlib import Path

from .engine import Engine
from .llm.ollama import OllamaLLM
from .embedding.ollama import OllamaEmbedding

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    raise ImportError("mcp package required. Install with: pip install mcp")

mcp = FastMCP("WovenImprint", version="0.1.0")

# Global engine + character cache — state persists between tool calls (C2 fix)
_engine: Engine | None = None
_db_path: str = str(Path.home() / ".woven_imprint" / "characters.db")
_char_cache: dict = {}  # character_id → Character instance


def _get_engine() -> Engine:
    global _engine
    if _engine is None:
        from .config import get_config

        cfg = get_config()
        db = cfg.storage.db_path
        Path(db).parent.mkdir(parents=True, exist_ok=True)
        _engine = Engine(
            db_path=db,
            llm=OllamaLLM(model=cfg.llm.model, num_ctx=cfg.llm.num_ctx),
            embedding=OllamaEmbedding(model=cfg.llm.embedding_model),
        )
    return _engine


def _get_character(character_id: str):
    """Get a character from cache or load it. Keeps state between calls."""
    if character_id in _char_cache:
        return _char_cache[character_id]
    engine = _get_engine()
    char = engine.load_character(character_id)
    if char:
        _char_cache[character_id] = char
    return char


@mcp.tool()
def list_characters() -> str:
    """List all characters in the database."""
    engine = _get_engine()
    chars = engine.list_characters()
    return json.dumps(chars, indent=2, default=str)


@mcp.tool()
def create_character(
    name: str,
    personality: str = "",
    backstory: str = "",
    speaking_style: str = "",
    birthdate: str = "",
) -> str:
    """Create a new persistent character.

    Args:
        name: Character name.
        personality: Personality description.
        backstory: Character backstory.
        speaking_style: How the character speaks.
        birthdate: ISO date (YYYY-MM-DD) for age tracking.
    """
    engine = _get_engine()
    persona = {}
    if personality:
        persona["personality"] = personality
    if backstory:
        persona["backstory"] = backstory
    if speaking_style:
        persona["speaking_style"] = speaking_style

    char = engine.create_character(
        name=name,
        persona=persona,
        birthdate=birthdate or None,
    )
    _char_cache[char.id] = char  # Cache for subsequent tool calls
    return json.dumps({"id": char.id, "name": char.name, "age": char.persona.age}, default=str)


@mcp.tool()
def chat(character_id: str, message: str, user_id: str = "mcp_user") -> str:
    """Send a message to a character and get an in-character response.

    Args:
        character_id: The character's ID.
        message: Your message to the character.
        user_id: Your identifier for relationship tracking.
    """
    char = _get_character(character_id)
    if not char:
        return json.dumps({"error": f"Character {character_id} not found"})

    response = char.chat(message, user_id=user_id)
    return json.dumps(
        {
            "character": char.name,
            "response": response,
            "emotion": char.emotion.mood,
            "emotion_intensity": char.emotion.intensity,
            "arc_phase": char.arc.current_phase.value,
            "arc_tension": char.arc.tension,
        }
    )


@mcp.tool()
def recall(character_id: str, query: str, limit: int = 5) -> str:
    """Search a character's memories.

    Args:
        character_id: The character's ID.
        query: What to search for in their memories.
        limit: Maximum number of memories to return.
    """
    char = _get_character(character_id)
    if not char:
        return json.dumps({"error": f"Character {character_id} not found"})

    memories = char.recall(query, limit=limit)
    results = [
        {
            "tier": m["tier"],
            "content": m["content"][:300],
            "importance": m.get("importance", 0.5),
            "certainty": m.get("certainty", 1.0),
        }
        for m in memories
    ]
    return json.dumps({"character": char.name, "query": query, "memories": results}, indent=2)


@mcp.tool()
def get_relationship(character_id: str, target_id: str) -> str:
    """Get a character's relationship with another entity.

    Args:
        character_id: The character's ID.
        target_id: The other entity's ID.
    """
    char = _get_character(character_id)
    if not char:
        return json.dumps({"error": f"Character {character_id} not found"})

    description = char.relationships.describe(target_id)
    rel = char.relationships.get(target_id)
    return json.dumps(
        {
            "character": char.name,
            "target": target_id,
            "description": description,
            "dimensions": rel["dimensions"] if rel else None,
            "trajectory": rel["trajectory"] if rel else None,
        },
        indent=2,
    )


@mcp.tool()
def reflect(character_id: str) -> str:
    """Have a character reflect on their recent experiences.

    Args:
        character_id: The character's ID.
    """
    char = _get_character(character_id)
    if not char:
        return json.dumps({"error": f"Character {character_id} not found"})

    reflection = char.reflect()
    return json.dumps({"character": char.name, "reflection": reflection})


@mcp.tool()
def evolve(character_id: str) -> str:
    """Detect and apply character growth from accumulated experiences.

    Args:
        character_id: The character's ID.
    """
    char = _get_character(character_id)
    if not char:
        return json.dumps({"error": f"Character {character_id} not found"})

    events = char.evolve()
    return json.dumps({"character": char.name, "growth_events": events}, indent=2)


@mcp.tool()
def end_session(character_id: str) -> str:
    """End the current session and generate a summary.

    Args:
        character_id: The character's ID.
    """
    char = _get_character(character_id)
    if not char:
        return json.dumps({"error": f"Character {character_id} not found"})

    summary = char.end_session()
    return json.dumps({"character": char.name, "summary": summary})


@mcp.tool()
def new_session(character_id: str) -> str:
    """Start a fresh conversation session. Ends the current session first if one is active.

    Args:
        character_id: The character's ID.
    """
    char = _get_character(character_id)
    if not char:
        return json.dumps({"error": f"Character {character_id} not found"})

    # End current session if active
    summary = None
    if char._session_id:
        summary = char.end_session()

    session_id = char.start_session()
    return json.dumps(
        {
            "character": char.name,
            "session_id": session_id,
            "previous_summary": summary,
        }
    )


@mcp.tool()
def consolidate(character_id: str) -> str:
    """Compress buffer memories into consolidated core memories.

    Args:
        character_id: The character's ID.
    """
    char = _get_character(character_id)
    if not char:
        return json.dumps({"error": f"Character {character_id} not found"})

    stats = char.consolidate()
    return json.dumps({"character": char.name, "stats": stats})


@mcp.tool()
def get_stats(character_id: str) -> str:
    """Get character statistics: memory counts, emotional state, arc phase, relationships.

    Args:
        character_id: The character's ID.
    """
    char = _get_character(character_id)
    if not char:
        return json.dumps({"error": f"Character {character_id} not found"})

    rels = char.relationships.get_all()
    rel_summaries = []
    for r in rels:
        rel_summaries.append(
            {
                "target": r["target_id"],
                "type": r.get("type", "stranger"),
                "dimensions": r["dimensions"],
                "trajectory": r.get("trajectory", "stable"),
            }
        )

    return json.dumps(
        {
            "character": char.name,
            "age": char.persona.age,
            "emotion": char.emotion.to_dict(),
            "arc_phase": char.arc.current_phase.value,
            "arc_tension": char.arc.tension,
            "memory": {
                "buffer": char.memory.count("buffer"),
                "core": char.memory.count("core"),
                "bedrock": char.memory.count("bedrock"),
            },
            "relationships": rel_summaries,
            "session_active": char._session_id is not None,
        },
        indent=2,
    )


@mcp.tool()
def delete_character(character_id: str) -> str:
    """Permanently delete a character and all their data.

    Args:
        character_id: The character's ID.
    """
    engine = _get_engine()
    char = _get_character(character_id)
    if not char:
        return json.dumps({"error": f"Character {character_id} not found"})

    name = char.name
    engine.delete_character(character_id)
    _char_cache.pop(character_id, None)
    return json.dumps({"deleted": name})


@mcp.tool()
def migrate_from_text(text: str, name: str = "") -> str:
    """Create a character from existing persona text, Custom GPT instructions, or description.

    Paste any character definition and the system will extract personality,
    backstory, speaking style, and key memories automatically.

    Args:
        text: The persona/instructions text to migrate from.
        name: Optional character name (extracted from text if not provided).
    """
    from .migrate import CharacterImporter

    engine = _get_engine()
    importer = CharacterImporter(engine)
    char = importer.from_text(text, name=name or None)
    _char_cache[char.id] = char
    return json.dumps(
        {
            "character": char.name,
            "id": char.id,
            "personality": char.persona.soft.get("personality", ""),
            "memories_seeded": char.memory.count("core"),
        }
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
