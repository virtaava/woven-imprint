"""Parsers for different source formats — extract raw data before LLM analysis."""

from __future__ import annotations

import json
from pathlib import Path


def parse_chatgpt_export(path: str | Path) -> dict:
    """Parse a ChatGPT data export (conversations.json).

    OpenAI export format: list of conversations, each with a
    mapping of message nodes.
    """
    with open(path) as f:
        data = json.load(f)

    conversations = data if isinstance(data, list) else [data]
    all_messages = []
    title = ""

    for conv in conversations:
        if not title and conv.get("title"):
            title = conv["title"]

        mapping = conv.get("mapping", {})
        for node in mapping.values():
            msg = node.get("message")
            if not msg or not msg.get("content"):
                continue
            parts = msg["content"].get("parts", [])
            text = " ".join(str(p) for p in parts if isinstance(p, str))
            if not text.strip():
                continue
            role = msg.get("author", {}).get("role", "unknown")
            all_messages.append({"role": role, "content": text[:2000]})

    return {
        "source": "chatgpt",
        "title": title,
        "messages": all_messages[:500],  # cap to avoid huge context
        "instructions": "",
    }


def parse_custom_gpt(text: str) -> dict:
    """Parse OpenAI Custom GPT instructions (copy-pasted text)."""
    return {
        "source": "custom_gpt",
        "title": "",
        "messages": [],
        "instructions": text[:10000],
    }


def parse_tavernai_card(path: str | Path) -> dict:
    """Parse a TavernAI / SillyTavern character card (JSON or PNG with embedded JSON).

    Character card spec v2: name, description, personality, first_mes,
    mes_example, scenario, creator_notes.
    """
    path = Path(path)

    if path.suffix == ".png":
        # Character card PNG has JSON in the tEXt chunk
        import base64

        with open(path, "rb") as f:
            raw = f.read()
        # Look for the 'chara' tEXt chunk
        marker = b"chara\x00"
        idx = raw.find(marker)
        if idx == -1:
            raise ValueError("No character data found in PNG")
        # Find the next chunk boundary or end
        start = idx + len(marker)
        # The base64 data runs until the next PNG chunk
        end = raw.find(b"\x00\x00\x00", start + 100)
        if end == -1:
            end = len(raw)
        b64_data = raw[start:end]
        card_json = base64.b64decode(b64_data)
        card = json.loads(card_json)
    else:
        with open(path) as f:
            card = json.load(f)

    # Handle v2 spec wrapper
    if "data" in card:
        card = card["data"]

    return {
        "source": "tavernai",
        "title": card.get("name", ""),
        "messages": [],
        "instructions": "",
        "card": {
            "name": card.get("name", ""),
            "description": card.get("description", ""),
            "personality": card.get("personality", ""),
            "first_mes": card.get("first_mes", ""),
            "mes_example": card.get("mes_example", ""),
            "scenario": card.get("scenario", ""),
            "creator_notes": card.get("creator_notes", ""),
            "tags": card.get("tags", []),
        },
    }


def parse_claude_project(path: str | Path) -> dict:
    """Parse Claude Code project files (CLAUDE.md + memory files)."""
    path = Path(path)

    instructions = ""
    memories = []

    # Read CLAUDE.md
    claude_md = path / "CLAUDE.md" if path.is_dir() else path
    if claude_md.exists():
        instructions = claude_md.read_text()[:10000]

    # Read memory files
    if path.is_dir():
        for md_file in path.glob("memory/*.md"):
            content = md_file.read_text()[:3000]
            memories.append({"file": md_file.name, "content": content})

    return {
        "source": "claude_project",
        "title": "",
        "messages": [],
        "instructions": instructions,
        "memories": memories,
    }


def parse_markdown(path: str | Path) -> dict:
    """Parse a generic markdown persona/character file."""
    text = Path(path).read_text()[:10000]
    return {
        "source": "markdown",
        "title": "",
        "messages": [],
        "instructions": text,
    }


def auto_detect(path: str | Path) -> dict:
    """Auto-detect format and parse."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.is_dir():
        if (path / "CLAUDE.md").exists():
            return parse_claude_project(path)
        raise ValueError(f"Cannot detect format for directory: {path}")

    suffix = path.suffix.lower()
    name = path.name.lower()

    if suffix == ".png":
        return parse_tavernai_card(path)

    if suffix == ".json":
        with open(path) as f:
            data = json.load(f)

        # ChatGPT export: list of conversations or single conversation with 'mapping'
        if isinstance(data, list) and data and "mapping" in data[0]:
            return parse_chatgpt_export(path)
        if isinstance(data, dict) and "mapping" in data:
            return parse_chatgpt_export(path)

        # TavernAI card: has 'name' + 'description' or 'data' wrapper
        if isinstance(data, dict):
            inner = data.get("data", data)
            if "personality" in inner or "first_mes" in inner:
                return parse_tavernai_card(path)

        # Generic JSON with instructions
        return {
            "source": "json",
            "title": data.get("name", data.get("title", "")),
            "messages": [],
            "instructions": json.dumps(data, indent=2)[:10000],
        }

    if suffix in (".md", ".txt"):
        if name == "claude.md":
            return parse_claude_project(path.parent)
        return parse_markdown(path)

    raise ValueError(f"Unsupported file format: {suffix}")
