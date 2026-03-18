"""Parsers for different source formats — extract raw data before LLM analysis."""

from __future__ import annotations

import json
from pathlib import Path


def parse_chatgpt_export(
    path: str | Path,
    max_messages: int = 0,
    max_message_length: int = 0,
) -> dict:
    """Parse a ChatGPT data export (conversations.json).

    OpenAI export format: list of conversations, each with a
    mapping of message nodes.

    Args:
        path: Path to conversations.json.
        max_messages: Maximum messages to extract (0 = unlimited).
        max_message_length: Maximum length per message (0 = unlimited).
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
            if max_message_length > 0:
                text = text[:max_message_length]
            all_messages.append({"role": role, "content": text})

    if max_messages > 0:
        all_messages = all_messages[:max_messages]

    return {
        "source": "chatgpt",
        "title": title,
        "messages": all_messages,
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
        # TavernAI character cards store Base64-encoded JSON in PNG tEXt chunks.
        # PNG chunks: 4-byte length + 4-byte type + data + 4-byte CRC.
        # tEXt chunks contain: keyword + null byte + text value.
        # We look for a tEXt chunk with keyword "chara".
        import base64
        import struct

        with open(path, "rb") as f:
            raw = f.read()

        # Skip 8-byte PNG signature
        pos = 8
        card_b64 = None
        while pos < len(raw) - 8:
            chunk_len = struct.unpack(">I", raw[pos : pos + 4])[0]
            chunk_type = raw[pos + 4 : pos + 8]
            chunk_data = raw[pos + 8 : pos + 8 + chunk_len]

            if chunk_type == b"tEXt":
                # tEXt: keyword\x00value
                null_idx = chunk_data.find(b"\x00")
                if null_idx != -1:
                    keyword = chunk_data[:null_idx]
                    if keyword == b"chara":
                        card_b64 = chunk_data[null_idx + 1 :]
                        break

            pos += 12 + chunk_len  # 4 len + 4 type + data + 4 CRC

        if card_b64 is None:
            raise ValueError("No 'chara' tEXt chunk found in PNG")

        card_json = base64.b64decode(card_b64)
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
    """Parse Claude Code project files — all markdown files + .claude/ directory."""
    path = Path(path)

    instructions = ""
    memories = []

    # Read CLAUDE.md
    claude_md = path / "CLAUDE.md" if path.is_dir() else path
    if claude_md.exists():
        instructions = claude_md.read_text()

    if path.is_dir():
        # Recursively scan all markdown files (not just memory/*.md)
        for md_file in path.rglob("*.md"):
            if md_file.name == "CLAUDE.md":
                continue  # Already read above
            content = md_file.read_text()
            memories.append({
                "file": str(md_file.relative_to(path)),
                "content": content,
            })

        # Also scan .claude/ directory if present
        claude_dir = path / ".claude"
        if claude_dir.is_dir():
            for md_file in claude_dir.rglob("*.md"):
                content = md_file.read_text()
                memories.append({
                    "file": f".claude/{md_file.relative_to(claude_dir)}",
                    "content": content,
                })

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
        title = ""
        if isinstance(data, dict):
            title = data.get("name", data.get("title", ""))
        return {
            "source": "json",
            "title": title,
            "messages": [],
            "instructions": json.dumps(data, indent=2)[:10000],
        }

    if suffix in (".md", ".txt"):
        if name == "claude.md":
            return parse_claude_project(path.parent)
        return parse_markdown(path)

    raise ValueError(f"Unsupported file format: {suffix}")
