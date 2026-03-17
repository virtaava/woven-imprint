"""Character importer — migrate from other AI systems to Woven Imprint.

Takes data from ChatGPT, Custom GPTs, SillyTavern character cards,
Claude projects, or generic text and creates a fully populated
Woven Imprint character with persona, memories, and backstory.
"""

from __future__ import annotations

from pathlib import Path

from ..engine import Engine
from ..character import Character
from .parsers import auto_detect, parse_custom_gpt


class CharacterImporter:
    """Import characters from other AI systems.

    Usage:
        importer = CharacterImporter(engine)

        # From a file (auto-detects format)
        character = importer.from_file("conversations.json")
        character = importer.from_file("character_card.png")
        character = importer.from_file("persona.md")

        # From raw text (Custom GPT instructions, copy-pasted persona)
        character = importer.from_text("You are Marcus, a gruff blacksmith...")

        # From ChatGPT export
        character = importer.from_chatgpt_export("conversations.json")
    """

    def __init__(self, engine: Engine):
        self.engine = engine
        self.llm = engine.llm

    def from_file(self, path: str | Path, name: str | None = None) -> Character:
        """Import from any supported file format (auto-detected).

        Supported: ChatGPT JSON export, TavernAI/SillyTavern cards (JSON/PNG),
        Claude project directories, markdown/text persona files.

        Args:
            path: Path to the file or directory.
            name: Override character name (auto-extracted if not provided).

        Returns:
            A fully created Character with persona and seeded memories.
        """
        parsed = auto_detect(path)
        return self._build_character(parsed, name_override=name)

    def from_text(self, text: str, name: str | None = None) -> Character:
        """Import from raw text (Custom GPT instructions, persona description, etc.).

        Args:
            text: The persona/instruction text.
            name: Character name (extracted from text if not provided).
        """
        parsed = parse_custom_gpt(text)
        return self._build_character(parsed, name_override=name)

    def from_chatgpt_export(self, path: str | Path, name: str | None = None) -> Character:
        """Import from a ChatGPT data export.

        Go to ChatGPT Settings → Data Controls → Export Data.
        Use the conversations.json file from the export.

        Args:
            path: Path to conversations.json.
            name: Character name (defaults to "Assistant" or extracted).
        """
        from .parsers import parse_chatgpt_export

        parsed = parse_chatgpt_export(path)
        return self._build_character(parsed, name_override=name)

    def _build_character(self, parsed: dict, name_override: str | None = None) -> Character:
        """Use LLM to analyze parsed data and create a character."""
        source = parsed.get("source", "unknown")

        # Build analysis prompt based on source type
        if source == "tavernai" and "card" in parsed:
            card = parsed["card"]
            analysis = self._analyze_tavernai(card)
        elif parsed.get("instructions"):
            analysis = self._analyze_instructions(
                parsed["instructions"],
                parsed.get("messages", []),
            )
        elif parsed.get("messages"):
            analysis = self._analyze_conversations(parsed["messages"])
        else:
            raise ValueError("No usable data found in the import source")

        # Override name if provided
        if name_override:
            analysis["name"] = name_override

        name = analysis.get("name", "Imported Character")
        persona = {
            "backstory": analysis.get("backstory", ""),
            "personality": analysis.get("personality", ""),
            "speaking_style": analysis.get("speaking_style", ""),
        }

        # Add any extra traits
        for key in ("occupation", "appearance", "quirks"):
            if analysis.get(key):
                persona[key] = analysis[key]

        # Create character
        char = self.engine.create_character(
            name=name,
            birthdate=analysis.get("birthdate"),
            persona=persona,
        )

        # Seed additional memories from conversations
        memories = analysis.get("key_memories", [])
        for mem in memories[:20]:
            if isinstance(mem, str) and len(mem) > 10:
                char.memory.add(content=mem, tier="core", importance=0.7)

        return char

    def _analyze_tavernai(self, card: dict) -> dict:
        """Extract character info from a TavernAI card — often already structured."""
        result = {
            "name": card.get("name", "Unknown"),
            "personality": card.get("personality", ""),
            "backstory": card.get("description", ""),
            "speaking_style": "",
            "key_memories": [],
        }

        # If personality is empty, try to extract from description
        if not result["personality"] and result["backstory"]:
            result = self._llm_extract(
                f"Character name: {result['name']}\n"
                f"Description: {result['backstory']}\n"
                f"Example dialogue: {card.get('mes_example', '')[:500]}",
                existing=result,
            )

        # Extract speaking style from example messages
        if card.get("mes_example"):
            examples = card["mes_example"][:1000]
            result["speaking_style"] = self._extract_speaking_style(examples, result["name"])

        return result

    def _analyze_instructions(self, instructions: str, messages: list[dict]) -> dict:
        """Extract character from instruction text + optional chat history."""
        context = f"Instructions/persona definition:\n{instructions[:3000]}"
        if messages:
            sample = "\n".join(f"{m['role']}: {m['content'][:200]}" for m in messages[:20])
            context += f"\n\nSample conversations:\n{sample}"

        return self._llm_extract(context)

    def _analyze_conversations(self, messages: list[dict]) -> dict:
        """Extract character from chat history alone (no explicit persona)."""
        # Get assistant messages to understand the character
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        sample = "\n".join(m["content"][:200] for m in assistant_msgs[:30])

        context = (
            "The following are messages from an AI assistant. "
            "Based on these messages, determine the assistant's character:\n\n"
            f"{sample}"
        )

        return self._llm_extract(context)

    def _llm_extract(self, context: str, existing: dict | None = None) -> dict:
        """Use LLM to extract structured character data."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You extract character information from text. Return JSON with:\n"
                    "- name: character name (string)\n"
                    "- personality: personality traits (string, comma-separated)\n"
                    "- backstory: character backstory (string, 2-4 sentences)\n"
                    "- speaking_style: how they talk (string)\n"
                    "- occupation: what they do (string, optional)\n"
                    "- key_memories: important facts to remember (list of strings, max 10)\n\n"
                    "Be concise. Extract only what the text supports."
                ),
            },
            {
                "role": "user",
                "content": f"Extract the character from this:\n\n{context[:4000]}",
            },
        ]

        try:
            result = self.llm.generate_json(messages)
            if not isinstance(result, dict):
                return existing or {
                    "name": "Imported Character",
                    "personality": "",
                    "backstory": "",
                }

            # Merge with existing if provided
            if existing:
                for key, val in existing.items():
                    if val and not result.get(key):
                        result[key] = val

            return result
        except (ValueError, KeyError):
            return existing or {"name": "Imported Character", "personality": "", "backstory": ""}

    def _extract_speaking_style(self, examples: str, name: str) -> str:
        """Extract speaking style from example dialogue."""
        messages = [
            {
                "role": "system",
                "content": "Describe this character's speaking style in one sentence.",
            },
            {
                "role": "user",
                "content": f"Character: {name}\nExample dialogue:\n{examples[:2000]}",
            },
        ]
        try:
            return self.llm.generate(messages, temperature=0.3, max_tokens=100)
        except Exception:
            return ""
