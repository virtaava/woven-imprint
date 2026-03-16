"""Sync adapter for Claude Code — inject character into CLAUDE.md + memory files."""

from __future__ import annotations

from pathlib import Path

from ..character import Character
from .base import SyncAdapter

# Claude Code stores project instructions in CLAUDE.md and memories in
# ~/.claude/projects/<project-hash>/memory/

_MARKER_START = "<!-- WOVEN_IMPRINT_START -->"
_MARKER_END = "<!-- WOVEN_IMPRINT_END -->"


class ClaudeCodeSync(SyncAdapter):
    """Inject a character into Claude Code's configuration.

    Writes:
    - CLAUDE.md section with persona instructions
    - Memory file with character memories and relationships
    """

    def __init__(
        self,
        project_dir: str | Path = ".",
        memory_dir: str | Path | None = None,
    ):
        self.project_dir = Path(project_dir).resolve()
        self.claude_md = self.project_dir / "CLAUDE.md"

        # Claude Code memory directory
        if memory_dir:
            self.memory_dir = Path(memory_dir)
        else:
            # Default: ~/.claude/projects/<hash>/memory/
            self.memory_dir = None  # Will be created at sync time

    def sync(self, character: Character) -> dict[str, Path]:
        files_written = {}

        # 1. Inject persona into CLAUDE.md
        persona_block = self._build_claude_md_block(character)
        self._inject_into_claude_md(persona_block)
        files_written["persona (CLAUDE.md)"] = self.claude_md

        # 2. Write character memory file
        memory_content = self._build_memory_file(character)
        memory_path = self._get_memory_path(character)
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        memory_path.write_text(memory_content)
        files_written["memories"] = memory_path

        return files_written

    def unsync(self) -> None:
        """Remove character injection from CLAUDE.md."""
        if not self.claude_md.exists():
            return

        content = self.claude_md.read_text()
        if _MARKER_START in content and _MARKER_END in content:
            before = content[: content.index(_MARKER_START)]
            after = content[content.index(_MARKER_END) + len(_MARKER_END) :]
            self.claude_md.write_text(before.rstrip() + "\n" + after.lstrip())

    def _build_claude_md_block(self, character: Character) -> str:
        """Build the CLAUDE.md injection block."""
        lines = [
            _MARKER_START,
            "",
            "## Active Character: " + character.name,
            "",
            "**IMPORTANT: You are operating as a character. Follow these instructions exactly.**",
            "",
            self._format_persona_section(character),
            self._format_emotional_state(character),
            "### Behavior Rules",
            "- Stay in character at all times",
            "- Your memories below are real experiences — reference them naturally",
            "- Your relationships affect how you respond to people",
            "- Your emotional state should color your responses",
            "- Hard facts about your identity NEVER change",
            "- Your personality can evolve slowly through experience",
            "",
            _MARKER_END,
        ]
        return "\n".join(lines)

    def _build_memory_file(self, character: Character) -> str:
        """Build the memory file content."""
        lines = [
            "---",
            f"name: woven-imprint-{character.name.lower().replace(' ', '-')}",
            f"description: Persistent character state for {character.name}",
            "type: project",
            "---",
            "",
            self._format_memory_section(character),
            self._format_relationships_section(character),
        ]
        return "\n".join(lines)

    def _inject_into_claude_md(self, block: str) -> None:
        """Inject or replace the character block in CLAUDE.md."""
        if self.claude_md.exists():
            content = self.claude_md.read_text()
            if _MARKER_START in content and _MARKER_END in content:
                # Replace existing block
                before = content[: content.index(_MARKER_START)]
                after = content[content.index(_MARKER_END) + len(_MARKER_END) :]
                content = before.rstrip() + "\n\n" + block + "\n" + after.lstrip()
            else:
                # Append
                content = content.rstrip() + "\n\n" + block + "\n"
        else:
            content = block + "\n"

        self.claude_md.write_text(content)

    def _get_memory_path(self, character: Character) -> Path:
        """Get the memory file path for a character."""
        if self.memory_dir:
            return self.memory_dir / f"woven_imprint_{character.name.lower().replace(' ', '_')}.md"
        # Default: same directory as CLAUDE.md
        return self.project_dir / f".woven_imprint_{character.name.lower().replace(' ', '_')}.md"
