"""Sync adapter for OpenClaw — inject character into workspace files."""

from __future__ import annotations

from pathlib import Path

from ..character import Character
from .base import SyncAdapter

_MARKER_START = "<!-- WOVEN_IMPRINT_START -->"
_MARKER_END = "<!-- WOVEN_IMPRINT_END -->"


class OpenClawSync(SyncAdapter):
    """Inject a character into OpenClaw's workspace.

    Writes character state to MEMORY.md and optionally to soul.md.
    """

    def __init__(self, workspace_dir: str | Path | None = None):
        if workspace_dir:
            self.workspace_dir = Path(workspace_dir)
        else:
            self.workspace_dir = Path.home() / ".openclaw" / "workspace"

    def sync(self, character: Character) -> dict[str, Path]:
        files_written = {}

        # Write character file
        char_path = self.workspace_dir / "woven_imprint_character.md"
        char_path.write_text(self._format_full_character(character))
        files_written["character"] = char_path

        # Inject into MEMORY.md if it exists
        memory_md = self.workspace_dir / "MEMORY.md"
        if memory_md.exists():
            self._inject_block(memory_md, character)
            files_written["MEMORY.md injection"] = memory_md

        return files_written

    def unsync(self) -> None:
        # Remove character file
        char_path = self.workspace_dir / "woven_imprint_character.md"
        if char_path.exists():
            char_path.unlink()

        # Remove injection from MEMORY.md
        memory_md = self.workspace_dir / "MEMORY.md"
        if memory_md.exists():
            content = memory_md.read_text()
            if _MARKER_START in content and _MARKER_END in content:
                before = content[: content.index(_MARKER_START)]
                after = content[content.index(_MARKER_END) + len(_MARKER_END) :]
                memory_md.write_text(before.rstrip() + "\n" + after.lstrip())

    def _inject_block(self, file_path: Path, character: Character) -> None:
        content = file_path.read_text()
        block = (
            f"\n{_MARKER_START}\n"
            f"## Active Character: {character.name}\n\n"
            f"{self._format_persona_section(character)}\n"
            f"{_MARKER_END}\n"
        )

        if _MARKER_START in content and _MARKER_END in content:
            before = content[: content.index(_MARKER_START)]
            after = content[content.index(_MARKER_END) + len(_MARKER_END) :]
            content = before.rstrip() + "\n" + block + after.lstrip()
        else:
            content = content.rstrip() + "\n" + block

        file_path.write_text(content)
