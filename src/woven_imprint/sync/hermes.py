"""Sync adapter for Hermes Agent — inject character into Hermes config."""

from __future__ import annotations

from pathlib import Path

from ..character import Character
from .base import SyncAdapter

_MARKER_START = "# WOVEN_IMPRINT_START"
_MARKER_END = "# WOVEN_IMPRINT_END"


class HermesSync(SyncAdapter):
    """Inject a character into Hermes Agent's persona and memory.

    Writes:
    - Character persona file in the Hermes workspace
    - Memory context for the agent
    """

    def __init__(self, hermes_dir: str | Path | None = None):
        if hermes_dir:
            self.hermes_dir = Path(hermes_dir)
        else:
            self.hermes_dir = Path.home() / ".hermes"

    def sync(self, character: Character) -> dict[str, Path]:
        files_written = {}

        # Write character persona file
        persona_path = self.hermes_dir / "woven_imprint_persona.md"
        persona_path.write_text(self._format_full_character(character))
        files_written["persona"] = persona_path

        return files_written

    def unsync(self) -> None:
        persona_path = self.hermes_dir / "woven_imprint_persona.md"
        if persona_path.exists():
            persona_path.unlink()
