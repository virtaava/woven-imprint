"""Generic markdown sync — write character state to any markdown file."""

from __future__ import annotations

from pathlib import Path

from ..character import Character
from .base import SyncAdapter

_MARKER_START = "<!-- WOVEN_IMPRINT_START -->"
_MARKER_END = "<!-- WOVEN_IMPRINT_END -->"


class GenericMarkdownSync(SyncAdapter):
    """Write character state to a markdown file.

    Works with any system that reads markdown files for context:
    Cursor (.cursorrules), Windsurf, Aider, custom setups.
    """

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)

    def sync(self, character: Character) -> dict[str, Path]:
        content = self._format_full_character(character)

        if self.output_path.exists():
            existing = self.output_path.read_text()
            if _MARKER_START in existing and _MARKER_END in existing:
                before = existing[: existing.index(_MARKER_START)]
                after = existing[existing.index(_MARKER_END) + len(_MARKER_END) :]
                content = (
                    before.rstrip()
                    + f"\n\n{_MARKER_START}\n{content}\n{_MARKER_END}\n"
                    + after.lstrip()
                )
            else:
                content = existing.rstrip() + f"\n\n{_MARKER_START}\n{content}\n{_MARKER_END}\n"
        else:
            content = f"{_MARKER_START}\n{content}\n{_MARKER_END}\n"

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(content)

        return {"character": self.output_path}

    def unsync(self) -> None:
        if not self.output_path.exists():
            return

        content = self.output_path.read_text()
        if _MARKER_START in content and _MARKER_END in content:
            before = content[: content.index(_MARKER_START)]
            after = content[content.index(_MARKER_END) + len(_MARKER_END) :]
            result = before.rstrip() + "\n" + after.lstrip()
            if result.strip():
                self.output_path.write_text(result)
            else:
                self.output_path.unlink()
