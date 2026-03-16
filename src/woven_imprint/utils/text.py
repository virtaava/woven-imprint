"""Text processing utilities."""

from __future__ import annotations

import re


def truncate(text: str, max_chars: int = 500) -> str:
    """Truncate text to max_chars, adding ellipsis if needed."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def generate_id(prefix: str = "") -> str:
    """Generate a short unique ID. Uses uuid4 for collision resistance."""
    import uuid

    h = uuid.uuid4().hex[:12]
    return f"{prefix}{h}" if prefix else h


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace into single spaces."""
    return re.sub(r"\s+", " ", text).strip()
