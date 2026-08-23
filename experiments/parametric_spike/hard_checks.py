"""Deterministic persona-violation detectors and the drift slope."""
from __future__ import annotations

import re

EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿\U0001F900-\U0001F9FF]")
AI_CLAIM_RE = re.compile(
    r"\b(as an ai|language model|i am an ai|i'm an ai|ai assistant|i am a chatbot|i'm a chatbot|"
    r"i am an artificial intelligence|i'm an artificial intelligence|large language model)\b", re.I)
_TERMINATORS = ".!?\"'”’)*"


def check(text: str) -> dict:
    t = text.strip()
    emoji = bool(EMOJI_RE.search(t))
    ai = bool(AI_CLAIM_RE.search(t))
    incomplete = (not t) or (t[-1] not in _TERMINATORS)
    return {"emoji": emoji, "ai_claim": ai, "incomplete": incomplete, "any": emoji or ai or incomplete}


def slope(scores: list[float]) -> float:
    n = len(scores)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx, my = sum(xs) / n, sum(scores) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, scores))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else 0.0
