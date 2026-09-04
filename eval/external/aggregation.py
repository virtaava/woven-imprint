"""Aggregation-question detection + enumerate-then-answer prompt for eval/external.

Tier 3n — spec: docs/superpowers/specs/2026-09-04-tier3n-aggregation-answer.md.

Enumeration questions ("how many", "in total", "what order...") are the Tier 3h/3l failure
shape: the answer is scattered across instance memories, and the model either abstains or
counts only what it notices in a truncated list under a tight token cap. This module gives
``runner.answer_question`` (behind ``RunConfig.agg_stage``, opt-in) a second answering path for
exactly these questions: list every matching item first, then answer — closer to how a human
would total up scattered facts than a single terse guess.
"""

from __future__ import annotations

import re

# Word-boundaried, case-insensitive. Each pattern is a standalone trigger — a question needs
# to match only one. Kept as a closed set (not a heuristic classifier): unit-tested against
# every multi-session question in the committed LME fixture plus explicit negative examples
# (see tests/test_aggregation_stage.py).
_AGGREGATION_PATTERNS = [
    re.compile(r"\bhow many\b", re.IGNORECASE),
    re.compile(r"\bhow much\b", re.IGNORECASE),
    re.compile(r"\bin total\b", re.IGNORECASE),
    re.compile(r"\btotal number\b", re.IGNORECASE),
    re.compile(r"\btotal amount\b", re.IGNORECASE),
    re.compile(r"\baltogether\b", re.IGNORECASE),
    re.compile(r"\blist all\b", re.IGNORECASE),
    re.compile(r"\border of\b", re.IGNORECASE),
    re.compile(r"\bsequence of\b", re.IGNORECASE),
    # "what ... order" — e.g. "What order did I visit the cities in?"
    re.compile(r"\bwhat\b.*\border\b", re.IGNORECASE),
    # "first ... then" style ordering asks — e.g. "What did I do first, and what then?"
    re.compile(r"\bfirst\b.*\bthen\b", re.IGNORECASE),
]


def is_aggregation_question(q: str) -> bool:
    """True iff ``q`` matches one of the closed aggregation-phrase patterns above."""
    return any(pattern.search(q) for pattern in _AGGREGATION_PATTERNS)


# Variant of QA_SYSTEM (see prompts.py) for aggregation questions only: same never-guess
# contract (only from the provided memories, absolute dates, no inference, "Not mentioned" if
# unstated), but the response protocol trades the flat 15-word answer for enumerate-then-answer
# — the model lists every matching item it found before giving a final concise verdict, so a
# scattered count/total/order isn't lost to a single terse guess. The 15-word limit now applies
# only to the final "Answer:" line, not the whole response.
AGG_QA_SYSTEM = (
    "You answer questions about a person using ONLY the memories provided. Convert relative "
    "dates to absolute dates. You may combine multiple memories, but never guess or infer "
    "beyond what they state. If the memories do not actually state the answer, reply exactly: "
    "Not mentioned. First list each matching item the memories state, one per line, with its "
    "date/amount. Then give the final line exactly as 'Answer: <concise answer>' (at most 15 "
    "words)."
)


def parse_agg_answer(text: str) -> str:
    """Return the text after the last ``Answer:`` marker, stripped.

    Multiple markers: the last one wins (a model that second-guesses itself mid-response should
    be judged on its final answer). No marker at all: the fallback is the whole response,
    stripped — judged honestly as-is rather than invented.
    """
    if not text:
        return ""
    marker = "Answer:"
    idx = text.rfind(marker)
    if idx == -1:
        return text.strip()
    return text[idx + len(marker) :].strip()


def agg_qa_messages(memory_block: str, question: str, today: str) -> list[dict]:
    """Build the aggregation QA prompt — mirrors ``prompts.qa_messages`` but with
    ``AGG_QA_SYSTEM`` in place of ``QA_SYSTEM``."""
    parts = [f"Today is {today}."]
    if memory_block:
        parts.append(memory_block)
    parts.append(f"Question: {question}")
    return [
        {"role": "system", "content": AGG_QA_SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
