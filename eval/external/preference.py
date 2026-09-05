"""Preference-question detection + advice-answer prompt for eval/external.

Tier 3q — spec: docs/superpowers/specs/2026-09-05-tier3q-preference-answer.md.

LME's ``single-session-preference`` category has been the worst category all campaign
(0/8 -> 1/17 -> 4/17): the questions are advice/recommendation REQUESTS ("Can you recommend a
show for me tonight?"), the golds are rubrics ("The user would prefer responses that reference
their stated ..."), and the hardened <=15-word factual QA prompt (``prompts.QA_SYSTEM``) makes
the model ABSTAIN — there is no fact to abstain from. This module gives
``runner.answer_question`` (behind ``RunConfig.pref_stage``, opt-in) a second answering path for
exactly these questions: answer the request helpfully, grounded in whatever preferences/plans/
past experience the memories do state, instead of forcing a terse factual lookup.
"""

from __future__ import annotations

import re

# Word-boundaried, case-insensitive. Each pattern is a standalone trigger — a question needs to
# match only one. Kept as a closed set (not a heuristic classifier): unit-tested against all 17
# `single-session-preference` question texts from the committed `external_lme-s-100-v5.json`
# fixture (>= 15 must match) plus the 7 `_abs` abstain questions and a spread of factual
# questions from other categories (must match 0) — see tests/test_preference_stage.py.
_PREFERENCE_PATTERNS = [
    re.compile(r"\bcan you (suggest|recommend)\b", re.IGNORECASE),
    re.compile(r"\bany suggestions\b", re.IGNORECASE),
    re.compile(r"\bany tips\b", re.IGNORECASE),
    re.compile(r"\bany advice\b", re.IGNORECASE),
    # "do you have any [helpful] tips/suggestions/..." — allow up to two words between "any"
    # and the noun (e.g. "any helpful tips") without opening this up to unrelated phrasing.
    re.compile(
        r"\bdo you have any\s+(?:\w+\s+){0,2}(suggestions|recommendations|ideas|tips)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(what|how) do you think\b", re.IGNORECASE),
    re.compile(r"\bdo you think it would\b", re.IGNORECASE),
    re.compile(r"\bshould i\b", re.IGNORECASE),
    re.compile(r"\brecommend (a|some|any)\b", re.IGNORECASE),
    re.compile(r"\bhelp me (choose|decide|plan|pick)\b", re.IGNORECASE),
    re.compile(r"\bwhat would you (recommend|suggest)\b", re.IGNORECASE),
    # "any documentary recommendations" — bounded gap so this stays an advice-request shape
    # rather than a generic "recommendations" mention anywhere in the sentence.
    re.compile(r"\bany\s+(?:\w+\s+){0,2}recommendations\b", re.IGNORECASE),
]


def is_preference_question(q: str) -> bool:
    """True iff ``q`` matches one of the closed preference/advice-request patterns above."""
    return any(pattern.search(q) for pattern in _PREFERENCE_PATTERNS)


# Variant of QA_SYSTEM (see prompts.py) for preference/advice-request questions only: instead of
# the never-guess, 15-word factual-lookup contract (which makes the model abstain — there's no
# fact to abstain from on an advice request), this answers the request helpfully while staying
# grounded in whatever the memories actually state about the person's preferences/plans/past
# experiences, and names the details it used so the judge (and a human) can see the grounding.
PREF_QA_SYSTEM = (
    "You are this person's assistant. Answer their request helpfully in 2-3 sentences, "
    "grounded in the specific preferences, plans, and past experiences the memories state "
    "about them — name the relevant details you are using. Never invent facts the memories do "
    "not support. If the memories contain nothing relevant, say you don't have their "
    "preferences on record and give one brief general answer."
)


def pref_qa_messages(memory_block: str, question: str, today: str) -> list[dict]:
    """Build the preference QA prompt — mirrors ``prompts.qa_messages`` but with
    ``PREF_QA_SYSTEM`` in place of ``QA_SYSTEM``."""
    parts = [f"Today is {today}."]
    if memory_block:
        parts.append(memory_block)
    parts.append(f"Question: {question}")
    return [
        {"role": "system", "content": PREF_QA_SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
