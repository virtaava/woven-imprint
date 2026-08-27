"""QA and judge prompts for the external benchmark harness.

Both strings are quoted verbatim in ``docs/BENCHMARKS.md`` — do not reword
them without updating that doc.
"""

from __future__ import annotations

QA_SYSTEM = (
    "You answer questions about a person using ONLY the memories provided. Be concise (at most 15 words). "
    "Convert relative dates to absolute dates. If the memories do not contain the answer, reply exactly: Not mentioned"
)

JUDGE_SYSTEM = (
    "You are an expert grader. Given a question, a gold answer and a model response, decide whether the "
    "response conveys the same information as the gold answer. Be lenient: accept paraphrases, partial dates "
    "that agree with the gold date, equivalent relative/absolute time expressions, and extra correct detail. "
    "Mark WRONG if the response contradicts the gold answer, is missing the key fact, or answers a different "
    "question."
)


def qa_messages(memory_block: str, question: str, today: str) -> list[dict]:
    """Build the QA prompt: today's date, the memory/facts block (if any), then the question."""
    parts = [f"Today is {today}."]
    if memory_block:
        parts.append(memory_block)
    parts.append(f"Question: {question}")
    return [
        {"role": "system", "content": QA_SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def judge_messages(question: str, gold: str, response: str) -> list[dict]:
    """Build the judge prompt; asks for JSON ``{"label": "CORRECT"|"WRONG", "reason": str}``."""
    user = (
        f"Question: {question}\n"
        f"Gold answer: {gold}\n"
        f"Model response: {response}\n\n"
        'Reply with JSON only: {"label": "CORRECT" or "WRONG", "reason": "..."}'
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]
