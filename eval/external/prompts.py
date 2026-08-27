"""QA and judge prompts for the external benchmark harness.

``QA_SYSTEM``/``JUDGE_SYSTEM``/``PLUS_COGNITIVE_JUDGE`` are quoted verbatim in
``docs/BENCHMARKS.md`` (Task 6) — do not reword them without updating that doc.

``PLUS_COGNITIVE_JUDGE`` is the "Cognitive" template from upstream
``xjtuleeyf/Locomo-Plus``'s ``evaluation_framework/task_eval/prompt.py``
(``PROMPT_TEMPLATES["Cognitive"]``), copied character-for-character (fetched
2026-08-27). Upstream fills it via ``template.format(gold=..., pred=...,
evidence=...)`` and sends the whole filled string as a single flat prompt to
``call_model`` — no separate system role — so :func:`plus_judge_messages`
does the same (one user-role message).
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
    "question. Relative time expressions in the response are interpreted relative to the reference date."
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


def judge_messages(question: str, gold: str, response: str, asked_on: str) -> list[dict]:
    """Build the judge prompt; asks for JSON ``{"label": "CORRECT"|"WRONG", "reason": str}``.

    ``asked_on`` is the question's ``asked_at`` date (``q.asked_at.date().isoformat()``) — the
    judge needs it to resolve relative time expressions in the response (e.g. "last week") the
    same way the QA prompt's "Today is {today}" line let the model resolve them when answering.
    """
    user = (
        f"Question: {question}\n"
        f"Gold answer: {gold}\n"
        f"Model response: {response}\n"
        f"Reference date (when the question was asked): {asked_on}\n\n"
        'Reply with JSON only: {"label": "CORRECT" or "WRONG", "reason": "..."}'
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


# Verbatim upstream PROMPT_TEMPLATES["Cognitive"] from xjtuleeyf/Locomo-Plus
# evaluation_framework/task_eval/prompt.py — do not reword.
PLUS_COGNITIVE_JUDGE = """
You are a Memory Awareness Judge.
Your task: Judge whether the Model Prediction considers or is linked to the Evidence. If there is a clear connection, the answer is correct (score 1); if not, it is wrong (no score).

Labels:
- "correct": The prediction explicitly or implicitly reflects/uses the evidence (memory or constraint). Give 1 point.
- "wrong": The prediction does not show such a link to the evidence. No point.

Memory/Evidence:
{evidence}

Model Prediction:
{pred}

Return your judgment strictly in JSON format:
{{"label": "correct"|"wrong", "reason": "<Does the prediction relate to the evidence?>"}}
"""


def plus_judge_messages(evidence: str, pred: str) -> list[dict]:
    """Build the LoCoMo-Plus Cognitive judge prompt.

    Upstream sends the whole filled template as a single flat prompt string (no separate
    system role) via ``call_model(prompt)`` — this mirrors that with one user-role message.
    """
    return [{"role": "user", "content": PLUS_COGNITIVE_JUDGE.format(evidence=evidence, pred=pred)}]
