"""Metrics for the external benchmark harness: token-F1, abstention detection, summarization."""

from __future__ import annotations

import re
from collections import Counter

_WORD_RE = re.compile(r"[a-z0-9]+")

_ABSTAIN_RE = re.compile(
    r"not mentioned|don'?t know|do not know|no information|cannot find|unable to find|"
    r"not stated|not specified|not sure|no mention|unknown",
    re.IGNORECASE,
)


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def token_f1(pred: str, gold: str) -> float:
    """Token-level F1 between a prediction and a gold answer (order-insensitive, bag-of-words)."""
    pred_tokens = _tokenize(pred)
    gold_tokens = _tokenize(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def is_abstention(response: str) -> bool:
    """True when ``response`` reads as an explicit "I don't know / not mentioned" abstention."""
    return bool(_ABSTAIN_RE.search(response or ""))


def summarize(items: list[dict]) -> dict:
    """Aggregate per-question records into per-category and overall metrics.

    Each item is expected to carry ``category``, ``kind`` ("qa" | "adversarial" | "abstain"),
    ``correct`` (bool), ``f1`` (float), and ``prompt_tokens_est`` (int). The overall J-score and
    F1 are computed over ``kind == "qa"`` items only — adversarial (LoCoMo category 5) and
    abstain (LongMemEval ``_abs``) items are reported separately as accuracy, per the Mem0/Zep
    convention (a "correct" abstention is not comparable to a "correct" factual recall).
    """
    per_category: dict[str, dict] = {}
    for it in items:
        cat = str(it.get("category"))
        bucket = per_category.setdefault(cat, {"n": 0, "correct": 0, "f1_sum": 0.0})
        bucket["n"] += 1
        bucket["correct"] += 1 if it.get("correct") else 0
        bucket["f1_sum"] += float(it.get("f1", 0.0))

    per_category_out = {
        cat: {
            "n": b["n"],
            "j": b["correct"] / b["n"] if b["n"] else 0.0,
            "f1": b["f1_sum"] / b["n"] if b["n"] else 0.0,
        }
        for cat, b in per_category.items()
    }

    qa_items = [it for it in items if it.get("kind") == "qa"]
    overall_j = sum(1 for it in qa_items if it.get("correct")) / len(qa_items) if qa_items else 0.0
    overall_f1 = (
        sum(float(it.get("f1", 0.0)) for it in qa_items) / len(qa_items) if qa_items else 0.0
    )

    adversarial_items = [it for it in items if it.get("kind") == "adversarial"]
    adversarial_accuracy = (
        sum(1 for it in adversarial_items if it.get("correct")) / len(adversarial_items)
        if adversarial_items
        else None
    )

    abstain_items = [it for it in items if it.get("kind") == "abstain"]
    abstain_accuracy = (
        sum(1 for it in abstain_items if it.get("correct")) / len(abstain_items)
        if abstain_items
        else None
    )

    prompt_tokens = [float(it.get("prompt_tokens_est", 0)) for it in items]
    mean_prompt_tokens = sum(prompt_tokens) / len(prompt_tokens) if prompt_tokens else 0.0

    return {
        "n_questions": len(items),
        "overall_j": overall_j,
        "overall_f1": overall_f1,
        "per_category": per_category_out,
        "adversarial_accuracy": adversarial_accuracy,
        "abstain_accuracy": abstain_accuracy,
        "mean_prompt_tokens_est": mean_prompt_tokens,
    }
