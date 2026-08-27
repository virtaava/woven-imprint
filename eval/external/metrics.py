"""Metrics for the external benchmark harness: token-F1, abstention detection, summarization."""

from __future__ import annotations

import re
import string
from collections import Counter

_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")
_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def _normalize_answer(text: str) -> str:
    """SQuAD-style normalization: lowercase, strip punctuation, drop articles (a/an/the), and
    collapse whitespace. Used for token-F1 so wording differences like "8 May 2023" vs.
    "May 8, 2023" or "a cat" vs. "the cat" don't cost overlap."""
    s = (text or "").lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _ARTICLE_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def _tokenize(text: str) -> list[str]:
    normalized = _normalize_answer(text)
    return normalized.split() if normalized else []


def token_f1(pred: str, gold: str) -> float:
    """Token-level F1 between a prediction and a gold answer (order-insensitive, bag-of-words,
    SQuAD-style normalized — see :func:`_normalize_answer`)."""
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


_ABSTAIN_PATTERNS = (
    r"not mentioned",
    r"does not mention",
    r"doesn'?t mention",
    r"do not mention",
    r"don'?t mention",
    r"do not have (?:this|that|the|any) information",
    r"don'?t have (?:this|that|the|any) information",
    r"cannot answer",
    r"can'?t answer",
    r"no info\b",
    r"no information",
    r"don'?t know",
    r"do not know",
    r"not stated",
    r"not specified",
    r"not sure",
    r"no mention",
    r"unable to find",
    r"cannot find",
    r"unknown",
)
_ABSTAIN_RE = re.compile("|".join(_ABSTAIN_PATTERNS), re.IGNORECASE)

# A response longer than this is assumed to be supplying a concrete answer rather than merely
# declining to answer, even if it happens to contain an abstention-shaped phrase somewhere.
_MAX_ABSTAIN_WORDS = 12


def is_abstention(response: str) -> bool:
    """True when ``response`` reads as an explicit "I don't know / not mentioned" abstention.

    Rule (deliberately simple, not a full NLI judgment): the response matches one of a fixed set
    of abstention phrase patterns (see ``_ABSTAIN_PATTERNS``) AND has at most
    ``_MAX_ABSTAIN_WORDS`` (12) whitespace-separated words. The word-count cap exists only to
    keep a long, otherwise-substantive answer that happens to contain an abstention-shaped
    fragment from being misclassified; it does not attempt to detect "opens with an abstention
    phrase but then supplies a concrete answer" as its own case — a short reply that does both
    (e.g. "Not mentioned; she adopted a dog") is still counted as an abstention under this rule.
    """
    text = (response or "").strip()
    if not text:
        return False
    if len(text.split()) > _MAX_ABSTAIN_WORDS:
        return False
    return bool(_ABSTAIN_RE.search(text))


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


def _bucket_accuracy(items: list[dict], key: str) -> dict[str, dict]:
    """Group ``items`` by ``str(item[key])`` and compute n/cognitive_accuracy per bucket."""
    buckets: dict[str, dict] = {}
    for it in items:
        bucket = buckets.setdefault(str(it.get(key)), {"n": 0, "correct": 0})
        bucket["n"] += 1
        bucket["correct"] += 1 if it.get("correct") else 0
    return {
        k: {"n": b["n"], "cognitive_accuracy": b["correct"] / b["n"] if b["n"] else 0.0}
        for k, b in buckets.items()
    }


def summarize_plus(items: list[dict]) -> dict:
    """Aggregate LoCoMo-Plus Cognitive judged probe records.

    Each item is expected to carry ``relation_type``, ``time_gap``, ``correct`` (bool),
    ``prompt_tokens_est`` (int), ``llm_calls`` (int), and ``seconds`` (float). Reports overall
    ``cognitive_accuracy`` (mean ``correct``) plus per-``relation_type`` and per-``time_gap``
    breakdowns, and mean tokens/LLM-calls/seconds per probe.
    """
    n = len(items)
    overall = sum(1 for it in items if it.get("correct")) / n if n else 0.0
    tokens = [float(it.get("prompt_tokens_est", 0)) for it in items]
    seconds = [float(it.get("seconds", 0.0)) for it in items]
    calls = [float(it.get("llm_calls", 0)) for it in items]

    return {
        "n_probes": n,
        "cognitive_accuracy": overall,
        "per_relation_type": _bucket_accuracy(items, "relation_type"),
        "per_time_gap": _bucket_accuracy(items, "time_gap"),
        "mean_prompt_tokens_est": sum(tokens) / n if n else 0.0,
        "mean_seconds": sum(seconds) / n if n else 0.0,
        "mean_llm_calls": sum(calls) / n if n else 0.0,
    }
