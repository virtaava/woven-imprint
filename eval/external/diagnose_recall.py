"""Diagnose WHY memory mode misses on LoCoMo, at the judged-answer level.

Reads the already-judged results of the ``locomo-mem-v1`` run
(``eval/results/external_locomo-mem-v1.json``) plus that run's per-conversation
SQLite DBs (``eval/external/runs/locomo-mem-v1/<conv_id>.db``), and for every
``kind == "qa"`` question:

- maps its LoCoMo ``evidence`` dia_ids to the dataset's turn text (via
  ``eval.external.locomo.load_locomo``, so image-caption folding matches the
  original run exactly);
- checks whether a memory containing that evidence text is stored (and
  whether it's ``active`` or only reachable through an ``archived``/
  ``contradicted`` row);
- re-runs real ``MemoryRetriever.retrieve()`` against a **copy** of the run's
  DB (never the original) to see whether an evidence-containing memory made
  the top-20, and if not, how far down the ranking it sits;
- checks whether the pinned/facts block or the top-20 memory text already
  contained the gold answer's text;
- classifies the model's response as an abstention or a wrong answer.

Every ``kind == "qa"`` question is processed (not just ``WRONG`` ones) so
retrieval-recall stats have a CORRECT-answer baseline and K-sensitivity can be
computed across the whole question set.

No LLM is called (a ``FakeLLM`` stands in for ``Engine``'s required LLM
provider — retrieval never calls it). The embedding server at
``127.0.0.1:11801`` *is* called, same as the original run.

Outputs:
    eval/external/runs/diagnostics/recall_diagnostic.json  (rows + aggregates)
    eval/external/runs/diagnostics/recall_diagnostic.md    (tables, human-readable)

Run: ``python -m eval.external.diagnose_recall`` (repo root, with the venv
that has ``woven_imprint`` installed).
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from woven_imprint import clock
from woven_imprint.engine import Engine

from tests.helpers import FakeLLM

from . import metrics
from .common import DATA_DIR, RESULTS_DIR, RUNS_DIR, embedder
from .locomo import load_locomo

RUN_ID = "locomo-mem-v1"
RESULTS_FILE = RESULTS_DIR / f"external_{RUN_ID}.json"
RUN_DIR = RUNS_DIR / RUN_ID
DIAG_DIR = RUNS_DIR / "diagnostics"
LOCOMO_PATH = DATA_DIR / "locomo.json"

RETRIEVE_LIMIT = 200  # `retrieve()`'s fused ranking doesn't depend on `limit` (see module docs
# on MemoryRetriever.retrieve: `limit` only slices the already-fully-computed fused order), so
# one limit=200 call per question yields the top-20 answer AND the rank-200 scan for free.
TOP_K = 20
K_LIST = (10, 20, 50, 100)
FUZZY_THRESHOLD = 0.8  # >=80% of the evidence turn's words found in the candidate's words

_WORD_RE = re.compile(r"[a-z0-9]+")


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _fuzzy_hit(evidence_words: list[str], content_words: set[str]) -> bool:
    if not evidence_words:
        return False
    hits = sum(1 for w in evidence_words if w in content_words)
    return (hits / len(evidence_words)) >= FUZZY_THRESHOLD


def _copy_db(conv_id: str) -> Path:
    """Copy ``<conv_id>.db`` (+ -wal/-shm if present) from the run dir into DIAG_DIR.

    Always re-copies from the source run dir so the script is safely re-runnable and never
    drifts from the original run's data; the original is never opened directly.
    """
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    src = RUN_DIR / f"{conv_id}.db"
    dst = DIAG_DIR / f"{conv_id}.db"
    shutil.copy2(src, dst)
    for suffix in ("-wal", "-shm"):
        s = RUN_DIR / f"{conv_id}.db{suffix}"
        d = DIAG_DIR / f"{conv_id}.db{suffix}"
        if s.exists():
            shutil.copy2(s, d)
        elif d.exists():
            d.unlink()
    return dst


def _load_all_memory_rows(db_path: Path, character_id: str) -> list[dict]:
    """Every memory row (any tier/status) for ``character_id``, read directly via sqlite3 —
    independent of the engine's active-only retrieval path, so evidence can be located even in
    archived/contradicted rows."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, tier, status, content FROM memories WHERE character_id = ?",
            (character_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _locate_evidence(evidence_text: str, rows: list[dict]) -> dict:
    """Classify where ``evidence_text`` lives among ``rows`` (all memory rows for the character).

    Returns a dict with exact/fuzzy match id lists and a status classification:
    ``stored_active`` (an exact-or-fuzzy match with status=='active' exists), ``stored_archived_only``
    (matches exist but none are active), ``not_stored`` (no match at all).
    """
    evidence_text = evidence_text.strip()
    ev_words = _words(evidence_text)
    exact_ids, fuzzy_ids = [], []
    for row in rows:
        if not evidence_text:
            break
        content = row["content"]
        if evidence_text in content:
            exact_ids.append(row["id"])
        elif _fuzzy_hit(ev_words, set(_words(content))):
            fuzzy_ids.append(row["id"])
    combined = {r["id"]: r for r in rows if r["id"] in exact_ids or r["id"] in fuzzy_ids}
    if not combined:
        turn_class = "not_stored"
    elif any(r["status"] == "active" for r in combined.values()):
        turn_class = "stored_active"
    else:
        turn_class = "stored_archived_only"
    return {
        "exact_ids": exact_ids,
        "fuzzy_ids": fuzzy_ids,
        "statuses": {mid: r["status"] for mid, r in combined.items()},
        "tiers": {mid: r["tier"] for mid, r in combined.items()},
        "turn_class": turn_class,
    }


def _rank_bucket(rank: int | None) -> str:
    if rank is None:
        return ">200"
    if rank <= 50:
        return "<=50"
    if rank <= 200:
        return "<=200"
    return ">200"


def _classify_question(evidence_details: list[dict]) -> dict:
    if not evidence_details:
        return {
            "evidence_class": "no_evidence_resolved",
            "any_retrieved": False,
            "best_rank_200": None,
        }
    any_active = any(d["turn_class"] == "stored_active" for d in evidence_details)
    any_archived_only = any(d["turn_class"] == "stored_archived_only" for d in evidence_details)
    any_retrieved = any(d["retrieved_in_top20"] for d in evidence_details)
    ranks = [d["rank_200"] for d in evidence_details if d["rank_200"] is not None]
    best_rank_200 = min(ranks) if ranks else None

    if any_active and any_retrieved:
        evidence_class = "retrieved"
    elif any_active:
        evidence_class = "stored_active_not_retrieved"
    elif any_archived_only:
        evidence_class = "stored_archived_only"
    else:
        evidence_class = "not_stored"
    return {
        "evidence_class": evidence_class,
        "any_retrieved": any_retrieved,
        "best_rank_200": best_rank_200,
    }


def _process_conversation(
    conv_id: str,
    qa_records: list[dict],
    dataset_conv,
    question_by_qid: dict[str, Any],
    db_path: Path,
    raw_rows: list[dict],
) -> list[dict]:
    dia_to_turn = {
        t.dia_id: t for s in dataset_conv.sessions for t in s.turns if t.dia_id is not None
    }

    engine = Engine(db_path=str(db_path), llm=FakeLLM(), embedding=embedder())
    char = engine.get_character(conv_id)
    char.background = False
    char.parallel = False
    char.enforce_consistency = False

    rows: list[dict] = []
    for rec in qa_records:
        qid = rec["qid"]
        q = question_by_qid.get(qid)
        if q is None:
            # Should not happen (every qa record was generated from this dataset), but don't
            # let a mismatch crash the whole run.
            rows.append({**rec, "diagnostic_error": "qid not found in dataset"})
            continue

        clock.override(q.asked_at)
        pinned, pinned_ids = char._format_pinned_block()
        facts_text = char._format_facts_block(dataset_conv.user_name, pinned_ids)
        mems200 = char.retriever.retrieve(
            q.question, limit=RETRIEVE_LIMIT, relationship_target=dataset_conv.user_name
        )
        rank_by_id = {m["id"]: i + 1 for i, m in enumerate(mems200)}

        evidence_details = []
        unresolved_dia_ids = []
        for dia_id in q.evidence:
            turn = dia_to_turn.get(dia_id)
            if turn is None:
                unresolved_dia_ids.append(dia_id)
                continue
            loc = _locate_evidence(turn.text, raw_rows)
            match_ids = set(loc["exact_ids"]) | set(loc["fuzzy_ids"])
            ranks = [rank_by_id[mid] for mid in match_ids if mid in rank_by_id]
            rank_200 = min(ranks) if ranks else None
            evidence_details.append(
                {
                    "dia_id": dia_id,
                    "turn_text": turn.text,
                    **loc,
                    "rank_200": rank_200,
                    "rank_bucket": _rank_bucket(rank_200),
                    "retrieved_in_top20": rank_200 is not None and rank_200 <= TOP_K,
                }
            )

        gold = rec.get("gold", "") or ""
        response = rec.get("response", "") or ""
        gold_in_facts = bool(gold) and gold.lower() in facts_text.lower()
        gold_in_top20 = bool(gold) and any(
            gold.lower() in m["content"].lower() for m in mems200[:TOP_K]
        )
        abstained = metrics.is_abstention(response)

        qclass = _classify_question(evidence_details)

        rows.append(
            {
                "qid": qid,
                "conv_id": conv_id,
                "category": rec.get("category"),
                "kind": rec.get("kind"),
                "label": rec.get("label"),
                "correct": rec.get("correct"),
                "question": rec.get("question"),
                "gold": gold,
                "response": response,
                "evidence_dia_ids": q.evidence,
                "unresolved_dia_ids": unresolved_dia_ids,
                "evidence_details": evidence_details,
                "evidence_class": qclass["evidence_class"],
                "any_retrieved": qclass["any_retrieved"],
                "best_rank_200": qclass["best_rank_200"],
                "best_rank_bucket": _rank_bucket(qclass["best_rank_200"]),
                "gold_in_facts": gold_in_facts,
                "gold_in_top20": gold_in_top20,
                "abstained": abstained,
                "response_class": "abstained" if abstained else "answered_wrong",
            }
        )

    return rows


def _memory_tier_status_counts(rows: list[dict]) -> Counter:
    counts: Counter = Counter()
    for r in rows:
        counts[(r["tier"], r["status"])] += 1
    return counts


def _recall_at_k(rows: list[dict]) -> dict[int, float]:
    """recall@K over ALL qa questions with at least one resolved evidence turn: fraction where
    some evidence-containing (any-status) memory ranks <= K in the limit=200 retrieval."""
    with_evidence = [r for r in rows if r["evidence_class"] != "no_evidence_resolved"]
    out = {}
    for k in K_LIST:
        hits = sum(
            1 for r in with_evidence if r["best_rank_200"] is not None and r["best_rank_200"] <= k
        )
        out[k] = hits / len(with_evidence) if with_evidence else 0.0
    return out


def _matrix(rows: list[dict]) -> dict:
    counts: Counter = Counter()
    rank_buckets: Counter = Counter()
    for r in rows:
        counts[(r["evidence_class"], r["response_class"])] += 1
        if r["evidence_class"] == "stored_active_not_retrieved":
            rank_buckets[r["best_rank_bucket"]] += 1
    return {
        "by_evidence_and_response": {f"{k[0]}|{k[1]}": v for k, v in counts.items()},
        "stored_active_not_retrieved_rank_buckets": dict(rank_buckets),
    }


def _not_stored_examples(rows: list[dict], n: int = 10) -> list[dict]:
    out = []
    for r in rows:
        if r["evidence_class"] != "not_stored":
            continue
        for d in r["evidence_details"]:
            if d["turn_class"] == "not_stored":
                out.append(
                    {
                        "qid": r["qid"],
                        "question": r["question"],
                        "gold": r["gold"],
                        "response": r["response"],
                        "dia_id": d["dia_id"],
                        "turn_text": d["turn_text"],
                        "turn_len": len(d["turn_text"]),
                        "is_photo_caption": "(shared a photo:" in d["turn_text"],
                    }
                )
                break
        if len(out) >= n:
            break
    return out


def build_diagnostics() -> dict:
    results = json.loads(RESULTS_FILE.read_text())
    all_conversations = results["conversations"]
    qa_records = [r for r in all_conversations if r.get("kind") == "qa"]

    dataset_convs = load_locomo(LOCOMO_PATH)
    dataset_by_id = {c.conv_id: c for c in dataset_convs}
    question_by_qid = {q.qid: q for c in dataset_convs for q in c.questions}

    qa_by_conv: dict[str, list[dict]] = defaultdict(list)
    for rec in qa_records:
        qa_by_conv[rec["conv_id"]].append(rec)

    all_rows: list[dict] = []
    all_raw_memory_rows: list[dict] = []
    buffer_total = 0
    buffer_archived = 0

    for conv_id in sorted(qa_by_conv):
        dataset_conv = dataset_by_id.get(conv_id)
        if dataset_conv is None:
            continue
        db_path = _copy_db(conv_id)
        raw_rows = _load_all_memory_rows(db_path, conv_id)
        all_raw_memory_rows.extend(raw_rows)
        for r in raw_rows:
            if r["tier"] == "buffer":
                buffer_total += 1
                if r["status"] == "archived":
                    buffer_archived += 1

        rows = _process_conversation(
            conv_id, qa_by_conv[conv_id], dataset_conv, question_by_qid, db_path, raw_rows
        )
        all_rows.extend(rows)

    wrong_rows = [r for r in all_rows if r["label"] == "WRONG"]
    correct_rows = [r for r in all_rows if r["label"] == "CORRECT"]

    tier_status_counts = _memory_tier_status_counts(all_raw_memory_rows)

    aggregates: dict[str, Any] = {
        "n_qa_total": len(all_rows),
        "n_wrong": len(wrong_rows),
        "n_correct": len(correct_rows),
        "overall": {
            "wrong": _matrix(wrong_rows),
            "correct_baseline": {
                "retrieved_rate": (
                    sum(1 for r in correct_rows if r["any_retrieved"]) / len(correct_rows)
                    if correct_rows
                    else 0.0
                ),
                "evidence_class_counts": dict(Counter(r["evidence_class"] for r in correct_rows)),
            },
            "recall_at_k": _recall_at_k(all_rows),
        },
        "per_category": {},
        "memories_by_tier_status": {f"{k[0]}/{k[1]}": v for k, v in tier_status_counts.items()},
        "buffer_lifecycle": {
            "buffer_rows_total": buffer_total,
            "buffer_rows_archived": buffer_archived,
            "buffer_archived_fraction": (buffer_archived / buffer_total) if buffer_total else 0.0,
        },
        "not_stored_examples": _not_stored_examples(wrong_rows, n=10),
        "retrieval_candidate_rule": (
            "MemoryRetriever.retrieve() candidates come only from "
            "SQLiteStorage.get_memories(status='active') and fts_search(... status='active'); "
            "archived and contradicted rows of any tier are never scored or returned, "
            "regardless of `limit`. All active-status memories (bedrock/core/buffer alike) are "
            "eligible up to memory.max_candidates (default 5000)."
        ),
    }

    for cat in sorted({r["category"] for r in all_rows}, key=lambda c: (len(c), c)):
        cat_wrong = [r for r in wrong_rows if r["category"] == cat]
        cat_correct = [r for r in correct_rows if r["category"] == cat]
        cat_all = [r for r in all_rows if r["category"] == cat]
        aggregates["per_category"][cat] = {
            "n": len(cat_all),
            "n_wrong": len(cat_wrong),
            "n_correct": len(cat_correct),
            "wrong": _matrix(cat_wrong),
            "correct_baseline_retrieved_rate": (
                sum(1 for r in cat_correct if r["any_retrieved"]) / len(cat_correct)
                if cat_correct
                else 0.0
            ),
            "recall_at_k": _recall_at_k(cat_all),
        }

    return {"rows": all_rows, "aggregates": aggregates}


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def render_markdown(data: dict) -> str:
    agg = data["aggregates"]
    lines = ["# LoCoMo memory-mode recall diagnostic", ""]
    lines.append(
        f"qa questions: {agg['n_qa_total']} (WRONG {agg['n_wrong']}, CORRECT {agg['n_correct']})"
    )
    lines.append("")

    lines.append("## Retrieval candidate rule (from source)")
    lines.append("")
    lines.append(agg["retrieval_candidate_rule"])
    lines.append("")

    lines.append("## Memories by tier/status (all diagnosed conversations, summed)")
    lines.append("")
    lines.append("| tier/status | count |")
    lines.append("|---|---:|")
    for k, v in sorted(agg["memories_by_tier_status"].items()):
        lines.append(f"| {k} | {v} |")
    lines.append("")
    bl = agg["buffer_lifecycle"]
    lines.append(
        f"Buffer rows: {bl['buffer_rows_total']} total, {bl['buffer_rows_archived']} archived "
        f"({_fmt_pct(bl['buffer_archived_fraction'])})."
    )
    lines.append("")

    lines.append("## Overall WRONG classification matrix (evidence_class | response_class)")
    lines.append("")
    lines.append("| evidence_class | response_class | count |")
    lines.append("|---|---|---:|")
    for k, v in sorted(agg["overall"]["wrong"]["by_evidence_and_response"].items()):
        ec, rc = k.split("|")
        lines.append(f"| {ec} | {rc} | {v} |")
    lines.append("")
    lines.append("`stored_active_not_retrieved` rank_200 distribution:")
    lines.append("")
    for k, v in sorted(agg["overall"]["wrong"]["stored_active_not_retrieved_rank_buckets"].items()):
        lines.append(f"- {k}: {v}")
    lines.append("")

    lines.append("## CORRECT-answer retrieval baseline")
    lines.append("")
    cb = agg["overall"]["correct_baseline"]
    lines.append(f"retrieved_rate: {_fmt_pct(cb['retrieved_rate'])}")
    lines.append("")
    for k, v in sorted(cb["evidence_class_counts"].items()):
        lines.append(f"- {k}: {v}")
    lines.append("")

    lines.append("## K-sensitivity — recall@K of evidence memories, all qa questions")
    lines.append("")
    lines.append("| K | recall |")
    lines.append("|---:|---:|")
    for k in K_LIST:
        lines.append(f"| {k} | {_fmt_pct(agg['overall']['recall_at_k'][k])} |")
    lines.append("")

    lines.append("## Per-category breakdown")
    lines.append("")
    for cat, c in agg["per_category"].items():
        lines.append(
            f"### Category {cat} (n={c['n']}, wrong={c['n_wrong']}, correct={c['n_correct']})"
        )
        lines.append("")
        lines.append(
            f"correct-baseline retrieved_rate: {_fmt_pct(c['correct_baseline_retrieved_rate'])}"
        )
        lines.append("")
        lines.append("| evidence_class | response_class | count |")
        lines.append("|---|---|---:|")
        for k, v in sorted(c["wrong"]["by_evidence_and_response"].items()):
            ec, rc = k.split("|")
            lines.append(f"| {ec} | {rc} | {v} |")
        lines.append("")
        lines.append(
            "recall@K: " + ", ".join(f"@{k}={_fmt_pct(c['recall_at_k'][k])}" for k in K_LIST)
        )
        lines.append("")

    lines.append("## `not_stored` examples (up to 10, from WRONG answers)")
    lines.append("")
    for ex in agg["not_stored_examples"]:
        lines.append(f"- **{ex['qid']}** — {ex['question']!r}")
        lines.append(f"  - gold: {ex['gold']!r}, response: {ex['response']!r}")
        lines.append(
            f"  - evidence turn ({ex['dia_id']}, len={ex['turn_len']}, "
            f"photo_caption={ex['is_photo_caption']}): {ex['turn_text']!r}"
        )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    data = build_diagnostics()
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    (DIAG_DIR / "recall_diagnostic.json").write_text(json.dumps(data, indent=2, default=str))
    (DIAG_DIR / "recall_diagnostic.md").write_text(render_markdown(data))
    print(f"wrote {DIAG_DIR / 'recall_diagnostic.json'}")
    print(f"wrote {DIAG_DIR / 'recall_diagnostic.md'}")


if __name__ == "__main__":
    main()
