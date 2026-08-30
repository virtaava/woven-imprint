"""Offline retrieval-ranking experiments on LoCoMo (Tier 3c recall investigation).

Answers: which ranking changes raise evidence recall@K on LoCoMo, cheaply
(no LLM calls), before spending brain hours on an actual memory-mode run?

Reuses ``diagnose_recall.py``'s DB-copying and evidence->memory-id mapping
helpers, and the product's own ``cosine_matrix``/``reciprocal_rank_fusion``/
``_recency_score`` functions from ``src/`` (not reimplemented — only the
*orchestration* that combines them with variable weights/embeddings is
reimplemented here, since ``MemoryConfig`` only supports one fixed weight set
per process and we need many in one run).

Candidate set: for every query, the candidate pool is "all active memories
for that character" (`SQLiteStorage.get_memories(limit=max_candidates=5000)`,
which is every active memory since no conversation here has >1500) — this
already equals what `MemoryRetriever.retrieve()` scores, since its
`fts_search(limit=50)` supplement only ever adds ids already active and thus
already in the pool. Only the *keyword-ranked list's length* (how many of
that pool get an FTS rank at all, feeding both the keyword RRF list and the
relevance gate's keyword-eligible set) varies between V0-family (50) and V3
(200).

Variants (see MEMORY.md task brief / commit message for full spec):
  V0  baseline product fusion (RRF, all weights 1.0, rrf_k=60, gate on)
  V1  semantic-only (cosine sort, no RRF)
  V2  RRF, weight_recency=weight_importance=0
  V3  V2 + FTS candidate list 50->200
  V4a query-side "search_query: " prefix, V0 fusion (docs unprefixed)
  V4b query-side prefix, V1 (semantic-only) fusion
  V5a both sides prefixed ("search_document: "/"search_query: "), V1 fusion
  V5b both sides prefixed, V2 fusion
  V6a contextualized buffer docs ("search_document: [date] speaker: content",
      non-buffer docs fall back to plain search_document-prefixed so only the
      buffer date/speaker addition is isolated vs V5), V1 fusion
  V6b contextualized buffer docs, V2 fusion
  V7  (rrf_k in {10, 60}) RRF of [V5-style semantic, FTS-keyword] only —
      no recency/importance/relationship lists at all

Run (repo root, product venv): ``./venv/bin/python -m eval.external.ranking_experiments``

Outputs under ``eval/external/runs/diagnostics/ranking/`` (gitignored, whole
``eval/external/runs/`` tree is): copied per-conv DBs, an embedding cache
(``embedding_cache/*.pkl``, so reruns after a code change don't re-embed),
``ranking_experiments_results.json`` and ``ranking_experiments_report.md``.
"""

from __future__ import annotations

import os

# Must be set before numpy is imported. Without this, numpy's BLAS backend spins up a full
# thread pool for every cosine_matrix() call — with >100k such calls across all variants x
# conversations x questions, most of them tiny (one query vector against ~1000 doc vectors),
# thread-launch/teardown overhead dominates and turns a ~6-minute run into one that doesn't
# finish in 30 minutes. Single-threaded BLAS is faster here since each matmul is too small to
# benefit from parallelism anyway.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import pickle
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from woven_imprint import clock
from woven_imprint.config import get_config
from woven_imprint.llm.base import LLMProvider
from woven_imprint.memory.retrieval import _get_tier_boosts, _recency_score, cosine_matrix
from woven_imprint.storage.sqlite import SQLiteStorage
from woven_imprint.utils.rrf import reciprocal_rank_fusion

from tests.helpers import FakeLLM

from .common import DATA_DIR, RESULTS_DIR, RUNS_DIR, embedder
from .diagnose_recall import _copy_db, _load_all_memory_rows, _locate_evidence
from .locomo import load_locomo

RUN_ID = "locomo-mem-v2"
RUN_DIR = RUNS_DIR / RUN_ID
OUT_DIR = RUNS_DIR / "diagnostics" / "ranking"
CACHE_DIR = OUT_DIR / "embedding_cache"
LOCOMO_PATH = DATA_DIR / "locomo.json"
RESULTS_FILE = RESULTS_DIR / f"external_{RUN_ID}.json"

K_LIST = (10, 20, 50)
VALIDATION_N = 50
SEED = 20260830
EMBED_BATCH = 256
SEARCH_QUERY_PREFIX = "search_query: "
SEARCH_DOC_PREFIX = "search_document: "


# ── Variant definitions ──────────────────────────────────────────────────


@dataclass
class Variant:
    name: str
    fusion: str  # "rrf" | "semantic_only" | "rrf2"
    keyword_limit: int = 50
    weight_semantic: float = 1.0
    weight_keyword: float = 1.0
    weight_recency: float = 1.0
    weight_importance: float = 1.0
    weight_relationship: float = 1.0
    rrf_k: int = 60
    query_prefix: str | None = None  # None = stored/plain query embedding
    doc_variant: str = "stored"  # "stored" | "search_document" | "contextualized"
    relevance_gate: bool = True
    requires_reingest: bool = False  # True = needs re-embedding stored memories in production


VARIANTS: list[Variant] = [
    Variant("V0_baseline", fusion="rrf"),
    Variant("V1_semantic_only", fusion="semantic_only"),
    Variant(
        "V2_rrf_no_recency_importance", fusion="rrf", weight_recency=0.0, weight_importance=0.0
    ),
    Variant(
        "V3_rrf_no_rec_imp_fts200",
        fusion="rrf",
        weight_recency=0.0,
        weight_importance=0.0,
        keyword_limit=200,
    ),
    Variant("V4a_query_prefix_v0fusion", fusion="rrf", query_prefix=SEARCH_QUERY_PREFIX),
    Variant("V4b_query_prefix_v1fusion", fusion="semantic_only", query_prefix=SEARCH_QUERY_PREFIX),
    Variant(
        "V5a_both_prefix_v1fusion",
        fusion="semantic_only",
        query_prefix=SEARCH_QUERY_PREFIX,
        doc_variant="search_document",
        requires_reingest=True,
    ),
    Variant(
        "V5b_both_prefix_v2fusion",
        fusion="rrf",
        weight_recency=0.0,
        weight_importance=0.0,
        query_prefix=SEARCH_QUERY_PREFIX,
        doc_variant="search_document",
        requires_reingest=True,
    ),
    Variant(
        "V6a_contextualized_v1fusion",
        fusion="semantic_only",
        query_prefix=SEARCH_QUERY_PREFIX,
        doc_variant="contextualized",
        requires_reingest=True,
    ),
    Variant(
        "V6b_contextualized_v2fusion",
        fusion="rrf",
        weight_recency=0.0,
        weight_importance=0.0,
        query_prefix=SEARCH_QUERY_PREFIX,
        doc_variant="contextualized",
        requires_reingest=True,
    ),
    Variant(
        "V7_rrf2_semantic_fts_k10",
        fusion="rrf2",
        rrf_k=10,
        query_prefix=SEARCH_QUERY_PREFIX,
        doc_variant="search_document",
        requires_reingest=True,
    ),
    Variant(
        "V7_rrf2_semantic_fts_k60",
        fusion="rrf2",
        rrf_k=60,
        query_prefix=SEARCH_QUERY_PREFIX,
        doc_variant="search_document",
        requires_reingest=True,
    ),
]


# ── Embedding cache helpers ───────────────────────────────────────────────


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}


def _save_cache(path: Path, cache: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(cache, f)
    tmp.replace(path)


def _embed_missing(
    cache: dict[str, np.ndarray],
    items: list[tuple[str, str]],
    embed_fn,
    batch_size: int = EMBED_BATCH,
) -> int:
    """items: (key, text) pairs. Mutates cache for missing keys. Returns #embedded."""
    missing = [(k, t) for k, t in items if k not in cache]
    for i in range(0, len(missing), batch_size):
        chunk = missing[i : i + batch_size]
        vecs = embed_fn([t for _, t in chunk])
        for (k, _), v in zip(chunk, vecs):
            cache[k] = np.asarray(v, dtype=np.float32)
    return len(missing)


# ── Conversation prep ─────────────────────────────────────────────────────


@dataclass
class ConvData:
    conv_id: str
    storage: SQLiteStorage
    active: list[dict]  # sorted by rowid ascending, same as retrieval.py's all_memories
    active_by_id: dict[str, dict]
    dataset_conv: Any
    dia_to_turn: dict[str, Any]


@dataclass
class QAItem:
    qid: str
    conv_id: str
    category: str
    question: str
    asked_at: Any
    evidence_active_ids: set[str]
    excluded: bool  # True == diagnose_recall's "no_evidence_resolved"
    label: str | None = None


def _prepare_conv(conv_id: str, dataset_conv: Any) -> tuple[ConvData, list[dict]]:
    db_path = _copy_db(conv_id, run_dir=RUN_DIR, diag_dir=OUT_DIR)
    raw_rows = _load_all_memory_rows(db_path, conv_id)
    storage = SQLiteStorage(str(db_path))
    mem_cfg = get_config().memory
    active = storage.get_memories(conv_id, limit=mem_cfg.max_candidates)
    active.sort(key=lambda m: m.get("rowid", 0))  # mirrors retrieval.py's all_memories.sort
    active_by_id = {m["id"]: m for m in active}
    dia_to_turn = {
        t.dia_id: t for s in dataset_conv.sessions for t in s.turns if t.dia_id is not None
    }
    return (
        ConvData(conv_id, storage, active, active_by_id, dataset_conv, dia_to_turn),
        raw_rows,
    )


def _build_qa_items(
    conv: ConvData, raw_rows: list[dict], label_by_qid: dict[str, str]
) -> list[QAItem]:
    items: list[QAItem] = []
    for q in conv.dataset_conv.questions:
        if q.kind != "qa":
            continue
        resolved_any = False
        active_ids: set[str] = set()
        for dia_id in q.evidence:
            turn = conv.dia_to_turn.get(dia_id)
            if turn is None:
                continue
            resolved_any = True
            loc = _locate_evidence(turn.text, raw_rows)
            for mid in set(loc["exact_ids"]) | set(loc["fuzzy_ids"]):
                if loc["statuses"].get(mid) == "active":
                    active_ids.add(mid)
        items.append(
            QAItem(
                qid=q.qid,
                conv_id=conv.conv_id,
                category=q.category,
                question=q.question,
                asked_at=q.asked_at,
                evidence_active_ids=active_ids,
                excluded=not resolved_any,
                label=label_by_qid.get(q.qid),
            )
        )
    return items


# ── Fusion (reimplemented orchestration; reuses product primitives) ──────


def fuse_rank(
    *,
    conv: ConvData,
    variant: Variant,
    query: str,
    query_vec: np.ndarray,
    doc_vecs: dict[str, np.ndarray],
    relationship_target: str,
) -> list[str]:
    """Full ranked id list for `query` under `variant`. Mirrors
    MemoryRetriever.retrieve()'s fusion logic (recency/importance/relationship
    scoring, relevance gate, weighted RRF) but takes weights/embeddings/rrf_k
    as parameters instead of reading the process-global MemoryConfig. Also
    mirrors the 2026-08-30 fix in the product's importance/relationship
    tie-breaks (see MemoryRetriever.retrieve): importance ties break
    newest-first (descending rowid), and the relationship ranked list contains
    only candidates with a positive boost (RRF already scores an id missing
    from a list as a 0 contribution) instead of ranking every gated candidate
    and leaving the untouched majority tied at 0 in oldest-first insertion
    order. Keep this in sync with retrieval.py by hand if either changes
    again — there is no shared implementation."""
    active = conv.active
    ids = [m["id"] for m in active]
    vecs = [doc_vecs[mid] for mid in ids]
    # cosine_matrix accepts numpy arrays directly (it asarray()s internally) — passing
    # arrays instead of converting to/from python lists here matters a lot at this call
    # volume (>100k calls across all variants x conversations x questions).
    sims = cosine_matrix(cast(list[float], query_vec), cast(list[list[float]], vecs))
    semantic_scores = sorted(zip(ids, sims), key=lambda x: x[1], reverse=True)
    semantic_ranked = [mid for mid, _ in semantic_scores]

    if variant.fusion == "semantic_only":
        return semantic_ranked

    fts_rows = conv.storage.fts_search(conv.conv_id, query, limit=variant.keyword_limit)
    keyword_ranked = [m["id"] for m in fts_rows]

    if variant.fusion == "rrf2":
        fused = reciprocal_rank_fusion(
            [semantic_ranked, keyword_ranked], k=variant.rrf_k, weights=[1.0, 1.0]
        )
        return [mid for mid, _ in fused]

    assert variant.fusion == "rrf"
    mem_cfg = get_config().memory
    gated = active
    if variant.relevance_gate:
        relevant = [mid for mid, sim in semantic_scores if sim > mem_cfg.relevance_min_similarity]
        eligible = set(relevant[: mem_cfg.relevance_semantic_topk]) | set(keyword_ranked)
        if eligible:
            gated = [m for m in active if m["id"] in eligible]

    recency_scores = [
        (m["id"], _recency_score(m, m.get("tier", "buffer")), m.get("rowid", 0)) for m in gated
    ]
    recency_scores.sort(key=lambda x: (-x[1], x[2]))
    recency_ranked = [mid for mid, _, _ in recency_scores]

    tier_boosts = _get_tier_boosts()
    importance_scores = []
    for m in gated:
        base = m.get("importance", 0.5) * m.get("certainty", 1.0)
        boost = tier_boosts.get(m.get("tier", "buffer"), 0.0)
        meta = m.get("metadata", {})
        if relationship_target and meta.get("user_id") == relationship_target:
            base += 0.2
        importance_scores.append((m["id"], base + boost, m.get("rowid", 0)))
    # Ties break newest-first (descending rowid) — see fuse_rank's docstring.
    importance_scores.sort(key=lambda x: (-x[1], -x[2]))
    importance_ranked = [mid for mid, _, _ in importance_scores]

    ranked_lists = [semantic_ranked, keyword_ranked, recency_ranked, importance_ranked]
    weights = [
        variant.weight_semantic,
        variant.weight_keyword,
        variant.weight_recency,
        variant.weight_importance,
    ]

    if relationship_target:
        target_lower = relationship_target.lower()
        rel_scores = []
        for m in gated:
            content_lower = m["content"].lower()
            meta = m.get("metadata", {})
            involves = target_lower in content_lower or meta.get("target_id") == relationship_target
            if involves:
                rel_scores.append((m["id"], 1.0, m.get("rowid", 0)))
        # Only boosted candidates enter the list — see fuse_rank's docstring.
        rel_scores.sort(key=lambda x: (-x[1], -x[2]))
        ranked_lists.append([mid for mid, _, _ in rel_scores])
        weights.append(variant.weight_relationship)

    fused = reciprocal_rank_fusion(ranked_lists, k=variant.rrf_k, weights=weights)
    return [mid for mid, _ in fused]


def _doc_vecs_for(
    variant: Variant,
    conv: ConvData,
    sd_cache: dict[str, np.ndarray],
    ctx_cache: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if variant.doc_variant == "stored":
        return {m["id"]: np.asarray(m["embedding"], dtype=np.float32) for m in conv.active}
    if variant.doc_variant == "search_document":
        return {m["id"]: sd_cache[m["id"]] for m in conv.active}
    assert variant.doc_variant == "contextualized"
    return {m["id"]: ctx_cache.get(m["id"], sd_cache[m["id"]]) for m in conv.active}


def _query_vec_for(variant: Variant, qid: str, plain_cache: dict, sq_cache: dict) -> np.ndarray:
    return sq_cache[qid] if variant.query_prefix else plain_cache[qid]


# ── Recall aggregation ─────────────────────────────────────────────────────


def rank_of_best_evidence(ranked_ids: list[str], evidence_ids: set[str]) -> int | None:
    if not evidence_ids:
        return None
    pos = {mid: i + 1 for i, mid in enumerate(ranked_ids)}
    ranks = [pos[mid] for mid in evidence_ids if mid in pos]
    return min(ranks) if ranks else None


def recall_at_k(ranks: list[int | None], k_list: tuple[int, ...]) -> dict[int, float]:
    n = len(ranks)
    if n == 0:
        return {k: 0.0 for k in k_list}
    return {k: sum(1 for r in ranks if r is not None and r <= k) / n for k in k_list}


# ── Validation: reimplementation must reproduce the product for a sample ──


def validate_v0(
    conv_data: dict[str, ConvData],
    qa_by_conv: dict[str, list[QAItem]],
    n: int,
    plain_cache: dict[str, np.ndarray],  # unused directly; validation re-embeds per-question
    # to exactly mirror the single embed() call the product's retrieve() makes (rather than
    # the batched embed_batch() call used to build the cache) — a local embedding server can
    # in principle score a batched vs single call for the same text a hair differently.
) -> dict:
    rng = random.Random(SEED)
    emb = embedder()
    candidates = [
        item for conv_id, items in qa_by_conv.items() for item in items if not item.excluded
    ]
    rng.shuffle(candidates)
    sample = candidates[:n]

    exact_matches = 0
    set_matches = 0
    mismatches: list[dict] = []

    # Group by conv so we build one Engine per conv, not per question.
    by_conv: dict[str, list[QAItem]] = defaultdict(list)
    for item in sample:
        by_conv[item.conv_id].append(item)

    v0 = VARIANTS[0]
    assert v0.name == "V0_baseline"

    for conv_id, items in by_conv.items():
        conv = conv_data[conv_id]
        from woven_imprint.engine import Engine

        engine = Engine(
            db_path=conv.storage.db_path, llm=cast(LLMProvider, FakeLLM()), embedding=embedder()
        )
        char = engine.get_character(conv_id)
        char.background = False
        char.parallel = False
        char.enforce_consistency = False
        user_name = conv.dataset_conv.user_name

        for item in items:
            clock.override(item.asked_at)
            real_top20 = [
                m["id"]
                for m in char.retriever.retrieve(
                    item.question, limit=20, relationship_target=user_name
                )
            ]

            query_vec = np.asarray(emb.embed(item.question), dtype=np.float32)
            doc_vecs = {m["id"]: np.asarray(m["embedding"], dtype=np.float32) for m in conv.active}
            reimpl = fuse_rank(
                conv=conv,
                variant=v0,
                query=item.question,
                query_vec=query_vec,
                doc_vecs=doc_vecs,
                relationship_target=user_name,
            )[:20]

            if reimpl == real_top20:
                exact_matches += 1
            elif set(reimpl) == set(real_top20):
                set_matches += 1
                mismatches.append(
                    {"qid": item.qid, "kind": "order_only", "real": real_top20, "reimpl": reimpl}
                )
            else:
                mismatches.append(
                    {"qid": item.qid, "kind": "set_mismatch", "real": real_top20, "reimpl": reimpl}
                )
        clock.override(None)

    return {
        "n": len(sample),
        "exact_order_matches": exact_matches,
        "same_set_diff_order": set_matches,
        "set_mismatches": len(sample) - exact_matches - set_matches,
        "mismatch_examples": mismatches[:5],
    }


# ── Main ────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    global RUN_ID, RUN_DIR, RESULTS_FILE
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument(
        "--limit-convs", type=int, default=None, help="debug: only process N conversations"
    )
    parser.add_argument(
        "--followup",
        action="store_true",
        help="run the coordinator follow-up sweep (a/b/c/d) instead of the main 12-variant pass; "
        "reuses cached embeddings, appends to the existing report instead of overwriting it",
    )
    parser.add_argument(
        "--run-id",
        default=RUN_ID,
        help=f"bench=locomo mode=memory run id whose ingested DBs/results to diagnose "
        f"(default: {RUN_ID!r})",
    )
    args = parser.parse_args(argv)

    RUN_ID = args.run_id
    RUN_DIR = RUNS_DIR / RUN_ID
    RESULTS_FILE = RESULTS_DIR / f"external_{RUN_ID}.json"

    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    conv_ids = sorted(p.stem for p in RUN_DIR.glob("conv-*.db"))
    if args.limit_convs:
        conv_ids = conv_ids[: args.limit_convs]
    dataset_convs = {c.conv_id: c for c in load_locomo(LOCOMO_PATH)}

    label_by_qid: dict[str, str] = {}
    if RESULTS_FILE.exists():
        results = json.loads(RESULTS_FILE.read_text())
        for rec in results["conversations"]:
            if rec.get("kind") == "qa":
                label_by_qid[rec["qid"]] = rec.get("label", "")

    print(f"[{time.time() - t_start:.1f}s] preparing {len(conv_ids)} conversations...")
    conv_data: dict[str, ConvData] = {}
    qa_by_conv: dict[str, list[QAItem]] = {}
    for conv_id in conv_ids:
        conv, raw_rows = _prepare_conv(conv_id, dataset_convs[conv_id])
        conv_data[conv_id] = conv
        qa_by_conv[conv_id] = _build_qa_items(conv, raw_rows, label_by_qid)

    all_items = [item for items in qa_by_conv.values() for item in items]
    n_excluded = sum(1 for i in all_items if i.excluded)
    print(
        f"[{time.time() - t_start:.1f}s] {len(all_items)} qa items "
        f"({n_excluded} no_evidence_resolved, excluded from recall denominator)"
    )

    # ── Embedding caches ──────────────────────────────────────────────
    emb = embedder()

    plain_cache = _load_cache(CACHE_DIR / "query_plain.pkl")
    sq_cache = _load_cache(CACHE_DIR / "query_search_query.pkl")
    n_new = _embed_missing(plain_cache, [(i.qid, i.question) for i in all_items], emb.embed_batch)
    print(
        f"[{time.time() - t_start:.1f}s] query plain embeddings: +{n_new} new, {len(plain_cache)} total"
    )
    n_new = _embed_missing(
        sq_cache,
        [(i.qid, SEARCH_QUERY_PREFIX + i.question) for i in all_items],
        emb.embed_batch,
    )
    print(
        f"[{time.time() - t_start:.1f}s] query search_query embeddings: +{n_new} new, {len(sq_cache)} total"
    )
    _save_cache(CACHE_DIR / "query_plain.pkl", plain_cache)
    _save_cache(CACHE_DIR / "query_search_query.pkl", sq_cache)

    sd_caches: dict[str, dict[str, np.ndarray]] = {}
    ctx_caches: dict[str, dict[str, np.ndarray]] = {}
    for conv_id in conv_ids:
        conv = conv_data[conv_id]
        sd_path = CACHE_DIR / f"doc_search_document_{conv_id}.pkl"
        sd_cache = _load_cache(sd_path)
        n_new = _embed_missing(
            sd_cache,
            [(m["id"], SEARCH_DOC_PREFIX + m["content"]) for m in conv.active],
            emb.embed_batch,
        )
        _save_cache(sd_path, sd_cache)
        sd_caches[conv_id] = sd_cache

        ctx_path = CACHE_DIR / f"doc_contextualized_{conv_id}.pkl"
        ctx_cache = _load_cache(ctx_path)
        buffer_mems = [m for m in conv.active if m.get("tier") == "buffer"]
        user_name = conv.dataset_conv.user_name
        character_name = conv.dataset_conv.character_name

        def _speaker(role: str) -> str:
            if role == "user":
                return user_name
            if role == "character":
                return character_name
            return role or "unknown"

        n_new_ctx = _embed_missing(
            ctx_cache,
            [
                (
                    m["id"],
                    f"{SEARCH_DOC_PREFIX}[{(m.get('created_at') or '')[:10]}] "
                    f"{_speaker(m.get('role') or '')}: {m['content']}",
                )
                for m in buffer_mems
            ],
            emb.embed_batch,
        )
        _save_cache(ctx_path, ctx_cache)
        ctx_caches[conv_id] = ctx_cache
        print(
            f"[{time.time() - t_start:.1f}s] {conv_id}: search_document +{n_new} new "
            f"({len(sd_cache)} total), contextualized +{n_new_ctx} new ({len(ctx_cache)} total, buffer-only)"
        )

    if args.followup:
        dated_cache_path = CACHE_DIR / "query_search_query_dated.pkl"
        query_dated_cache = _load_cache(dated_cache_path)
        n_new = _embed_missing(
            query_dated_cache,
            [
                (i.qid, f"{SEARCH_QUERY_PREFIX}[{i.asked_at.date().isoformat()}] {i.question}")
                for i in all_items
            ],
            emb.embed_batch,
        )
        _save_cache(dated_cache_path, query_dated_cache)
        print(
            f"[{time.time() - t_start:.1f}s] dated query embeddings: +{n_new} new, "
            f"{len(query_dated_cache)} total"
        )
        followup_out = run_followup(
            conv_data=conv_data,
            qa_by_conv=qa_by_conv,
            plain_cache=plain_cache,
            sq_cache=sq_cache,
            sq_dated_cache=query_dated_cache,
            sd_caches=sd_caches,
            ctx_caches=ctx_caches,
        )
        (OUT_DIR / "ranking_followup_results.json").write_text(
            json.dumps(followup_out, indent=2, default=str)
        )
        _append_followup_report(followup_out)
        print(f"[{time.time() - t_start:.1f}s] wrote {OUT_DIR / 'ranking_followup_results.json'}")
        print(
            f"[{time.time() - t_start:.1f}s] appended follow-up section to ranking_experiments_report.md"
        )
        return

    # ── Validation (V0 reimplementation vs real product retrieve()) ──
    validation: dict = {}
    if not args.skip_validation:
        print(
            f"[{time.time() - t_start:.1f}s] validating V0 reimplementation on {VALIDATION_N} sampled questions..."
        )
        validation = validate_v0(conv_data, qa_by_conv, VALIDATION_N, plain_cache)
        print(
            f"[{time.time() - t_start:.1f}s] validation: {validation['exact_order_matches']}/{validation['n']} exact order match"
        )

    # ── Run all variants ──────────────────────────────────────────────
    per_variant_ranks: dict[str, list[tuple[QAItem, int | None]]] = {}
    relationship_target_cache: dict[str, str] = {
        conv_id: conv.dataset_conv.user_name for conv_id, conv in conv_data.items()
    }

    for variant in VARIANTS:
        t_v = time.time()
        ranks: list[tuple[QAItem, int | None]] = []
        for conv_id in conv_ids:
            conv = conv_data[conv_id]
            doc_vecs = _doc_vecs_for(variant, conv, sd_caches[conv_id], ctx_caches[conv_id])
            rel_target = relationship_target_cache[conv_id]
            items = qa_by_conv[conv_id]
            for item in items:
                if item.excluded:
                    continue
                clock.override(item.asked_at)
                query_vec = _query_vec_for(variant, item.qid, plain_cache, sq_cache)
                ranked = fuse_rank(
                    conv=conv,
                    variant=variant,
                    query=item.question,
                    query_vec=query_vec,
                    doc_vecs=doc_vecs,
                    relationship_target=rel_target,
                )
                r = rank_of_best_evidence(ranked, item.evidence_active_ids)
                ranks.append((item, r))
        clock.override(None)
        per_variant_ranks[variant.name] = ranks
        recalls = recall_at_k([r for _, r in ranks], K_LIST)
        print(
            f"[{time.time() - t_start:.1f}s] {variant.name}: "
            + ", ".join(f"@{k}={recalls[k] * 100:.1f}%" for k in K_LIST)
            + f"  ({time.time() - t_v:.1f}s)"
        )

    # ── Aggregate results ─────────────────────────────────────────────
    variant_recalls = {
        name: recall_at_k([r for _, r in ranks], K_LIST)
        for name, ranks in per_variant_ranks.items()
    }
    best_name = max(variant_recalls, key=lambda n: variant_recalls[n][20])
    best_ranks = per_variant_ranks[best_name]

    per_category_best: dict[str, float] = {}
    for cat in sorted({item.category for item in all_items}):
        cat_ranks = [r for item, r in best_ranks if item.category == cat]
        per_category_best[cat] = recall_at_k(cat_ranks, (20,))[20]

    retrieved_but_wrong_note = (
        "From the pre-existing recall_diagnostic.md (locomo-mem-v2, response-level): 113 "
        "questions (64 abstained + 49 answered_wrong) already had evidence in the product's "
        "top-20 (evidence_class='retrieved') yet were still graded WRONG. Ranking changes "
        "cannot move these — the failure is downstream (generation/abstention), not retrieval — "
        "so this count is unaffected by any variant above and is reported here only as the "
        "ceiling on how much recall improvements alone can move the WRONG total."
    )

    variant_dump = [
        {
            "name": v.name,
            "fusion": v.fusion,
            "keyword_limit": v.keyword_limit,
            "weights": {
                "semantic": v.weight_semantic,
                "keyword": v.weight_keyword,
                "recency": v.weight_recency,
                "importance": v.weight_importance,
                "relationship": v.weight_relationship,
            },
            "rrf_k": v.rrf_k,
            "query_prefix": v.query_prefix,
            "doc_variant": v.doc_variant,
            "requires_reingest": v.requires_reingest,
            "recall_at_k": variant_recalls[v.name],
        }
        for v in VARIANTS
    ]

    out = {
        "run_id": RUN_ID,
        "n_qa_total": len(all_items),
        "n_excluded_no_evidence_resolved": n_excluded,
        "n_scored": len(all_items) - n_excluded,
        "validation": validation,
        "variants": variant_dump,
        "best_variant": best_name,
        "best_variant_per_category_recall_at_20": per_category_best,
        "retrieved_but_wrong_note": retrieved_but_wrong_note,
        "elapsed_seconds": time.time() - t_start,
    }
    (OUT_DIR / "ranking_experiments_results.json").write_text(
        json.dumps(out, indent=2, default=str)
    )

    _write_report(out)
    print(f"[{time.time() - t_start:.1f}s] wrote {OUT_DIR / 'ranking_experiments_results.json'}")
    print(f"[{time.time() - t_start:.1f}s] wrote {OUT_DIR / 'ranking_experiments_report.md'}")


def _write_report(out: dict) -> None:
    lines = ["# LoCoMo ranking experiments (offline, no LLM calls)", ""]
    lines.append(
        f"run_id: {out['run_id']}, qa questions scored: {out['n_scored']}/{out['n_qa_total']}"
    )
    lines.append("")
    v = out["validation"]
    if v:
        lines.append(
            f"## V0 reimplementation validation ({v['n']} sampled questions)\n\n"
            f"exact order match: {v['exact_order_matches']}/{v['n']}, "
            f"same set diff order: {v['same_set_diff_order']}, "
            f"set mismatches: {v['set_mismatches']}\n"
        )
    lines.append("## Variant x recall@K\n")
    lines.append(
        "| variant | fusion | rrf_k | weights (s/k/r/i/rel) | query prefix | doc variant | @10 | @20 | @50 |"
    )
    lines.append("|---|---|---:|---|---|---|---:|---:|---:|")
    for vd in out["variants"]:
        w = vd["weights"]
        wstr = (
            f"{w['semantic']}/{w['keyword']}/{w['recency']}/{w['importance']}/{w['relationship']}"
        )
        r = vd["recall_at_k"]
        lines.append(
            f"| {vd['name']} | {vd['fusion']} | {vd['rrf_k']} | {wstr} | "
            f"{vd['query_prefix'] or '-'} | {vd['doc_variant']} | "
            f"{r[10] * 100:.1f}% | {r[20] * 100:.1f}% | {r[50] * 100:.1f}% |"
        )
    lines.append("")
    lines.append(f"**Best variant (by recall@20): {out['best_variant']}**\n")
    lines.append("## Best variant — per-category recall@20\n")
    lines.append("| category | recall@20 |")
    lines.append("|---|---:|")
    for cat, val in sorted(out["best_variant_per_category_recall_at_20"].items()):
        lines.append(f"| {cat} | {val * 100:.1f}% |")
    lines.append("")
    lines.append("## 'Retrieved but wrong' note\n")
    lines.append(out["retrieved_but_wrong_note"])
    lines.append("")
    (OUT_DIR / "ranking_experiments_report.md").write_text("\n".join(lines))


# ── Follow-up sweep (coordinator request 2026-08-30) ──────────────────────
#
# (a) V0-style fusion (raw/stored content embeddings, no prefixes) swept over
#     weight_recency == weight_importance in {0, 0.1, 0.25, 0.5} x
#     weight_relationship in {0, 1} -- 8 cells. Purpose: find a product
#     default that keeps a *little* recency/importance signal if it's free.
# (b) V6b-style fusion (contextualized buffer docs, "search_query: " prefixed
#     queries) swept over the same recency/importance grid (0, 0.1, 0.25) x
#     rrf_k in {20, 60, 120} -- 9 cells; then weight_keyword in
#     {0.5, 1.0, 2.0} at the best (recency/importance, rrf_k) cell.
# (d) V6b but the query is ALSO contextualized with the question's asked_at
#     date ("search_query: [YYYY-MM-DD] question") -- does a date anchor on
#     the query side help category 2 (temporal)?
#
# No new embedding calls for (a)/(b): both reuse the exact caches the main
# run already built (the stored embedding blob for (a), doc_search_document_*
# + doc_contextualized_* for (b)). (d) needs one new query-embedding variant
# (date-prefixed text), cached to query_search_query_dated.pkl so a rerun
# costs nothing either.


def _eval_variant(
    variant,
    conv_data,
    qa_by_conv,
    query_cache,
    doc_vecs_by_conv,
):
    """Rank of the best evidence memory for every non-excluded qa item, under `variant`,
    using caller-supplied query/doc embeddings (bypasses _query_vec_for/_doc_vecs_for's
    variant.query_prefix/doc_variant dispatch so sweep cells don't need placeholder Variant
    fields)."""
    ranks = []
    for conv_id, conv in conv_data.items():
        doc_vecs = doc_vecs_by_conv[conv_id]
        rel_target = conv.dataset_conv.user_name
        for item in qa_by_conv[conv_id]:
            if item.excluded:
                continue
            clock.override(item.asked_at)
            qv = query_cache[item.qid]
            ranked = fuse_rank(
                conv=conv,
                variant=variant,
                query=item.question,
                query_vec=qv,
                doc_vecs=doc_vecs,
                relationship_target=rel_target,
            )
            ranks.append(rank_of_best_evidence(ranked, item.evidence_active_ids))
    clock.override(None)
    return ranks


def run_followup(
    *,
    conv_data,
    qa_by_conv,
    plain_cache,
    sq_cache,
    sq_dated_cache,
    sd_caches,
    ctx_caches,
):
    doc_vecs_stored = {
        conv_id: {m["id"]: np.asarray(m["embedding"], dtype=np.float32) for m in conv.active}
        for conv_id, conv in conv_data.items()
    }
    doc_vecs_ctx = {
        conv_id: {
            m["id"]: ctx_caches[conv_id].get(m["id"], sd_caches[conv_id][m["id"]])
            for m in conv.active
        }
        for conv_id, conv in conv_data.items()
    }

    # ── (a) V0 fusion, recency=importance grid x relationship grid ──
    a_rows = []
    for wr in (0.0, 0.1, 0.25, 0.5):
        for wrel in (0.0, 1.0):
            v = Variant(
                "a_wr{}_wrel{}".format(wr, wrel),
                fusion="rrf",
                weight_semantic=1.0,
                weight_keyword=1.0,
                weight_recency=wr,
                weight_importance=wr,
                weight_relationship=wrel,
                rrf_k=60,
            )
            ranks = _eval_variant(v, conv_data, qa_by_conv, plain_cache, doc_vecs_stored)
            row = {
                "weight_recency_importance": wr,
                "weight_relationship": wrel,
                "recall": recall_at_k(ranks, K_LIST),
            }
            a_rows.append(row)
            print(
                "  (a) wr=wi={} wrel={}: {}".format(
                    wr,
                    wrel,
                    ", ".join("@{}={:.1f}%".format(k, row["recall"][k] * 100) for k in K_LIST),
                )
            )

    # ── (b) V6b fusion, recency=importance x rrf_k grid ──
    b_rows = []
    for wr in (0.0, 0.1, 0.25):
        for k in (20, 60, 120):
            v = Variant(
                "b_wr{}_k{}".format(wr, k),
                fusion="rrf",
                weight_semantic=1.0,
                weight_keyword=1.0,
                weight_recency=wr,
                weight_importance=wr,
                weight_relationship=1.0,
                rrf_k=k,
            )
            ranks = _eval_variant(v, conv_data, qa_by_conv, sq_cache, doc_vecs_ctx)
            row = {
                "weight_recency_importance": wr,
                "rrf_k": k,
                "recall": recall_at_k(ranks, K_LIST),
            }
            b_rows.append(row)
            print(
                "  (b) wr=wi={} rrf_k={}: {}".format(
                    wr,
                    k,
                    ", ".join("@{}={:.1f}%".format(kk, row["recall"][kk] * 100) for kk in K_LIST),
                )
            )

    best_b = max(b_rows, key=lambda row: row["recall"][20])

    # ── (b) keyword-weight sweep at the best (recency/importance, rrf_k) cell ──
    b_kw_rows = []
    for wk in (0.5, 1.0, 2.0):
        v = Variant(
            "b_kw{}".format(wk),
            fusion="rrf",
            weight_semantic=1.0,
            weight_keyword=wk,
            weight_recency=best_b["weight_recency_importance"],
            weight_importance=best_b["weight_recency_importance"],
            weight_relationship=1.0,
            rrf_k=best_b["rrf_k"],
        )
        ranks = _eval_variant(v, conv_data, qa_by_conv, sq_cache, doc_vecs_ctx)
        row = {"weight_keyword": wk, "recall": recall_at_k(ranks, K_LIST)}
        b_kw_rows.append(row)
        print(
            "  (b) keyword sweep @best cell wk={}: {}".format(
                wk, ", ".join("@{}={:.1f}%".format(kk, row["recall"][kk] * 100) for kk in K_LIST)
            )
        )

    # ── (d) V6b (best-b weights) but the query is ALSO date-contextualized ──
    v_d = Variant(
        "d_dated_query",
        fusion="rrf",
        weight_semantic=1.0,
        weight_keyword=1.0,
        weight_recency=best_b["weight_recency_importance"],
        weight_importance=best_b["weight_recency_importance"],
        weight_relationship=1.0,
        rrf_k=best_b["rrf_k"],
    )
    items_by_id = {i.qid: i for items in qa_by_conv.values() for i in items}
    ranks_d = []
    id_order = []
    for conv_id, conv in conv_data.items():
        rel_target = conv.dataset_conv.user_name
        for item in qa_by_conv[conv_id]:
            if item.excluded:
                continue
            clock.override(item.asked_at)
            qv = sq_dated_cache[item.qid]
            ranked = fuse_rank(
                conv=conv,
                variant=v_d,
                query=item.question,
                query_vec=qv,
                doc_vecs=doc_vecs_ctx[conv_id],
                relationship_target=rel_target,
            )
            ranks_d.append(rank_of_best_evidence(ranked, item.evidence_active_ids))
            id_order.append(item.qid)
    clock.override(None)
    d_overall = recall_at_k(ranks_d, K_LIST)
    d_per_cat = {}
    for cat in sorted(set(items_by_id[qid].category for qid in id_order)):
        cat_ranks = [r for qid, r in zip(id_order, ranks_d) if items_by_id[qid].category == cat]
        d_per_cat[cat] = recall_at_k(cat_ranks, (20,))[20]
    print(
        "  (d) dated query: "
        + ", ".join("@{}={:.1f}%".format(k, d_overall[k] * 100) for k in K_LIST)
    )
    print("  (d) dated query per-category @20: {}".format(d_per_cat))

    # Same v_d weights/rrf_k but WITHOUT the date-in-query change, as the direct baseline
    # for isolating (d)'s effect (equivalent to "b" at its own best cell).
    ranks_b_best = _eval_variant(v_d, conv_data, qa_by_conv, sq_cache, doc_vecs_ctx)
    b_best_overall = recall_at_k(ranks_b_best, K_LIST)
    b_best_per_cat = {}
    for cat in sorted(set(items_by_id[qid].category for qid in id_order)):
        cat_ranks = [
            r for qid, r in zip(id_order, ranks_b_best) if items_by_id[qid].category == cat
        ]
        b_best_per_cat[cat] = recall_at_k(cat_ranks, (20,))[20]

    doc_template_lines = [
        'Buffer-tier memories only: `f"search_document: [{date}] {speaker}: {content}"`',
        "where date = `(m['created_at'] or '')[:10]` (SQLite TEXT timestamp's YYYY-MM-DD",
        "prefix, i.e. the memory's created_at, not the conversation/session date) and speaker",
        "is derived purely from the memories.role column (NOT metadata.user_id):",
        "role=='user' -> conv.user_name (LoCoMo speaker_a), role=='character' ->",
        "conv.character_name (speaker_b) -- buffer rows only ever carry role in",
        "{'user','character'} in this dataset. `content` is the stored memory content",
        "UNCHANGED, which for buffer rows already carries an ingestion-time speaker bracket",
        "tag (e.g. '[User] Hey Mel!...' or '[Melanie] Hey Caroline!...'), so the speaker name",
        "appears TWICE in the final embedded string -- once from the injected 'speaker:'",
        "prefix and once from that pre-existing bracket tag. This redundancy wasn't",
        "deliberate and is worth resolving (e.g. strip the bracket tag before injecting the",
        "context prefix) when implementing in production.",
        "",
        "Core 'extraction' fact rows (e.g. \"Caroline attended an LGBTQ support group",
        "yesterday.\", metadata.source='extraction'), core '[Consolidated] <summary>' rows",
        "(metadata.type='consolidation', role='observation', content literally prefixed",
        "'[Consolidated] ' by ConsolidationEngine._consolidate_cluster) and the 2 bedrock",
        "persona/self-description rows were NOT given date/speaker context in this",
        "experiment -- the buffer-only filter (tier == 'buffer') skips all of them, so they",
        'fall back to the plain V5-style `f"search_document: {content}"` embedding (no',
        "context at all). If this ships, core/bedrock rows need their own template decision",
        "(they have no natural single 'speaker'; role is always 'observation') rather than",
        "silently inheriting the no-context fallback.",
    ]
    doc_template_note = "\n".join(doc_template_lines)

    return {
        "a": a_rows,
        "b_grid": b_rows,
        "b_best_cell": {
            "weight_recency_importance": best_b["weight_recency_importance"],
            "rrf_k": best_b["rrf_k"],
        },
        "b_keyword_sweep": b_kw_rows,
        "b_best_no_date_query": {"overall": b_best_overall, "per_category_at_20": b_best_per_cat},
        "d_dated_query": {"overall": d_overall, "per_category_at_20": d_per_cat},
        "doc_template_note": doc_template_note,
    }


def _append_followup_report(out):
    path = OUT_DIR / "ranking_experiments_report.md"
    existing = path.read_text() if path.exists() else ""

    lines = ["", "---", "", "# Follow-up sweep (coordinator request, 2026-08-30)", ""]
    lines.append(
        "No new embedding calls for (a)/(b) -- both reuse the main run's cached stored / "
        "search_document / contextualized embeddings. (d) adds one new query-embedding "
        "variant (date-prefixed queries), cached to `query_search_query_dated.pkl`."
    )
    lines.append("")

    lines.append(
        "## (a) V0 fusion (raw/stored content embeddings) -- recency=importance x relationship weight"
    )
    lines.append("")
    lines.append("| w_recency=w_importance | w_relationship | @10 | @20 | @50 |")
    lines.append("|---:|---:|---:|---:|---:|")
    for row in out["a"]:
        r = row["recall"]
        lines.append(
            "| {} | {} | {:.1f}% | {:.1f}% | {:.1f}% |".format(
                row["weight_recency_importance"],
                row["weight_relationship"],
                r[10] * 100,
                r[20] * 100,
                r[50] * 100,
            )
        )
    lines.append("")

    lines.append("## (b) V6b fusion (contextualized docs) -- recency=importance x rrf_k")
    lines.append("")
    lines.append("| w_recency=w_importance | rrf_k | @10 | @20 | @50 |")
    lines.append("|---:|---:|---:|---:|---:|")
    for row in out["b_grid"]:
        r = row["recall"]
        lines.append(
            "| {} | {} | {:.1f}% | {:.1f}% | {:.1f}% |".format(
                row["weight_recency_importance"],
                row["rrf_k"],
                r[10] * 100,
                r[20] * 100,
                r[50] * 100,
            )
        )
    lines.append("")
    bc = out["b_best_cell"]
    lines.append(
        "Best (b) cell by recall@20: weight_recency=weight_importance={}, rrf_k={}.".format(
            bc["weight_recency_importance"], bc["rrf_k"]
        )
    )
    lines.append("")
    lines.append("### (b) keyword-weight sweep at the best cell")
    lines.append("")
    lines.append("| weight_keyword | @10 | @20 | @50 |")
    lines.append("|---:|---:|---:|---:|")
    for row in out["b_keyword_sweep"]:
        r = row["recall"]
        lines.append(
            "| {} | {:.1f}% | {:.1f}% | {:.1f}% |".format(
                row["weight_keyword"],
                r[10] * 100,
                r[20] * 100,
                r[50] * 100,
            )
        )
    lines.append("")

    lines.append("## (c) V6 document embedding template (for ingestion-path parity)")
    lines.append("")
    lines.append(out["doc_template_note"])
    lines.append("")

    lines.append("## (d) V6b with the query ALSO date-contextualized")
    lines.append("")
    lines.append("| | @10 | @20 | @50 | cat1@20 | cat2@20 | cat3@20 | cat4@20 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    base = out["b_best_no_date_query"]
    dd = out["d_dated_query"]
    bo, do = base["overall"], dd["overall"]
    bp, dp = base["per_category_at_20"], dd["per_category_at_20"]
    lines.append(
        "| b best cell, plain query | {:.1f}% | {:.1f}% | {:.1f}% | {:.1f}% | {:.1f}% | {:.1f}% | {:.1f}% |".format(
            bo[10] * 100,
            bo[20] * 100,
            bo[50] * 100,
            bp.get("1", 0) * 100,
            bp.get("2", 0) * 100,
            bp.get("3", 0) * 100,
            bp.get("4", 0) * 100,
        )
    )
    lines.append(
        "| + dated query ('[date] question') | {:.1f}% | {:.1f}% | {:.1f}% | {:.1f}% | {:.1f}% | {:.1f}% | {:.1f}% |".format(
            do[10] * 100,
            do[20] * 100,
            do[50] * 100,
            dp.get("1", 0) * 100,
            dp.get("2", 0) * 100,
            dp.get("3", 0) * 100,
            dp.get("4", 0) * 100,
        )
    )
    lines.append("")
    lines.append(
        "Note: LoCoMo's `asked_at` is computed once per conversation (last session's time + 1 "
        "day), not per question -- every question in a conversation shares the same date, so "
        "this variant tests whether a constant per-conversation date anchor on the query "
        "shifts the embedding geometry, not whether per-question dates help."
    )
    lines.append("")

    path.write_text(existing + "\n".join(lines))


if __name__ == "__main__":
    main()
