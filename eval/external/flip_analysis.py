"""Flip analysis: why locomo-mem-v2b scores lower than locomo-mem-v2 despite the new
retrieval defaults roughly doubling offline evidence recall@20 (~23.9% -> ~50%).

v2 and v2b share identical ingested per-conversation DBs (v2b is a *re-answer* run: same
ingestion, only the answering-phase ranking changed). This script re-derives, from those DBs,
BOTH the OLD (v2) and NEW (v2b) ranked memory lists for every question both runs have judged so
far, and cross-tabulates that against which answers flipped correct<->wrong, to find out whether
the recall win is actually reaching the prompt in a form the model can use.

Ranking reconstruction
-----------------------
"NEW" == current ``src/woven_imprint/memory/retrieval.py`` (commit 5c84a76, the live v2b
config: ``weight_recency=0.1``, ``weight_importance=0.0``, relationship strategy ranks only
*boosted* candidates, importance ties break newest-first/descending-rowid). This is exercised
via the real ``MemoryRetriever`` class (not reimplemented) and validated per :func:`validate_new`
by comparing to ``MemoryRetriever.retrieve()`` directly, the same technique
``ranking_experiments.validate_v0`` uses.

"OLD" == the pre-5c84a76 code (``weight_recency=1.0``, ``weight_importance=1.0``, importance ties
break OLDEST-first/ascending-rowid, relationship strategy ranks *every* gated candidate with the
untouched majority tied at 0.0 and falling back to Python's stable sort over `gated`'s
ascending-rowid input order -- i.e. oldest-first). This is NOT the code path present in the repo
any more (5c84a76 replaced it in place), so it cannot be validated against a live
``MemoryRetriever.retrieve()`` call the way NEW can -- instead :func:`fuse_rank` mode="old" is a
hand-transcription of ``git show 5c84a76 -- src/woven_imprint/memory/retrieval.py``'s pre-image
(verified line-by-line against that diff when this script was written), plus
:func:`_selftest_old_tiebreaks`, a DB-free synthetic check that OLD's relationship/importance
tie-break behavior actually IS oldest-first/ascending-rowid as claimed (run automatically, see
``main``).

Both modes share every other primitive verbatim from the product (``cosine_matrix``,
``_recency_score``, ``_get_tier_boosts``, ``reciprocal_rank_fusion``) and the live relevance-gate
config (``relevance_gate``/``relevance_min_similarity``/``relevance_semantic_topk`` did not change
between v2 and v2b -- only the weights and the two tie-breaks did, see the commit diff).

No LLM calls (a ``FakeLLM`` stands in for ``Engine``'s required LLM provider -- retrieval and
facts-block rendering never call it). The embedding server at ``127.0.0.1:11801`` *is* called
(query embeddings only -- document embeddings are already stored on each memory row).

DBs are copied from ``eval/external/runs/<new-run-id>/conv-*.db`` (v2b's, per the task brief --
identical content to v2's, and answering never mutates them) into
``eval/external/runs/diagnostics/flip/`` and only ever opened there; the original run
directories are never opened for writing and (v2b in particular) are still being appended to by
a live run -- this script only ever reads whichever ``conv-*.answers.json`` records exist at
the moment it runs and reports how many it saw.

Run (repo root, product venv): ``./venv/bin/python -m eval.external.flip_analysis``

Outputs under ``eval/external/runs/diagnostics/flip/`` (gitignored):
    flip_matrix.json / .md           -- item 1
    flip_sample.json / flip_sample.md -- item 2 (40-item table + 8 worked examples)
    top20_aggregate.json / .md        -- item 3
    consolidated_ousted.json / .md    -- item 4
    abstention_despite_evidence.json  -- item 5
    all_items.json                    -- the full per-question computed record set (for reruns/
                                          further slicing without recomputation)
    validation.json                   -- NEW-mode validation + OLD tie-break self-test
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import cast

from woven_imprint import clock
from woven_imprint.config import get_config
from woven_imprint.engine import Engine
from woven_imprint.llm.base import LLMProvider
from woven_imprint.memory.retrieval import _get_tier_boosts, _recency_score, cosine_matrix
from woven_imprint.storage.sqlite import SQLiteStorage
from woven_imprint.utils.rrf import reciprocal_rank_fusion

from tests.helpers import FakeLLM

from . import metrics
from .common import DATA_DIR, RESULTS_DIR, RUNS_DIR, embedder
from .diagnose_recall import _copy_db, _load_all_memory_rows, _locate_evidence
from .locomo import load_locomo

OLD_RUN_ID_DEFAULT = "locomo-mem-v2"
NEW_RUN_ID_DEFAULT = "locomo-mem-v2b"
DIAG_DIR = RUNS_DIR / "diagnostics" / "flip"
LOCOMO_PATH = DATA_DIR / "locomo.json"

SAMPLE_N = 40
WORKED_N = 8
SAMPLE_SEED = 20260830
VALIDATE_N = 30
TOP_K = 20


# ── Row classification ────────────────────────────────────────────────────


def row_kind(m: dict) -> str:
    """Coarse tier/kind bucket for a memory row: buffer_raw_turn / core_fact /
    consolidated_summary / bedrock / other. See ConsolidationEngine._consolidate_cluster
    (metadata.type == "consolidation", content prefixed "[Consolidated] ") and
    character.py's extraction path (metadata.source == "extraction") for core rows.
    """
    tier = m.get("tier")
    meta = m.get("metadata") or {}
    content = m.get("content") or ""
    if tier == "bedrock":
        return "bedrock"
    if tier == "buffer":
        return "buffer_raw_turn"
    if tier == "core":
        if meta.get("type") == "consolidation" or content.startswith("[Consolidated]"):
            return "consolidated_summary"
        return "core_fact"
    return "other"


# ── Memoizing embedder (query text is embedded once, reused across old/new/validate) ──


class MemoEmbedder:
    def __init__(self, inner):
        self._inner = inner
        self._cache: dict[str, list[float]] = {}
        self.model = getattr(inner, "model", None)

    def embed(self, text: str) -> list[float]:
        if text not in self._cache:
            self._cache[text] = self._inner.embed(text)
        return self._cache[text]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_batch(texts)

    def dimensions(self) -> int:
        return self._inner.dimensions()


# ── Fusion: both OLD and NEW modes in one function (only weights + 2 tie-breaks differ) ──


def build_candidates(
    storage: SQLiteStorage, character_id: str, base_candidates: list[dict], query: str
) -> tuple[list[dict], list[str]]:
    """Mirrors MemoryRetriever.retrieve()'s candidate assembly: base (status='active', up to
    max_candidates, newest first) merged with an FTS supplement, deduped, sorted ascending-rowid.
    """
    try:
        fts_candidates = storage.fts_search(character_id, query, limit=50)
    except Exception:
        fts_candidates = []
    memory_map: dict[str, dict] = {}
    for m in base_candidates + fts_candidates:
        memory_map.setdefault(m["id"], m)
    all_memories = sorted(memory_map.values(), key=lambda m: m.get("rowid", 0))
    fts_ids = [m["id"] for m in fts_candidates]
    return all_memories, fts_ids


def fuse_rank(
    *,
    all_memories: list[dict],
    fts_ids: list[str],
    query: str,
    query_vec: list[float],
    relationship_target: str,
    mode: str,
    mem_cfg,
) -> tuple[list[str], set[str]]:
    """Full ranked id list for `query` under `mode` ("old" | "new"). Returns (ranked_ids, gated_ids).

    mode="new": current retrieval.py verbatim (weight_recency/importance from mem_cfg, i.e. the
    live 0.1/0.0 defaults; importance ties newest-first; relationship list = boosted candidates
    only). Validated against real MemoryRetriever.retrieve() by validate_new().

    mode="old": pre-5c84a76 code (weight_recency=weight_importance=1.0 forced; importance ties
    oldest-first; relationship list = every gated candidate, untouched majority tied at 0.0,
    Python's stable sort then preserves `gated`'s ascending-rowid order for that tied group).
    Hand-transcribed from `git show 5c84a76 -- src/woven_imprint/memory/retrieval.py`'s pre-image.
    """
    embedded = [m for m in all_memories if m.get("embedding")]
    sims = cosine_matrix(query_vec, [m["embedding"] for m in embedded])
    semantic_scores = list(zip((m["id"] for m in embedded), sims))
    semantic_scores.sort(key=lambda x: x[1], reverse=True)
    semantic_ranked = [mid for mid, _ in semantic_scores]

    keyword_ranked = fts_ids

    gated = all_memories
    if mem_cfg.relevance_gate and query.strip():
        relevant = [mid for mid, sim in semantic_scores if sim > mem_cfg.relevance_min_similarity]
        eligible = set(relevant[: mem_cfg.relevance_semantic_topk]) | set(keyword_ranked)
        if eligible:
            gated = [m for m in all_memories if m["id"] in eligible]

    recency_scores = [
        (m["id"], _recency_score(m, m.get("tier", "buffer")), m.get("rowid", 0)) for m in gated
    ]
    recency_scores.sort(key=lambda x: (-x[1], x[2]))
    recency_ranked = [mid for mid, _, _ in recency_scores]

    importance_ranked = _importance_ranked(gated, mode, relationship_target)

    ranked_lists = [semantic_ranked, keyword_ranked, recency_ranked, importance_ranked]
    if mode == "old":
        w_recency, w_importance = 1.0, 1.0
    else:
        w_recency, w_importance = mem_cfg.weight_recency, mem_cfg.weight_importance
    weights = [mem_cfg.weight_semantic, mem_cfg.weight_keyword, w_recency, w_importance]

    if relationship_target:
        ranked_lists.append(_relationship_ranked(gated, relationship_target, mode))
        weights.append(mem_cfg.weight_relationship)

    fused = reciprocal_rank_fusion(ranked_lists, k=mem_cfg.rrf_k, weights=weights)
    return [mid for mid, _ in fused], {m["id"] for m in gated}


def _importance_ranked(gated: list[dict], mode: str, relationship_target: str) -> list[str]:
    """Strategy 4 (importance + tier boost + user affinity). Tie-break is the only thing that
    differs by mode: OLD (pre-5c84a76) breaks ties ascending-rowid (oldest wins); NEW breaks
    ties descending-rowid (newest wins) — see fuse_rank's docstring / the 5c84a76 diff."""
    tier_boosts = _get_tier_boosts()
    importance_scores = []
    for m in gated:
        base = m.get("importance", 0.5) * m.get("certainty", 1.0)
        boost = tier_boosts.get(m.get("tier", "buffer"), 0.0)
        meta = m.get("metadata") or {}
        if relationship_target and meta.get("user_id") == relationship_target:
            base += 0.2
        importance_scores.append((m["id"], base + boost, m.get("rowid", 0)))
    if mode == "old":
        importance_scores.sort(key=lambda x: (-x[1], x[2]))  # oldest wins ties (pre-fix)
    else:
        importance_scores.sort(key=lambda x: (-x[1], -x[2]))  # newest wins ties (current)
    return [mid for mid, _, _ in importance_scores]


def _relationship_ranked(gated: list[dict], relationship_target: str, mode: str) -> list[str]:
    """Strategy 5 (relationship boost). OLD (pre-5c84a76) ranks EVERY gated candidate — the
    untouched majority all tie at 0.0 and Python's stable sort then preserves `gated`'s
    ascending-rowid input order for that tied group (oldest-first bias). NEW ranks only
    candidates with a positive boost, ties among those broken descending-rowid (newest wins) —
    see fuse_rank's docstring / the 5c84a76 diff."""
    target_lower = relationship_target.lower()
    if mode == "old":
        rel_scores = []
        for m in gated:
            content_lower = m["content"].lower()
            meta = m.get("metadata") or {}
            involves = target_lower in content_lower or meta.get("target_id") == relationship_target
            rel_scores.append((m["id"], 1.0 if involves else 0.0))
        rel_scores.sort(key=lambda x: x[1], reverse=True)  # stable: 0.0-tie keeps input order
        return [mid for mid, _ in rel_scores]
    rel_scores = []
    for m in gated:
        content_lower = m["content"].lower()
        meta = m.get("metadata") or {}
        involves = target_lower in content_lower or meta.get("target_id") == relationship_target
        if involves:
            rel_scores.append((m["id"], 1.0, m.get("rowid", 0)))
    rel_scores.sort(key=lambda x: (-x[1], -x[2]))
    return [mid for mid, _, _ in rel_scores]


class _FakeMemCfg:
    """Minimal stand-in for MemoryConfig, weights fully controlled by the caller (unlike the
    live get_config().memory, whose weight_importance=0.0 default would zero out importance's
    RRF contribution regardless of tiebreak direction — useless for isolating that tiebreak)."""

    def __init__(self, **kw):
        self.weight_semantic = 0.0
        self.weight_keyword = 0.0
        self.weight_recency = 0.0
        self.weight_importance = 0.0
        self.weight_relationship = 0.0
        self.rrf_k = 60
        self.relevance_gate = False
        self.__dict__.update(kw)


def _mem(mid: str, rowid: int, importance: float = 0.5, content: str = "filler") -> dict:
    return {
        "id": mid,
        "tier": "core",
        "content": content,
        "embedding": [1.0, 0.0],
        "importance": importance,
        "certainty": 1.0,
        "metadata": {},
        "rowid": rowid,
        "created_at": "2024-01-01 00:00:00",
    }


def _selftest_old_tiebreaks() -> dict:
    """DB-free check that OLD/NEW mode really do what their docstrings claim, by calling the
    exact tie-break helpers fuse_rank() itself calls (:func:`_importance_ranked`,
    :func:`_relationship_ranked`) directly — not through the full RRF fusion, since OLD mode's
    hardcoded weight_recency=weight_importance=1.0 (mirroring v2's actual defaults) means the
    recency and importance lists both always contribute at full strength in OLD, confounding any
    attempt to isolate just the relationship list's contribution via RRF weights alone.

    Part 1 (importance tiebreak): 4 candidates tied on importance/certainty/tier. OLD ties break
    ascending-rowid (oldest wins) -> [m1,m2,m3,m4]. NEW ties break descending-rowid (newest
    wins) -> [m4,m3,m2,m1].

    Part 2 (relationship list membership + tiebreak): 4 candidates (m1 rowid1, m3 rowid3 don't
    mention "alice"; m2 rowid2, m4 rowid4 do). OLD ranks ALL 4 — matches first, in original
    ascending-rowid order (m2 then m4), then the untouched m1/m3 tied at 0.0 in that same
    original order -> [m2, m4, m1, m3]. NEW ranks ONLY the 2 matching candidates, ties among
    them broken descending-rowid (newest wins) -> [m4, m2] (m1/m3 excluded from the list
    entirely, not merely ranked last).
    """
    mems = [_mem(f"m{i}", i) for i in range(1, 5)]
    old_imp = _importance_ranked(mems, "old", "")
    new_imp = _importance_ranked(mems, "new", "")
    imp_old_ok = old_imp == ["m1", "m2", "m3", "m4"]
    imp_new_ok = new_imp == ["m4", "m3", "m2", "m1"]

    rel_mems = [
        _mem("m1", 1, content="filler"),
        _mem("m2", 2, content="alice was there"),
        _mem("m3", 3, content="filler"),
        _mem("m4", 4, content="alice again"),
    ]
    old_rel = _relationship_ranked(rel_mems, "alice", "old")
    new_rel = _relationship_ranked(rel_mems, "alice", "new")
    rel_old_ok = old_rel == ["m2", "m4", "m1", "m3"]
    rel_new_ok = new_rel == ["m4", "m2"]

    return {
        "importance_old_ranked": old_imp,
        "importance_new_ranked": new_imp,
        "importance_old_oldest_first_as_claimed": imp_old_ok,
        "importance_new_newest_first_as_claimed": imp_new_ok,
        "relationship_old_ranked": old_rel,
        "relationship_new_ranked": new_rel,
        "relationship_old_ranks_all_gated_oldest_tie_as_claimed": rel_old_ok,
        "relationship_new_boosted_only_newest_tie_as_claimed": rel_new_ok,
        "passed": imp_old_ok and imp_new_ok and rel_old_ok and rel_new_ok,
    }


# ── Loading judged records ────────────────────────────────────────────────


def load_old_records(run_id: str) -> dict[str, dict]:
    path = RESULTS_DIR / f"external_{run_id}.json"
    data = json.loads(path.read_text())
    return {r["qid"]: r for r in data["conversations"] if r.get("kind") == "qa"}


def load_new_records(run_id: str) -> tuple[dict[str, dict], dict[str, int]]:
    run_dir = RUNS_DIR / run_id
    files = sorted(run_dir.glob("conv-*.answers.json"))
    out: dict[str, dict] = {}
    file_counts: dict[str, int] = {}
    for f in files:
        recs = json.loads(f.read_text())
        file_counts[f.name] = len(recs)
        for r in recs:
            if r.get("kind") == "qa":
                out[r["qid"]] = r
    return out, file_counts


# ── Per-question reconstruction ───────────────────────────────────────────


def _evidence_active_ids(q, dia_to_turn: dict, raw_rows: list[dict]) -> set[str]:
    ids: set[str] = set()
    for dia_id in q.evidence:
        turn = dia_to_turn.get(dia_id)
        if turn is None:
            continue
        loc = _locate_evidence(turn.text, raw_rows)
        for mid in set(loc["exact_ids"]) | set(loc["fuzzy_ids"]):
            if loc["statuses"].get(mid) == "active":
                ids.add(mid)
    return ids


def validate_new(conv_data: dict, qids_by_conv: dict[str, list[str]], n: int) -> dict:
    """Sample n qa items and confirm fuse_rank(mode="new") reproduces real
    MemoryRetriever.retrieve()'s top-20 exactly (same technique as
    ranking_experiments.validate_v0)."""
    rng = random.Random(SAMPLE_SEED)
    all_pairs = [(cid, qid) for cid, qids in qids_by_conv.items() for qid in qids]
    rng.shuffle(all_pairs)
    sample = all_pairs[:n]
    exact = 0
    set_only = 0
    mismatches: list[dict] = []
    for conv_id, qid in sample:
        cd = conv_data[conv_id]
        q = cd["question_by_qid"][qid]
        clock.override(q.asked_at)
        real_top20 = [
            m["id"]
            for m in cd["char"].retriever.retrieve(
                q.question, limit=TOP_K, relationship_target=cd["user_name"]
            )
        ]
        query_vec = cd["emb"].embed(q.question)
        all_memories, fts_ids = build_candidates(
            cd["storage"], conv_id, cd["base_candidates"], q.question
        )
        reimpl, _ = fuse_rank(
            all_memories=all_memories,
            fts_ids=fts_ids,
            query=q.question,
            query_vec=query_vec,
            relationship_target=cd["user_name"],
            mode="new",
            mem_cfg=get_config().memory,
        )
        reimpl_top20 = reimpl[:TOP_K]
        if reimpl_top20 == real_top20:
            exact += 1
        elif set(reimpl_top20) == set(real_top20):
            set_only += 1
            mismatches.append({"qid": qid, "kind": "order_only"})
        else:
            mismatches.append(
                {"qid": qid, "kind": "set_mismatch", "real": real_top20, "reimpl": reimpl_top20}
            )
    clock.override(None)
    return {
        "n": len(sample),
        "exact_order_matches": exact,
        "same_set_diff_order": set_only,
        "set_mismatches": len(sample) - exact - set_only,
        "mismatch_examples": mismatches[:5],
    }


def compute_items(
    common_qids: list[str],
    old_qa: dict[str, dict],
    new_qa: dict[str, dict],
    new_run_id: str,
) -> tuple[list[dict], dict]:
    """Build one fully-reconstructed record per common qid. Returns (items, conv_data) —
    conv_data is kept around so validate_new() can reuse the already-built engines/embedders."""
    dataset_convs = {c.conv_id: c for c in load_locomo(LOCOMO_PATH)}
    by_conv: dict[str, list[str]] = defaultdict(list)
    for qid in common_qids:
        rec = old_qa[qid]
        by_conv[rec["conv_id"]].append(qid)

    mem_cfg = get_config().memory
    conv_data: dict[str, dict] = {}
    items: list[dict] = []

    for conv_id in sorted(by_conv):
        dataset_conv = dataset_convs[conv_id]
        db_path = _copy_db(conv_id, run_dir=RUNS_DIR / new_run_id, diag_dir=DIAG_DIR)
        raw_rows = _load_all_memory_rows(db_path, conv_id)
        storage = SQLiteStorage(str(db_path))

        engine = Engine(
            db_path=str(db_path), llm=cast(LLMProvider, FakeLLM()), embedding=embedder()
        )
        char = engine.get_character(conv_id)
        char.background = False
        char.parallel = False
        char.enforce_consistency = False
        emb = MemoEmbedder(char.retriever.embedder)
        setattr(
            char.retriever, "embedder", emb
        )  # real retrieve() calls (validate_new) hit the same cache

        dia_to_turn = {
            t.dia_id: t for s in dataset_conv.sessions for t in s.turns if t.dia_id is not None
        }
        question_by_qid = {q.qid: q for q in dataset_conv.questions}
        base_candidates = storage.get_memories(conv_id, limit=mem_cfg.max_candidates)
        active_by_id = {m["id"]: m for m in base_candidates}

        conv_data[conv_id] = {
            "char": char,
            "storage": storage,
            "emb": emb,
            "user_name": dataset_conv.user_name,
            "question_by_qid": question_by_qid,
            "base_candidates": base_candidates,
            "active_by_id": active_by_id,
        }

        def get_row(mid: str, _active_by_id=active_by_id, _storage=storage) -> dict:
            row = _active_by_id.get(mid)
            if row is not None:
                return row
            row = _storage.get_memory(mid)
            if row is not None:
                _active_by_id[mid] = row
            return row or {"id": mid, "tier": "unknown", "content": "", "metadata": {}}

        for qid in by_conv[conv_id]:
            q = question_by_qid[qid]
            clock.override(q.asked_at)
            pinned, pinned_ids = char._format_pinned_block()
            facts_text = char._format_facts_block(dataset_conv.user_name, pinned_ids)

            query_vec = emb.embed(q.question)
            all_memories, fts_ids = build_candidates(storage, conv_id, base_candidates, q.question)

            old_full, old_gated = fuse_rank(
                all_memories=all_memories,
                fts_ids=fts_ids,
                query=q.question,
                query_vec=query_vec,
                relationship_target=dataset_conv.user_name,
                mode="old",
                mem_cfg=mem_cfg,
            )
            new_full, new_gated = fuse_rank(
                all_memories=all_memories,
                fts_ids=fts_ids,
                query=q.question,
                query_vec=query_vec,
                relationship_target=dataset_conv.user_name,
                mode="new",
                mem_cfg=mem_cfg,
            )
            old_top20 = old_full[:TOP_K]
            new_top20 = new_full[:TOP_K]

            evidence_ids = _evidence_active_ids(q, dia_to_turn, raw_rows)
            old_rec = old_qa[qid]
            new_rec = new_qa[qid]
            gold = (old_rec.get("gold") or "").strip()

            old_kinds = [row_kind(get_row(mid)) for mid in old_top20]
            new_kinds = [row_kind(get_row(mid)) for mid in new_top20]

            gold_lower = gold.lower()
            gold_in_old_top20 = bool(gold) and any(
                gold_lower in (get_row(mid).get("content") or "").lower() for mid in old_top20
            )
            gold_in_new_top20 = bool(gold) and any(
                gold_lower in (get_row(mid).get("content") or "").lower() for mid in new_top20
            )

            old_correct = bool(old_rec.get("correct"))
            new_correct = bool(new_rec.get("correct"))
            if old_correct and not new_correct:
                flip_class = "A_v2_correct_v2b_wrong"
            elif not old_correct and new_correct:
                flip_class = "B_v2_wrong_v2b_correct"
            elif old_correct and new_correct:
                flip_class = "both_correct"
            else:
                flip_class = "both_wrong"

            old_response = old_rec.get("response") or ""
            new_response = new_rec.get("response") or ""

            items.append(
                {
                    "qid": qid,
                    "conv_id": conv_id,
                    "category": old_rec.get("category"),
                    "question": q.question,
                    "gold": gold,
                    "old_correct": old_correct,
                    "new_correct": new_correct,
                    "flip_class": flip_class,
                    "old_response": old_response,
                    "new_response": new_response,
                    "old_abstained": metrics.is_abstention(old_response),
                    "new_abstained": metrics.is_abstention(new_response),
                    "evidence_active_ids": sorted(evidence_ids),
                    "evidence_resolved": bool(evidence_ids)
                    or any(
                        dia_to_turn.get(d) is not None for d in q.evidence
                    ),  # evidence turn(s) located in the dataset even if not stored active
                    "evidence_in_old_top20": bool(evidence_ids & set(old_top20)),
                    "evidence_in_new_top20": bool(evidence_ids & set(new_top20)),
                    "old_top20": old_top20,
                    "new_top20": new_top20,
                    "old_kinds": old_kinds,
                    "new_kinds": new_kinds,
                    "gold_in_facts": bool(gold) and gold_lower in facts_text.lower(),
                    "gold_in_old_top20": gold_in_old_top20,
                    "gold_in_new_top20": gold_in_new_top20,
                    "old_gated_size": len(old_gated),
                    "new_gated_size": len(new_gated),
                }
            )
        clock.override(None)

    return items, conv_data


# ── Item 1: flip matrix ────────────────────────────────────────────────────


def build_flip_matrix(items: list[dict]) -> dict:
    def resp_class(abstained: bool) -> str:
        return "abstained" if abstained else "answered_wrong"

    overall = Counter(it["flip_class"] for it in items)
    per_category: dict[str, Counter] = defaultdict(Counter)
    for it in items:
        per_category[str(it["category"])][it["flip_class"]] += 1

    a_flips = [it for it in items if it["flip_class"] == "A_v2_correct_v2b_wrong"]
    b_flips = [it for it in items if it["flip_class"] == "B_v2_wrong_v2b_correct"]
    a_response_class = Counter(resp_class(it["new_abstained"]) for it in a_flips)
    b_response_class = Counter(resp_class(it["old_abstained"]) for it in b_flips)

    return {
        "n_total": len(items),
        "overall": dict(overall),
        "per_category": {k: dict(v) for k, v in per_category.items()},
        "n_a_flips": len(a_flips),
        "n_b_flips": len(b_flips),
        "a_flip_v2b_response_class": dict(a_response_class),
        "b_flip_v2_response_class": dict(b_response_class),
    }


# ── Item 2: 40-sample table + 8 worked examples ────────────────────────────


def build_flip_sample(items: list[dict], n: int, worked_n: int, seed: int) -> dict:
    a_flips = [it for it in items if it["flip_class"] == "A_v2_correct_v2b_wrong"]
    rng = random.Random(seed)
    sample = a_flips[:]
    rng.shuffle(sample)
    sample = sample[:n]

    rows = []
    for it in sample:
        old_set, new_set = set(it["old_top20"]), set(it["new_top20"])
        gained = new_set - old_set
        lost = old_set - new_set
        gained_kinds = Counter(it["new_kinds"][it["new_top20"].index(mid)] for mid in gained)
        lost_kinds = Counter(it["old_kinds"][it["old_top20"].index(mid)] for mid in lost)
        rows.append(
            {
                "qid": it["qid"],
                "category": it["category"],
                "evidence_in_old_top20": it["evidence_in_old_top20"],
                "evidence_in_new_top20": it["evidence_in_new_top20"],
                "n_gained": len(gained),
                "n_lost": len(lost),
                "gained_kinds": dict(gained_kinds),
                "lost_kinds": dict(lost_kinds),
                "gold_in_facts": it["gold_in_facts"],
                "old_response": it["old_response"],
                "new_response": it["new_response"],
            }
        )

    worked = sample[:worked_n]
    return {"n_sampled": len(sample), "rows": rows, "worked_qids": [it["qid"] for it in worked]}


def render_worked_examples(items_by_qid: dict[str, dict], qids: list[str]) -> str:
    lines = ["# 8 fully worked A-flip examples (v2 correct -> v2b wrong)", ""]
    for qid in qids:
        it = items_by_qid[qid]
        lines.append(f"## {qid} (category {it['category']})")
        lines.append("")
        lines.append(f"**Question:** {it['question']}")
        lines.append(f"**Gold:** {it['gold']!r}")
        lines.append(f"**v2 (old ranking) response — CORRECT:** {it['old_response']!r}")
        lines.append(f"**v2b (new ranking) response — WRONG:** {it['new_response']!r}")
        lines.append("")
        lines.append(f"gold_in_facts_block: {it['gold_in_facts']}")
        lines.append(
            f"evidence_active_ids: {it['evidence_active_ids']} | "
            f"in OLD top20: {it['evidence_in_old_top20']} | in NEW top20: {it['evidence_in_new_top20']}"
        )
        lines.append("")
        old_set, new_set = set(it["old_top20"]), set(it["new_top20"])
        gained = [mid for mid in it["new_top20"] if mid not in old_set]
        lost = [mid for mid in it["old_top20"] if mid not in new_set]
        lines.append(f"OLD top20 kinds: {it['old_kinds']}")
        lines.append(f"NEW top20 kinds: {it['new_kinds']}")
        lines.append(
            f"Gained in NEW (not in OLD), {len(gained)}: "
            + ", ".join(f"{mid}[{it['new_kinds'][it['new_top20'].index(mid)]}]" for mid in gained)
        )
        lines.append(
            f"Lost from OLD (not in NEW), {len(lost)}: "
            + ", ".join(f"{mid}[{it['old_kinds'][it['old_top20'].index(mid)]}]" for mid in lost)
        )
        lines.append(
            f"gold_in_old_top20 (any row's content contains gold text): {it['gold_in_old_top20']}"
        )
        lines.append(
            f"gold_in_new_top20 (any row's content contains gold text): {it['gold_in_new_top20']}"
        )
        lines.append("")
    return "\n".join(lines)


# ── Item 3: aggregate top-20 composition + conditional correct-rate ───────


KIND_ORDER = ["buffer_raw_turn", "core_fact", "consolidated_summary", "bedrock", "other"]


def build_top20_aggregate(items: list[dict]) -> dict:
    n = len(items)

    def mean_composition(field: str) -> dict[str, float]:
        totals: dict[str, float] = {k: 0.0 for k in KIND_ORDER}
        for it in items:
            c = Counter(it[field])
            for k in KIND_ORDER:
                totals[k] += c.get(k, 0)
        return {k: (v / n if n else 0.0) for k, v in totals.items()}

    with_evidence = [it for it in items if it["evidence_active_ids"]]
    n_ev = len(with_evidence)
    recall_old = (
        sum(1 for it in with_evidence if it["evidence_in_old_top20"]) / n_ev if n_ev else 0.0
    )
    recall_new = (
        sum(1 for it in with_evidence if it["evidence_in_new_top20"]) / n_ev if n_ev else 0.0
    )

    def conditional_correct(rank_key: str, correct_key: str) -> dict:
        yes = [it for it in with_evidence if it[rank_key]]
        no = [it for it in with_evidence if not it[rank_key]]
        return {
            "n_evidence_in_top20": len(yes),
            "correct_rate_given_evidence_in_top20": (
                sum(1 for it in yes if it[correct_key]) / len(yes) if yes else 0.0
            ),
            "n_evidence_not_in_top20": len(no),
            "correct_rate_given_evidence_not_in_top20": (
                sum(1 for it in no if it[correct_key]) / len(no) if no else 0.0
            ),
        }

    return {
        "n_qa_total": n,
        "n_with_evidence_resolved": n_ev,
        "mean_top20_composition_old": mean_composition("old_kinds"),
        "mean_top20_composition_new": mean_composition("new_kinds"),
        "evidence_in_top20_rate_old": recall_old,
        "evidence_in_top20_rate_new": recall_new,
        "conditional_correct_rate_old": conditional_correct("evidence_in_old_top20", "old_correct"),
        "conditional_correct_rate_new": conditional_correct("evidence_in_new_top20", "new_correct"),
    }


# ── Item 4: consolidated/core rows ousted by the new ranking ──────────────


def build_consolidated_ousted(items: list[dict]) -> dict:
    a_flips = [it for it in items if it["flip_class"] == "A_v2_correct_v2b_wrong"]
    hits = []
    for it in a_flips:
        old_set, new_set = set(it["old_top20"]), set(it["new_top20"])
        ousted = []
        for mid in old_set - new_set:
            idx = it["old_top20"].index(mid)
            kind = it["old_kinds"][idx]
            if kind in ("core_fact", "consolidated_summary"):
                ousted.append({"id": mid, "kind": kind})
        # We don't have raw content cached in the item dict (kept lean); recompute containment
        # from gold_in_old_top20/gold_in_new_top20 flags plus the ousted-kind list: a strict
        # per-row gold-containment check for *these specific* rows is done in the qid-indexed
        # detail pass below (build_consolidated_ousted_detailed).
        if ousted:
            hits.append({"qid": it["qid"], "ousted_core_or_consolidated": ousted})
    return {
        "n_a_flips": len(a_flips),
        "n_a_flips_with_ousted_core_or_consolidated_row": len(hits),
        "examples": hits[:20],
    }


def build_consolidated_ousted_detailed(
    items_by_qid: dict[str, dict], conv_data: dict, item_index: dict[str, dict]
) -> dict:
    """Strict version of item 4: for each A flip, does the OLD top20 contain a core_fact or
    consolidated_summary row whose CONTENT contains the gold text, and is that specific row
    absent from the NEW top20?"""
    hits = []
    for qid, it in items_by_qid.items():
        if it["flip_class"] != "A_v2_correct_v2b_wrong":
            continue
        conv_id = it["conv_id"]
        get_row = conv_data[conv_id]["active_by_id"]
        gold_lower = it["gold"].lower()
        if not gold_lower:
            continue
        new_set = set(it["new_top20"])
        matches = []
        for mid in it["old_top20"]:
            idx = it["old_top20"].index(mid)
            kind = it["old_kinds"][idx]
            if kind not in ("core_fact", "consolidated_summary"):
                continue
            row = get_row.get(mid)
            content = (row.get("content") or "") if row else ""
            if gold_lower in content.lower() and mid not in new_set:
                matches.append({"id": mid, "kind": kind, "content": content[:200]})
        if matches:
            hits.append({"qid": qid, "gold": it["gold"], "matches": matches})
    return {
        "n_a_flips_with_gold_bearing_core_or_consolidated_row_dropped": len(hits),
        "examples": hits[:15],
    }


# ── Item 5: abstention despite evidence present ────────────────────────────


def build_abstention_despite_evidence(items: list[dict]) -> dict:
    def rates(evidence_key: str, abstain_key: str) -> dict:
        with_ev = [it for it in items if it[evidence_key]]
        return {
            "n_evidence_in_top20": len(with_ev),
            "abstain_rate_given_evidence_in_top20": (
                sum(1 for it in with_ev if it[abstain_key]) / len(with_ev) if with_ev else 0.0
            ),
        }

    overall_abstain_old = (
        sum(1 for it in items if it["old_abstained"]) / len(items) if items else 0.0
    )
    overall_abstain_new = (
        sum(1 for it in items if it["new_abstained"]) / len(items) if items else 0.0
    )
    return {
        "overall_abstain_rate_old": overall_abstain_old,
        "overall_abstain_rate_new": overall_abstain_new,
        "old": rates("evidence_in_old_top20", "old_abstained"),
        "new": rates("evidence_in_new_top20", "new_abstained"),
    }


# ── Report rendering ────────────────────────────────────────────────────────


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def render_flip_matrix_md(fm: dict) -> str:
    lines = ["# Flip matrix (v2 vs v2b, on qa questions judged by both)", ""]
    lines.append(f"n common qa questions: {fm['n_total']}")
    lines.append("")
    lines.append("| class | count |")
    lines.append("|---|---:|")
    for k in ["both_correct", "both_wrong", "A_v2_correct_v2b_wrong", "B_v2_wrong_v2b_correct"]:
        lines.append(f"| {k} | {fm['overall'].get(k, 0)} |")
    lines.append("")
    lines.append("## Per category")
    lines.append("")
    lines.append("| category | both_correct | both_wrong | A (v2✓→v2b✗) | B (v2✗→v2b✓) |")
    lines.append("|---|---:|---:|---:|---:|")
    for cat in sorted(fm["per_category"]):
        c = fm["per_category"][cat]
        lines.append(
            f"| {cat} | {c.get('both_correct', 0)} | {c.get('both_wrong', 0)} | "
            f"{c.get('A_v2_correct_v2b_wrong', 0)} | {c.get('B_v2_wrong_v2b_correct', 0)} |"
        )
    lines.append("")
    lines.append(
        f"A flips ({fm['n_a_flips']}) — v2b response class: {fm['a_flip_v2b_response_class']}"
    )
    lines.append(
        f"B flips ({fm['n_b_flips']}) — v2 response class: {fm['b_flip_v2_response_class']}"
    )
    lines.append("")
    return "\n".join(lines)


def render_top20_aggregate_md(agg: dict) -> str:
    lines = ["# Top-20 composition and evidence-conditional correctness (all common qa items)", ""]
    lines.append(
        f"n qa total: {agg['n_qa_total']}, n with evidence resolved: {agg['n_with_evidence_resolved']}"
    )
    lines.append("")
    lines.append("## Mean top-20 composition")
    lines.append("")
    lines.append("| kind | OLD (v2) | NEW (v2b) |")
    lines.append("|---|---:|---:|")
    for k in KIND_ORDER:
        lines.append(
            f"| {k} | {agg['mean_top20_composition_old'][k]:.2f} | {agg['mean_top20_composition_new'][k]:.2f} |"
        )
    lines.append("")
    lines.append("## Evidence-in-top20 rate")
    lines.append("")
    lines.append(
        f"OLD: {_pct(agg['evidence_in_top20_rate_old'])}  NEW: {_pct(agg['evidence_in_top20_rate_new'])}"
    )
    lines.append("")
    lines.append("## Correct-rate conditional on evidence-in-top20 (KEY NUMBER)")
    lines.append("")
    lines.append("| | OLD (v2) | NEW (v2b) |")
    lines.append("|---|---:|---:|")
    co, cn = agg["conditional_correct_rate_old"], agg["conditional_correct_rate_new"]
    lines.append(
        f"| n evidence-in-top20 | {co['n_evidence_in_top20']} | {cn['n_evidence_in_top20']} |"
    )
    lines.append(
        f"| correct-rate \\| evidence-in-top20 | {_pct(co['correct_rate_given_evidence_in_top20'])} | "
        f"{_pct(cn['correct_rate_given_evidence_in_top20'])} |"
    )
    lines.append(
        f"| n evidence-NOT-in-top20 | {co['n_evidence_not_in_top20']} | {cn['n_evidence_not_in_top20']} |"
    )
    lines.append(
        f"| correct-rate \\| evidence-NOT-in-top20 | {_pct(co['correct_rate_given_evidence_not_in_top20'])} | "
        f"{_pct(cn['correct_rate_given_evidence_not_in_top20'])} |"
    )
    lines.append("")
    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--old", default=OLD_RUN_ID_DEFAULT, help="old (v2) run id")
    parser.add_argument("--new", default=NEW_RUN_ID_DEFAULT, help="new (v2b) run id")
    parser.add_argument("--sample-n", type=int, default=SAMPLE_N)
    parser.add_argument("--worked-n", type=int, default=WORKED_N)
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED)
    parser.add_argument("--validate-n", type=int, default=VALIDATE_N)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir) if args.out_dir else DIAG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading old run {args.old!r}...")
    old_qa = load_old_records(args.old)
    print(f"  {len(old_qa)} qa records")

    print(f"loading new run {args.new!r} (may still be writing)...")
    new_qa, file_counts = load_new_records(args.new)
    print(f"  files seen: {file_counts}")
    print(f"  {len(new_qa)} qa records total across those files")

    common_qids = sorted(set(old_qa) & set(new_qa))
    print(f"common qa qids judged by both runs: {len(common_qids)}")

    selftest = _selftest_old_tiebreaks()
    print(f"OLD tie-break self-test: {'PASS' if selftest['passed'] else 'FAIL'}")

    print("reconstructing OLD/NEW top-20 rankings from the DBs (embedding server calls, no LLM)...")
    items, conv_data = compute_items(common_qids, old_qa, new_qa, args.new)
    print(f"  {len(items)} items reconstructed")

    validation = {}
    if not args.skip_validation:
        qids_by_conv: dict[str, list[str]] = defaultdict(list)
        for it in items:
            qids_by_conv[it["conv_id"]].append(it["qid"])
        print(
            f"validating NEW mode against real MemoryRetriever.retrieve() on {args.validate_n} items..."
        )
        validation = validate_new(conv_data, qids_by_conv, args.validate_n)
        print(
            f"  exact order match: {validation['exact_order_matches']}/{validation['n']}, "
            f"same set diff order: {validation['same_set_diff_order']}, "
            f"set mismatches: {validation['set_mismatches']}"
        )

    (out_dir / "all_items.json").write_text(json.dumps(items, indent=2, default=str))
    (out_dir / "validation.json").write_text(
        json.dumps({"new_mode_validation": validation, "old_tiebreak_selftest": selftest}, indent=2)
    )

    # Item 1
    flip_matrix = build_flip_matrix(items)
    (out_dir / "flip_matrix.json").write_text(json.dumps(flip_matrix, indent=2))
    (out_dir / "flip_matrix.md").write_text(render_flip_matrix_md(flip_matrix))
    print(f"flip matrix: {flip_matrix['overall']}")

    # Item 2
    flip_sample = build_flip_sample(items, args.sample_n, args.worked_n, args.seed)
    (out_dir / "flip_sample.json").write_text(json.dumps(flip_sample, indent=2))
    items_by_qid = {it["qid"]: it for it in items}
    worked_md = render_worked_examples(items_by_qid, flip_sample["worked_qids"])
    (out_dir / "flip_sample_worked_examples.md").write_text(worked_md)
    sample_table_lines = [
        "# 40-sample A-flip table",
        "",
        "| qid | cat | ev_old | ev_new | +gained | -lost | gold_in_facts |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in flip_sample["rows"]:
        sample_table_lines.append(
            f"| {r['qid']} | {r['category']} | {r['evidence_in_old_top20']} | {r['evidence_in_new_top20']} | "
            f"{r['n_gained']} ({r['gained_kinds']}) | {r['n_lost']} ({r['lost_kinds']}) | {r['gold_in_facts']} |"
        )
    (out_dir / "flip_sample.md").write_text("\n".join(sample_table_lines))
    print(
        f"sampled {flip_sample['n_sampled']} A flips; {len(flip_sample['worked_qids'])} worked in full"
    )

    # Item 3
    top20_agg = build_top20_aggregate(items)
    (out_dir / "top20_aggregate.json").write_text(json.dumps(top20_agg, indent=2))
    (out_dir / "top20_aggregate.md").write_text(render_top20_aggregate_md(top20_agg))
    print(
        f"evidence-in-top20: old={_pct(top20_agg['evidence_in_top20_rate_old'])} "
        f"new={_pct(top20_agg['evidence_in_top20_rate_new'])}"
    )
    print(
        f"conditional correct|evidence-in-top20: "
        f"old={_pct(top20_agg['conditional_correct_rate_old']['correct_rate_given_evidence_in_top20'])} "
        f"new={_pct(top20_agg['conditional_correct_rate_new']['correct_rate_given_evidence_in_top20'])}"
    )

    # Item 4
    consolidated_ousted = build_consolidated_ousted(items)
    consolidated_ousted_detailed = build_consolidated_ousted_detailed(items_by_qid, conv_data, {})
    (out_dir / "consolidated_ousted.json").write_text(
        json.dumps(
            {"loose": consolidated_ousted, "strict_gold_bearing": consolidated_ousted_detailed},
            indent=2,
        )
    )
    print(
        f"A flips with a dropped core/consolidated row (loose): "
        f"{consolidated_ousted['n_a_flips_with_ousted_core_or_consolidated_row']}/{consolidated_ousted['n_a_flips']}; "
        f"strict (gold-bearing, actually dropped): "
        f"{consolidated_ousted_detailed['n_a_flips_with_gold_bearing_core_or_consolidated_row_dropped']}"
    )

    # Item 5
    abstention = build_abstention_despite_evidence(items)
    (out_dir / "abstention_despite_evidence.json").write_text(json.dumps(abstention, indent=2))
    print(
        f"abstain-given-evidence-in-top20: old={_pct(abstention['old']['abstain_rate_given_evidence_in_top20'])} "
        f"new={_pct(abstention['new']['abstain_rate_given_evidence_in_top20'])}"
    )

    print(f"wrote all outputs to {out_dir}")


if __name__ == "__main__":
    main()
