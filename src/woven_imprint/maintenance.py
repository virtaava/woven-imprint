"""Offline batch maintenance — the nightly-job primitive.

All heavy multi-pass LLM work (consolidation, reflection, growth, dedup,
reinforcement, contradiction sweeps, callback generation) runs here:
budgeted, idempotent, per-character, callable headless. This is where
small models get retries and where "runs overnight while charging" lives.
"""

from __future__ import annotations

import time
from datetime import timedelta

from . import clock
from .log import logger
from .memory.retrieval import cosine_matrix
from .prompts import render


class Budget:
    """LLM-call budget for one maintenance run. limit=None → unlimited."""

    def __init__(self, limit: int | None):
        self.limit = limit
        self.used = 0

    def take(self, n: int = 1) -> bool:
        if self.limit is not None and self.used + n > self.limit:
            return False
        self.used += n
        return True

    @property
    def remaining(self) -> int | None:
        return None if self.limit is None else max(0, self.limit - self.used)


class MaintenanceRunner:
    # Task 7 shipped the non-LLM-heavy-dependency-free core three jobs;
    # Task 8 added dedup/reinforce/contradictions; Task 9 added reflect/evolve;
    # Task 10 added callbacks.
    DEFAULT_JOBS = [
        "consolidate",
        "buffer_hygiene",
        "score_importance",
        "dedup",
        "reinforce",
        "contradictions",
        "reflect",
        "evolve",
        "callbacks",
    ]

    def __init__(self, character, budget: Budget | None = None):
        from .config import get_config

        self.character = character
        self.cfg = get_config().maintenance
        self.budget = budget if budget is not None else Budget(self.cfg.max_llm_calls_per_run)

    def run(self, jobs: list[str] | None = None) -> dict:
        report: dict = {"character_id": self.character.id, "jobs": {}, "llm_calls_used": 0}
        for name in jobs or self.DEFAULT_JOBS:
            started = time.perf_counter()
            used_before = self.budget.used
            handler = getattr(self, f"_job_{name}", None)
            entry: dict
            if handler is None:
                entry = {"status": "failed", "error": f"unknown job: {name}"}
            else:
                try:
                    details = handler()
                    entry = {"status": details.pop("_status", "ok"), **details}
                except Exception as e:
                    logger.warning("Maintenance job '%s' failed: %s", name, e)
                    entry = {"status": "failed", "error": f"{type(e).__name__}: {e}"[:300]}
            entry["llm_calls"] = self.budget.used - used_before
            entry["duration_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
            report["jobs"][name] = entry
        report["llm_calls_used"] = self.budget.used
        return report

    # ── Jobs ──────────────────────────────────────────────────

    def _job_consolidate(self) -> dict:
        char = self.character
        if not char.consolidator.needs_consolidation():
            return {"_status": "skipped", "reason": "buffer below threshold"}
        if self.budget.remaining == 0:
            return {"_status": "skipped", "reason": "budget exhausted"}
        return char.consolidator.drain(budget=self.budget)

    def _job_buffer_hygiene(self) -> dict:
        char = self.character
        cutoff = clock.now() - timedelta(days=self.cfg.buffer_ttl_days)
        cutoff_str = clock.sqlite_ts(cutoff)
        limit = 1000
        candidates = char.storage.get_memories(
            char.id, tier="buffer", limit=limit, oldest_first=True
        )
        stale = [
            m["id"]
            for m in candidates
            if (m.get("created_at") or "") < cutoff_str
            and m.get("importance", 0.5) <= self.cfg.buffer_hygiene_max_importance
        ]
        char.storage.archive_memories_batch(stale)
        result = {"archived": len(stale)}
        if len(candidates) == limit:
            result["truncated"] = True
        return result

    def _job_score_importance(self) -> dict:
        char = self.character
        limit = 500
        fetched = char.storage.get_memories(char.id, tier="buffer", limit=limit, oldest_first=True)
        extra = {"truncated": True} if len(fetched) == limit else {}
        candidates = [m for m in fetched if m.get("importance") == 0.5][
            : self.cfg.importance_scoring_batch
        ]
        if not candidates:
            return {"_status": "skipped", "reason": "nothing to score", **extra}
        if not self.budget.take(1):
            return {"_status": "skipped", "reason": "budget exhausted", **extra}
        numbered = "\n".join(
            f"{i + 1}. ({(m.get('created_at') or '')[:10]}) {m['content'][:200]}"
            for i, m in enumerate(candidates)
        )
        messages = render("maintenance_importance", numbered=numbered)
        scores = char.llm.generate_json_robust(messages)
        if not isinstance(scores, list):
            return {"_status": "failed", "error": "non-list score response", **extra}
        scored = 0
        for m, s in zip(candidates, scores):
            if isinstance(s, (int, float)):
                score = max(0.1, min(0.9, float(s) / 10))
                # 0.5 is the "unscored" sentinel (see the `importance == 0.5`
                # candidate filter above) — a genuine 5/10 score would land
                # exactly on it and get re-picked as a candidate forever,
                # starving newer unscored memories of budget. Nudge it off.
                if score == 0.5:
                    score = 0.51
                char.storage.set_memory_importance(m["id"], score)
                scored += 1
        return {"scored": scored, "candidates": len(candidates), **extra}

    def _active_core_observations(self, limit: int) -> list[dict]:
        return [
            m
            for m in self.character.storage.get_memories(
                self.character.id, tier="core", limit=limit
            )
            if m.get("role") == "observation" and m.get("embedding")
        ]

    def _job_dedup(self) -> dict:
        mems = self._active_core_observations(self.cfg.dedup_scan_limit)
        # Similarities are fetched a row at a time via cosine_matrix (one
        # batched/vectorized call per i against every embedding) instead of
        # one _cosine_similarity call per (i, j) pair — same values, O(n)
        # numpy calls instead of O(n^2) pure-Python ones. See maintenance
        # perf note in eval/bench_longhorizon.py's report for why this
        # mattered (a higher-dimensional benchmark embedder made the O(n^2)
        # pairwise Python path the dominant cost of the whole suite).
        embeddings = [m["embedding"] for m in mems]
        archived: list[str] = []
        archived_set: set[str] = set()
        for i in range(len(mems)):
            if mems[i]["id"] in archived_set:
                continue
            row = cosine_matrix(embeddings[i], embeddings)
            for j in range(i + 1, len(mems)):
                if mems[j]["id"] in archived_set:
                    continue
                sim = row[j]
                if sim < self.cfg.dedup_similarity:
                    continue
                if mems[i].get("importance", 0) >= mems[j].get("importance", 0):
                    keep, drop = mems[i], mems[j]
                else:
                    keep, drop = mems[j], mems[i]
                archived.append(drop["id"])
                archived_set.add(drop["id"])
                if keep["id"] not in archived_set:
                    self.character.belief.reinforce(keep["id"])
                if drop is mems[i]:
                    break  # i itself archived — stop comparing it
        self.character.storage.archive_memories_batch(archived)
        return {"archived": len(archived), "scanned": len(mems)}

    def _job_reinforce(self) -> dict:
        char = self.character
        cores = self._active_core_observations(self.cfg.dedup_scan_limit)
        buffers = [
            m
            for m in char.storage.get_memories(char.id, tier="buffer", limit=200)
            if m.get("embedding")
        ]
        core_embeddings = [c["embedding"] for c in cores]
        reinforced: set[str] = set()
        for b in buffers:
            # One batched cosine_matrix call per buffer memory against every
            # core, instead of one _cosine_similarity call per (buffer, core)
            # pair — see _job_dedup's comment.
            sims = cosine_matrix(b["embedding"], core_embeddings)
            for c, sim in zip(cores, sims):
                if c["id"] in reinforced:
                    continue
                if sim >= self.cfg.reinforce_similarity:
                    char.belief.reinforce(c["id"])
                    reinforced.add(c["id"])
        return {"reinforced": len(reinforced), "buffer_scanned": len(buffers)}

    def _job_contradictions(self) -> dict:
        char = self.character
        # _active_core_observations is newest-first by default (correct for
        # dedup/reinforce scan priority). For contradiction pairing, sort a
        # local copy chronologically so "first"/"second" in the LLM prompt
        # consistently mean older/newer — the verdict's superseded/current
        # resolution below depends on that ordering.
        mems = sorted(
            self._active_core_observations(self.cfg.dedup_scan_limit),
            key=lambda m: (m.get("created_at") or "", m.get("rowid", 0)),
        )
        # See _job_dedup's comment: batched cosine_matrix per row instead of
        # one _cosine_similarity call per pair.
        embeddings = [m["embedding"] for m in mems]
        pairs = []
        for i in range(len(mems)):
            row = cosine_matrix(embeddings[i], embeddings)
            for j in range(i + 1, len(mems)):
                sim = row[j]
                if self.cfg.contradiction_candidate_similarity <= sim < self.cfg.dedup_similarity:
                    pairs.append((sim, mems[i], mems[j]))
        pairs.sort(key=lambda p: p[0], reverse=True)
        contradicted = 0
        checked = 0
        for _sim, a, b in pairs[: self.cfg.contradiction_max_pairs]:
            if not self.budget.take(1):
                break
            checked += 1
            messages = render(
                "maintenance_contradiction",
                a_date=(a.get("created_at") or "")[:10],
                a_content=a["content"][:300],
                b_date=(b.get("created_at") or "")[:10],
                b_content=b["content"][:300],
            )
            try:
                verdict = char.llm.generate_json_robust(messages)
            except ValueError:
                continue
            if not isinstance(verdict, dict) or not verdict.get("contradictory"):
                continue
            current = verdict.get("current")
            if current == "first":
                superseded = b
            elif current == "second":
                superseded = a
            else:
                continue  # unclear — leave both, low certainty is belief revision's job
            char.storage.update_memory_status(superseded["id"], "contradicted", certainty=0.0)
            contradicted += 1
        return {"contradicted": contradicted, "pairs_checked": checked, "candidates": len(pairs)}

    def _job_reflect(self) -> dict:
        char = self.character
        reflections = [
            m
            for m in char.storage.get_memories(char.id, tier="core", limit=1000)
            if m["content"].startswith("[Reflection]")
        ]
        last_ts = max((m.get("created_at") or "" for m in reflections), default="")
        buffer = char.storage.get_memories(char.id, tier="buffer", limit=1000)
        pending = sum(
            m.get("importance", 0.5) for m in buffer if (m.get("created_at") or "") > last_ts
        )
        if pending < self.cfg.reflect_importance_sum:
            return {"_status": "skipped", "reason": f"importance sum {pending:.1f} below threshold"}
        if not self.budget.take(1):
            return {"_status": "skipped", "reason": "budget exhausted"}
        char.reflect()
        return {"triggered_at_sum": round(pending, 1)}

    def _job_evolve(self) -> dict:
        from .config import get_config

        persona_cfg = get_config().persona
        char = self.character
        core_count = char.storage.count_memories(char.id, tier="core")
        if core_count < persona_cfg.growth_min_memories:
            return {"_status": "skipped", "reason": f"{core_count} core memories < min"}
        if not self.budget.take(1):
            return {"_status": "skipped", "reason": "budget exhausted"}
        events = char.evolve(
            min_memories=persona_cfg.growth_min_memories,
            threshold=persona_cfg.growth_threshold,
        )
        return {"growth_events": len(events)}

    def _job_callbacks(self) -> dict:
        if self.budget.remaining == 0:
            return {"_status": "skipped", "reason": "budget exhausted", "created": 0}
        created = self.character.refresh_callbacks(budget=self.budget)
        return {"created": created}
