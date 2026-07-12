"""Offline batch maintenance — the nightly-job primitive.

All heavy multi-pass LLM work (consolidation, reflection, growth, dedup,
reinforcement, contradiction sweeps, callback generation) runs here:
budgeted, idempotent, per-character, callable headless. This is where
small models get retries and where "runs overnight while charging" lives.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

from .log import logger


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
    # Task 7 ships only the non-LLM-heavy-dependency-free core three jobs.
    # dedup/reinforce/contradictions/reflect/evolve arrive in Tasks 8-9,
    # callbacks in Task 10 — those job handlers get appended to DEFAULT_JOBS
    # as they land.
    DEFAULT_JOBS = [
        "consolidate",
        "buffer_hygiene",
        "score_importance",
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
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.cfg.buffer_ttl_days)
        cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")
        limit = 1000
        candidates = char.storage.get_memories(char.id, tier="buffer", limit=limit)
        # get_memories returns newest-first; the TTL sweep cares about the
        # oldest rows, so sort ascending before filtering/capping.
        candidates.sort(key=lambda m: m.get("created_at") or "")
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
        fetched = char.storage.get_memories(char.id, tier="buffer", limit=limit)
        # get_memories returns newest-first; the scoring sweep cares about the
        # oldest rows, so sort ascending before filtering/capping.
        fetched.sort(key=lambda m: m.get("created_at") or "")
        extra = {"truncated": True} if len(fetched) == limit else {}
        candidates = [m for m in fetched if m.get("importance") == 0.5][
            : self.cfg.importance_scoring_batch
        ]
        if not candidates:
            return {"_status": "skipped", "reason": "nothing to score", **extra}
        if not self.budget.take(1):
            return {"_status": "skipped", "reason": "budget exhausted", **extra}
        numbered = "\n".join(f"{i + 1}. {m['content'][:200]}" for i, m in enumerate(candidates))
        messages = [
            {
                "role": "system",
                "content": (
                    "You score the long-term importance of memories on a 1-10 scale "
                    "(1 = mundane small talk, 10 = life-changing). "
                    "Return a JSON array of integers, one per numbered memory, in order."
                ),
            },
            {"role": "user", "content": f"Score these memories:\n{numbered}"},
        ]
        scores = char.llm.generate_json_robust(messages)
        if not isinstance(scores, list):
            return {"_status": "failed", "error": "non-list score response", **extra}
        scored = 0
        for m, s in zip(candidates, scores):
            if isinstance(s, (int, float)):
                char.storage.set_memory_importance(m["id"], max(0.1, min(0.9, float(s) / 10)))
                scored += 1
        return {"scored": scored, "candidates": len(candidates), **extra}
