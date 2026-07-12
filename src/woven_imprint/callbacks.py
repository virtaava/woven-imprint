"""Callback surfacing — the research-backed #1 companion feature.

Callbacks are generated in BATCH (maintenance job / session end) and read
INSTANTLY from the DB. Hard rule: hooks are paraphrased, in-character
references — never verbatim quotes of stored memories (verbatim reads as
creepy; paraphrase captures the benefit).
"""

from __future__ import annotations

from datetime import datetime, timezone

from .log import logger
from .utils.text import generate_id

VALID_KINDS = ("open_thread", "callback", "milestone", "curiosity")


def _freshness(created_at: str) -> str:
    try:
        created = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - created
        if delta.days >= 1:
            return f"{delta.days}d"
        hours = delta.seconds // 3600
        return f"{hours}h" if hours else f"{delta.seconds // 60}m"
    except (ValueError, AttributeError):
        return "?"


class CallbackEngine:
    def __init__(self, character):
        from .config import get_config

        self.character = character
        self.cfg = get_config().maintenance

    # ── Read path (instant) ───────────────────────────────────

    def get(self, limit: int = 3) -> list[dict]:
        rows = self.character.storage.get_callbacks(self.character.id, status="ready", limit=limit)
        for r in rows:
            r["freshness"] = _freshness(r.get("created_at", ""))
        return rows

    # ── Generation (batch) ────────────────────────────────────

    def _gather_sources(self) -> list[dict]:
        char = self.character
        sources: list[dict] = []
        for m in char.storage.get_memories(char.id, tier="core", limit=200):
            if m.get("role") == "observation" and m.get("importance", 0) >= 0.7:
                sources.append(
                    {
                        "text": m["content"][:200],
                        "memory_id": m["id"],
                        "importance": m.get("importance", 0.5),
                    }
                )
        sources = sources[:15]
        for rel in char.relationships.get_all():
            for moment in (rel.get("key_moments") or [])[-5:]:
                sources.append({"text": str(moment)[:200], "memory_id": None, "importance": 0.5})
        for beat in char.arc.beats[-5:]:
            sources.append({"text": beat.description[:200], "memory_id": None, "importance": 0.5})
        return sources

    def refresh(self, budget=None, limit: int | None = None) -> int:
        char = self.character
        limit = limit or self.cfg.callbacks_refresh_limit
        sources = self._gather_sources()
        if not sources:
            return 0
        if budget is not None and not budget.take(1):
            return 0

        numbered = "\n".join(f"{i + 1}. {s['text']}" for i, s in enumerate(sources))
        messages = [
            {
                "role": "system",
                "content": (
                    f"You create conversation hooks for {char.name} — things the "
                    "character would naturally bring up with the person they know. "
                    f"Return a JSON array (max {limit}) of objects: "
                    '{"hook": <one natural in-character sentence>, '
                    '"kind": "open_thread"|"callback"|"milestone"|"curiosity", '
                    '"sources": [<numbers of the memories it draws on>]}. '
                    "RULES: paraphrase naturally — NEVER quote the memory text "
                    "verbatim. open_thread = unresolved thing to ask about; "
                    "callback = warm reference to a shared moment; milestone = "
                    "anniversary or achievement; curiosity = something the "
                    "character genuinely wonders about."
                ),
            },
            {"role": "user", "content": f"Memories and moments:\n{numbered}"},
        ]
        try:
            result = char.llm.generate_json_robust(messages)
        except ValueError as e:
            logger.debug("Callback generation failed: %s", e)
            return 0
        if not isinstance(result, list):
            return 0

        created = 0
        for item in result[:limit]:
            if not isinstance(item, dict) or not item.get("hook"):
                continue
            kind = item.get("kind", "callback")
            if kind not in VALID_KINDS:
                kind = "callback"
            idxs = [i for i in item.get("sources", []) if isinstance(i, int)]
            mem_ids = [
                sources[i - 1]["memory_id"]
                for i in idxs
                if 0 < i <= len(sources) and sources[i - 1]["memory_id"]
            ]
            sals = [sources[i - 1]["importance"] for i in idxs if 0 < i <= len(sources)]
            char.storage.save_callback(
                {
                    "id": generate_id("cb-"),
                    "character_id": char.id,
                    "kind": kind,
                    "hook": str(item["hook"])[:400],
                    "source_memory_ids": mem_ids,
                    "salience": round(sum(sals) / len(sals), 3) if sals else 0.5,
                }
            )
            created += 1

        # Enforce ready-queue cap: expire oldest beyond cap
        ready = char.storage.get_callbacks(char.id, status="ready", limit=1000)
        if len(ready) > self.cfg.callbacks_ready_cap:
            # get_callbacks orders salience DESC, created_at DESC → expire the tail
            for stale in ready[self.cfg.callbacks_ready_cap :]:
                char.storage.mark_callback(stale["id"], "expired")
        return created
