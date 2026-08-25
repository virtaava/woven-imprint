"""Long-horizon benchmark: 60 simulated days through chat() with a fake clock and scripted LLM."""

from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.framework import BenchmarkResult, SuiteResult
from tests.helpers import FakeEmbedder
from woven_imprint import Engine, clock
from woven_imprint.llm.base import LLMProvider
from woven_imprint.maintenance import MaintenanceRunner

NOUNS = [
    "lighthouse",
    "violin",
    "orchard",
    "compass",
    "lantern",
    "harbor",
    "sparrow",
    "anvil",
    "meadow",
    "kettle",
    "saddle",
    "quartz",
    "willow",
    "beacon",
    "cellar",
    "thimble",
    "falcon",
    "canvas",
    "ember",
    "furnace",
]
T0 = datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc)


def noun_for(day: int) -> str:
    return f"{NOUNS[day % len(NOUNS)]}{day}"


class LongHorizonLLM(LLMProvider):
    """Scripted bookkeeping: one unique dated fact per day, trust up (days 1-40), down (41-50), up (51-60)."""

    def __init__(self):
        self.day = 1
        self.json_calls: list[str] = []
        self.gen_calls = 0

    def generate(self, messages, temperature=0.7, max_tokens=2048):
        self.gen_calls += 1
        head = messages[0]["content"].lower()
        if "summarizing a conversation session" in head:
            return f"Day {self.day}: the visitor talked about the {noun_for(self.day)}."
        if "summarize" in head or "consolidat" in head or "dense" in head:
            return "Consolidated notes about routine visits."
        return f"Ah, the {noun_for(self.day)}. I remember."

    def generate_json(self, messages, temperature=0.3):
        head = messages[0]["content"].lower()
        self.json_calls.append(head[:40])
        if "bookkeeping assistant" in head:
            trust = 0.05 if self.day <= 40 or self.day > 50 else -0.10
            tension = 0.0 if self.day <= 40 or self.day > 50 else 0.08
            facts = [f"On day {self.day} the visitor mentioned the {noun_for(self.day)}."]
            if self.day == 10:
                facts.append("The visitor likes tea.")
            if self.day == 40:
                facts.append("The visitor dislikes tea.")
            return {
                "emotion": {"mood": "content", "intensity": 0.4, "cause": "a pleasant visit"},
                "relationship": {
                    "trust": trust,
                    "affection": 0.02,
                    "respect": 0.0,
                    "familiarity": 0.02,
                    "tension": tension,
                },
                "beat": None,
                "facts": facts,
            }
        if "importance" in head or "score" in head:
            n = messages[1]["content"].count("\n") + 1
            return [6] * n
        if "contradict" in head:
            return {"contradictory": False, "current": "unclear"}
        if "conversation hooks" in head:
            return []
        if "personality" in head or "growth" in head:
            return []
        return {}

    def generate_json_robust(self, messages, temperature=0.3):
        return self.generate_json(messages, temperature)


def _simulate(days: int):
    llm = LongHorizonLLM()
    engine = Engine(db_path=":memory:", llm=llm, embedding=FakeEmbedder())
    trust_at: dict[int, float] = {}
    calls_per_turn: list[int] = []
    with clock.override(T0):
        # Character creation (and its bedrock-memory seeding) must happen
        # inside the clock override — otherwise seeded memories are stamped
        # with the real wall-clock "now" instead of T0, which then makes
        # them look like the newest thing in the store (they sort ahead of
        # every simulated day) and skews recency-sensitive retrieval.
        char = engine.create_character("Meridian", persona={"personality": "patient"})
        char.parallel = False
        char.background = False
        char.enforce_consistency = False
        char.unified_assessment = True
        runner = MaintenanceRunner(char)
        for day in range(1, days + 1):
            llm.day = day
            char.start_session()
            for turn in range(3):
                before = len(llm.json_calls)
                char.chat(
                    f"Today I want to talk about the {noun_for(day)}, turn {turn}.", user_id="toni"
                )
                calls_per_turn.append(len(llm.json_calls) - before)
            char.end_session()
            trust_at[day] = char.relationships.get("toni")["dimensions"]["trust"]
            runner.run()
            clock.advance(timedelta(days=1))
        results = _score(engine, char, llm, trust_at, calls_per_turn, days)
    engine.close()
    return results


def _score(engine, char, llm, trust_at, calls_per_turn, days) -> list[BenchmarkResult]:
    out: list[BenchmarkResult] = []
    # 1 paraphrase recall of day 5
    hits = char.retriever.retrieve(f"{noun_for(5)} mentioned visitor", limit=10)
    found = any("day 5 " in m["content"].lower() and noun_for(5) in m["content"] for m in hits)
    out.append(
        BenchmarkResult(
            "paraphrase_recall_day5",
            found,
            1.0 if found else 0.0,
            {"top": [m["content"][:60] for m in hits[:3]]},
        )
    )
    # 2 dates rendered
    text = char._format_memories(hits)
    msgs = char._build_context("hello", hits, "")
    volatile = " ".join(m["content"] for m in msgs if m["role"] == "system")
    dated = (
        ("2026-01-" in text)
        and ("weeks ago" in text or "months ago" in text)
        and ("Today is" in volatile)
    )
    out.append(
        BenchmarkResult("dates_rendered", dated, 1.0 if dated else 0.0, {"sample": text[:200]})
    )
    # 3 recency ordering: newest day's fact ranks above day 2's for the same noun family
    ranked = char.retriever.retrieve("the visitor mentioned", limit=50)
    pos = {m["content"]: i for i, m in enumerate(ranked)}
    new = next((k for k in pos if f"day {days} " in k.lower()), None)
    old = next((k for k in pos if "day 2 " in k.lower()), None)
    ok3 = new is not None and (old is None or pos[new] < pos[old])
    out.append(
        BenchmarkResult("recency_ordering", ok3, 1.0 if ok3 else 0.0, {"new": new, "old": old})
    )
    # 4 contradiction supersession
    tea = char.retriever.retrieve("tea", limit=5)
    rows = {
        m["content"]: m
        for m in engine.storage.get_memories(char.id, status="contradicted", limit=None)
    }
    superseded = any("likes tea" in c for c in rows)
    first_is_new = bool(tea) and "dislikes tea" in tea[0]["content"]
    ok4 = superseded and first_is_new
    out.append(
        BenchmarkResult(
            "contradiction_supersession",
            ok4,
            1.0 if ok4 else 0.0,
            {"superseded": superseded, "top": tea[0]["content"] if tea else None},
        )
    )
    # 5 relationship trajectory
    ok5 = trust_at[40] > trust_at[1] and trust_at[50] < trust_at[40]
    fam = char.relationships.get("toni")["dimensions"]["familiarity"]
    out.append(
        BenchmarkResult(
            "relationship_trajectory",
            ok5 and fam > 0,
            1.0 if ok5 else 0.0,
            {"t1": trust_at[1], "t40": trust_at[40], "t50": trust_at[50], "fam": fam},
        )
    )
    # 6 session summaries dated
    sums = [
        m
        for m in char.memory.get_all(tier="core", limit=5000)
        if m["content"].startswith("[Session Summary")
    ]
    ok6 = bool(sums) and all(
        m["metadata"].get("started_at") and m["content"].startswith("[Session Summary 2026-")
        for m in sums
    )
    out.append(
        BenchmarkResult("session_summaries_dated", ok6, 1.0 if ok6 else 0.0, {"count": len(sums)})
    )
    # 7 one bookkeeping call per turn
    ok7 = all(c == 1 for c in calls_per_turn)
    out.append(
        BenchmarkResult(
            "bookkeeping_call_count",
            ok7,
            1.0 if ok7 else 0.0,
            {"max": max(calls_per_turn), "min": min(calls_per_turn)},
        )
    )
    return out


def run_longhorizon_suite(days: int = 60) -> SuiteResult:
    start = time.time()
    suite = SuiteResult(suite_name="Long Horizon (60 simulated days)")
    suite.results.extend(_simulate(days))
    suite.total_duration_ms = (time.time() - start) * 1000
    return suite


if __name__ == "__main__":
    print(run_longhorizon_suite().summary())
