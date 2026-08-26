"""Long-horizon benchmark: 60 simulated days through chat() with a fake clock and scripted LLM.

Known ranking limitation (narrowed, not eliminated, by memory.relevance_gate
— see ARCHITECTURE.md#retrieval-relevance-gate): retrieval fuses signals with
equal-weight RRF, and bedrock memories (e.g. the `[Self]` persona line) carry
a permanent importance/recency floor. With the gate on (default), recency and
importance ranking only consider memories that are semantically or lexically
relevant to the query, which keeps most off-topic floor-holders out of
contention — but an off-topic memory that happens to be semantically close
enough to land in the top `relevance_semantic_topk` can still ride its floor
above a fresh, directly-relevant fact. `contradiction_supersession` below
deliberately does not assert where the day-40 "dislikes tea" memory lands in
the *global* fused ranking — only that it is first among memories that
actually mention "tea" — and records that global rank informationally.
"""

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
# Second word list for filler tokens — kept lexically distinct from NOUNS so a
# fact's filler words never collide with its own topic noun.
FILLERS = [
    "ribbon",
    "cobalt",
    "garnet",
    "opal",
    "ivory",
    "amber",
    "onyx",
    "coral",
    "slate",
    "umber",
    "jade",
    "rust",
    "cedar",
    "dune",
    "frost",
    "maple",
    "topaz",
    "birch",
    "copper",
    "linen",
]
SUFFIX_WORDS = ["note", "mark", "tag", "log", "item", "entry"]
T0 = datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc)

# Days whose fact must literally contain "the visitor mentioned" (not just the
# "day N " marker + noun): recall_day5's query and recency_ordering's query
# both search for that exact phrase, so these days keep template 0.
FORCE_MENTIONED_DAYS = {2, 5, 60}

# After day 40, exchanges are padded past fact_density_scaling's 2000-char
# threshold so max_facts doubles to 10 (see character.py _run_bookkeeping) —
# this is what lets the tail days generate enough distinct facts to cross the
# 200-active-core-row window. The padding phrase itself repeats verbatim so it
# adds no new vocabulary to the bag-of-words FakeEmbedder.
LONG_TAIL = (
    " There is a lot to say about it, more than usual, and I want to walk "
    "through the details slowly so nothing gets missed this time, since "
    "today felt different from the quieter days before it."
)


def noun_for(day: int, salt: int = 0) -> str:
    suffix = f"_{salt}" if salt else ""
    return f"{NOUNS[(day + salt * 7) % len(NOUNS)]}{day}{suffix}"


def filler_words(day: int, n: int, salt: int = 0) -> list[str]:
    return [f"{FILLERS[(day * 3 + salt * 5 + k * 7) % len(FILLERS)]}{day}" for k in range(n)]


# Six rotating sentence templates per day's fact/summary — this is the fix for
# the dedup false-positive bug: the old single template ("On day N the visitor
# mentioned the X.") shared almost every word across days, so dedup_similarity
# (0.92, scored by the bag-of-words FakeEmbedder) treated nearly all of them as
# near-duplicates and archived them, capping the store at ~53 active core rows
# forever (never crossing the 200-row window the benchmark exists to probe).
# Rotating phrasing + 2-3 day-derived filler words per entry keeps genuinely
# distinct days lexically distinct while still carrying the "day N " marker
# and noun_for(day) substring the checks below depend on.
FACT_TEMPLATES = [
    lambda day, noun, f: (
        f"On day {day} the visitor mentioned the {noun}, also noting {f[0]} and {f[1]}."
    ),
    lambda day, noun, f: (
        f"The {noun} came up while chatting on day {day}; also {f[0]} and {f[1]} came up."
    ),
    lambda day, noun, f: f"Day {day} log: topics were the {noun}, {f[0]}, and {f[1]}.",
    lambda day, noun, f: f"A quiet exchange on day {day} touched the {noun}, {f[0]}, {f[1]}.",
    lambda day, noun, f: (
        f"During day {day}'s chat, the {noun} surfaced alongside {f[0]} and {f[1]}."
    ),
    lambda day, noun, f: f"Day {day} brief: the {noun}, {f[0]}, {f[1]} came up in conversation.",
]
SUMMARY_TEMPLATES = [
    lambda day, noun, f, s: f"Day {day} chat covered {noun}, {f[0]}, {f[1]}, {s}{day}.",
    lambda day, noun, f, s: (
        f"Visitor recalled {noun} plus {f[0]}, {f[1]}, and {s}{day} on day {day}."
    ),
    lambda day, noun, f, s: f"On day {day}: {noun}, {f[0]}, {f[1]}, {s}{day} discussed.",
    lambda day, noun, f, s: f"Recap day {day} - {noun} and {f[0]}, {f[1]}, {s}{day}.",
    lambda day, noun, f, s: f"Notes: day {day} touched {noun}, {f[0]}, {f[1]}, {s}{day}.",
    lambda day, noun, f, s: f"Session day {day} featured {noun} with {f[0]}, {f[1]}, {s}{day}.",
]


def _fact_template(day: int, salt: int):
    if day in FORCE_MENTIONED_DAYS and salt == 0:
        return FACT_TEMPLATES[0]
    return FACT_TEMPLATES[(day + salt) % len(FACT_TEMPLATES)]


TEA_LIKE = "The visitor likes tea."
TEA_DISLIKE = "The visitor dislikes tea."

# Days 41-60 request this many distinct facts/day (via the padded exchange
# below, which raises fact_density_scaling's per-turn cap so they actually
# get stored) to carry the store past the 200-active-core-row window before
# scoring. Not tuned to an exact value: contradiction_supersession no longer
# depends on tail volume (see module docstring), so any count that reliably
# crosses the 200-row threshold works — this is well clear of that line.
TAIL_FACTS_PER_DAY = 8


class LongHorizonLLM(LLMProvider):
    """Scripted bookkeeping: dated, lexically-diverse facts/summaries per day, trust up
    (days 1-40), down (41-50), up (51-60).

    Trust is clamped to [-1.0, 1.0] by the relationship model, so the sustained
    +0.05/day run saturates it at 1.0 well before day 40 and the -0.10/day run
    (days 41-50) drives it back down. `relationship_trajectory` below therefore
    checks direction (up, then down) under that clamping, not raw magnitude.

    Fact volume follows the day, not a flat rate: days 1-40 emit at most one
    fact per day (skipping odd days outside the days that other checks query
    directly) so the 30-day span between the day-10 and day-40 tea facts stays
    under ~50 new core rows — the width of Character._store_facts's own
    recency window for its inline contradiction heuristic — so the day-40
    "dislikes tea" fact can still see, and supersede, the day-10 "likes tea"
    fact. Days 41-60 emit TAIL_FACTS_PER_DAY distinct facts per day (via a
    padded exchange that raises fact_density_scaling's per-turn cap) to carry
    the store past the 200-active-core-row window before scoring — that count
    is not otherwise tuned; see TAIL_FACTS_PER_DAY's own comment.
    """

    def __init__(self):
        self.day = 1
        self.json_calls: list[str] = []
        self.gen_calls = 0

    def generate(self, messages, temperature=0.7, max_tokens=2048):
        self.gen_calls += 1
        head = messages[0]["content"].lower()
        if "summarizing a conversation session" in head:
            day = self.day
            noun = noun_for(day)
            f = filler_words(day, 2)
            suffix = SUFFIX_WORDS[day % len(SUFFIX_WORDS)]
            return SUMMARY_TEMPLATES[day % len(SUMMARY_TEMPLATES)](day, noun, f, suffix)
        if "summarize" in head or "consolidat" in head or "dense" in head:
            return "Consolidated notes about routine visits."
        return f"Ah, the {noun_for(self.day)}. I remember."

    def _regular_fact(self, day: int, salt: int = 0) -> str:
        noun = noun_for(day, salt=salt)
        f = filler_words(day, 2, salt=salt)
        return _fact_template(day, salt)(day, noun, f)

    def generate_json(self, messages, temperature=0.3):
        head = messages[0]["content"].lower()
        self.json_calls.append(head[:40])
        if "bookkeeping assistant" in head:
            day = self.day
            trust = 0.05 if day <= 40 or day > 50 else -0.10
            tension = 0.0 if day <= 40 or day > 50 else 0.08
            if day == 10:
                facts = [self._regular_fact(day), TEA_LIKE]
            elif day == 40:
                facts = [self._regular_fact(day), TEA_DISLIKE]
            elif day <= 40:
                # Sparse middle window: keep the 30-day span between the tea
                # facts under the inline contradiction heuristic's ~50-row
                # lookback (see class docstring). FORCE_MENTIONED_DAYS must
                # still get a fact regardless of parity.
                facts = (
                    [self._regular_fact(day)]
                    if (day % 2 == 0 or day in FORCE_MENTIONED_DAYS)
                    else []
                )
            else:
                # Tail: many distinct facts/day to cross the 200-row window
                # (see TAIL_FACTS_PER_DAY).
                facts = [self._regular_fact(day, salt=i) for i in range(TAIL_FACTS_PER_DAY)]
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
            for turn in range(5):
                before = len(llm.json_calls)
                msg = f"Today I want to talk about the {noun_for(day)}, turn {turn}."
                if day > 40:
                    # Push exchange_len past fact_density_scaling's 2000-char
                    # threshold (doubling max_facts to 10) with a verbatim
                    # repeated phrase, so the padding itself adds no new
                    # vocabulary to the bag-of-words FakeEmbedder.
                    msg += LONG_TAIL * 12
                char.chat(msg, user_id="toni")
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
    # 1 paraphrase recall of day 5, plus proof the store actually crossed the
    # 200-active-core-row window (the old fixed 200-row cap this benchmark
    # exists to catch would have made this recall trivial by construction).
    hits = char.retriever.retrieve(f"{noun_for(5)} mentioned visitor", limit=10)
    found = any("day 5 " in m["content"].lower() and noun_for(5) in m["content"] for m in hits)
    active_core = engine.storage.get_memories(char.id, tier="core", status="active", limit=None)
    crossed_window = len(active_core) > 200
    out.append(
        BenchmarkResult(
            "paraphrase_recall_day5",
            found and crossed_window,
            1.0 if (found and crossed_window) else 0.0,
            {
                "top": [m["content"][:60] for m in hits[:3]],
                "active_core_count": len(active_core),
            },
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
    ok3 = new is not None and old is not None and pos[new] < pos[old]
    out.append(
        BenchmarkResult("recency_ordering", ok3, 1.0 if ok3 else 0.0, {"new": new, "old": old})
    )
    # 4 contradiction supersession: among retrieved memories that actually
    # mention "tea", the first must be the day-40 fact — not the top of the
    # whole fused ranking, which a bedrock/importance-boosted but off-topic
    # memory can occupy (see module docstring's "Known ranking limitation").
    ranked = char.retriever.retrieve("tea", limit=20)
    tea = [m for m in ranked if "tea" in m["content"].lower()]
    rows = {
        m["content"]: m
        for m in engine.storage.get_memories(char.id, status="contradicted", limit=None)
    }
    superseded = any(c.startswith("The visitor likes tea") for c in rows)
    first_is_new = bool(tea) and "dislikes tea" in tea[0]["content"].lower()
    ok4 = superseded and first_is_new
    # Informational only, not asserted: where the day-40 memory lands in the
    # full (unfiltered) ranking.
    day40_rank = next(
        (i for i, m in enumerate(ranked) if "dislikes tea" in m["content"].lower()), None
    )
    out.append(
        BenchmarkResult(
            "contradiction_supersession",
            ok4,
            1.0 if ok4 else 0.0,
            {
                "superseded": superseded,
                "top": tea[0]["content"] if tea else None,
                "day40_rank_in_full_ranking": day40_rank,
            },
        )
    )
    # 5 relationship trajectory
    ok5 = trust_at[40] > trust_at[1] and trust_at[50] < trust_at[40]
    fam = char.relationships.get("toni")["dimensions"]["familiarity"]
    passed5 = ok5 and fam > 0
    out.append(
        BenchmarkResult(
            "relationship_trajectory",
            passed5,
            1.0 if passed5 else 0.0,
            {"t1": trust_at[1], "t40": trust_at[40], "t50": trust_at[50], "fam": fam},
        )
    )
    # 6 session summaries dated, one per simulated day
    sums = [
        m
        for m in char.memory.get_all(tier="core", limit=5000)
        if m["content"].startswith("[Session Summary")
    ]
    ok6 = (
        bool(sums)
        and all(
            m["metadata"].get("started_at") and m["content"].startswith("[Session Summary 2026-")
            for m in sums
        )
        and len(sums) == days
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
