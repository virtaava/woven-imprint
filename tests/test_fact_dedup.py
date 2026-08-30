"""Tests for semantic dedup of fact-derived core memories (Tier 3d Task 3).

Evidence (Tier 3c diagnostics, 2026-08-30): near-duplicate extracted facts dominated
the retrieved top-20 (17.7/20 average); 78/491 active core rows in one LoCoMo
conversation fell into 6-word-prefix paraphrase groups (e.g. "considering a career
in counseling and mental health" vs "...or mental health work"). `MemoryStore.add`
now accepts `dedup_similarity`/`dedup_scope`: when a fact-derived core insert's
embedding cosine-matches an existing active row of the same tier at or above the
threshold, no new row is written — the existing row is reinforced (importance
bumped, `metadata.dup_count` incremented) and returned instead, with a transient
`"deduped"` key telling the caller which happened.

`FakeEmbedder` (tests/helpers.py) is bag-of-words over the first 10 words of the
*embedded* text — with `memory.embedding_context` on (the default), that's the
date+speaker contextualized string (`build_embed_text`), not raw `content`. Two
identical fact statements recorded on the same day embed identically (cosine
1.0); clearly different wording shares no vocabulary and scores 0.0.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from eval.external.common import Conversation, Session, Turn
from eval.external.runner import RunConfig, ingest_conversation
from tests.helpers import FakeEmbedder, FakeLLM, make_test_engine
from woven_imprint import clock
from woven_imprint.engine import Engine
from woven_imprint.memory.store import MemoryStore
from woven_imprint.storage.sqlite import SQLiteStorage


@pytest.fixture(autouse=True)
def _reset_clock():
    """`ingest_conversation` drives `clock.override()` as a plain (non-restoring)
    call, by design (see tests/test_external_runner.py) — reset it after each
    test here so the harness-stat test below doesn't leak a stale override
    into later tests."""
    yield
    clock.override(None)


# --- MemoryStore.add(dedup_similarity=...) — unit level ------------------------------------


@pytest.fixture
def store():
    s = SQLiteStorage(":memory:")
    s.save_character("c1", "Ada", {})
    yield MemoryStore(s, FakeEmbedder(), "c1"), s
    s.close()


class TestMemoryStoreDedup:
    def test_identical_statement_dedupes_to_one_core_row(self, store):
        mem, _s = store
        first = mem.add(
            content="The visitor is considering a career in counseling and mental health.",
            tier="core",
            role="observation",
            importance=0.75,
            metadata={"source": "extraction"},
            dedup_similarity=0.92,
        )
        assert first["deduped"] is False

        second = mem.add(
            content="The visitor is considering a career in counseling and mental health.",
            tier="core",
            role="observation",
            importance=0.75,
            metadata={"source": "extraction"},
            dedup_similarity=0.92,
        )

        assert second["deduped"] is True
        assert second["id"] == first["id"]
        assert second["metadata"]["dup_count"] == 1
        assert second["metadata"]["last_confirmed"]
        assert second["importance"] == pytest.approx(0.8)  # 0.75 + 0.05

        rows = mem.get_all(tier="core")
        assert len(rows) == 1

    def test_paraphrase_below_threshold_creates_two_rows(self, store):
        mem, _s = store
        mem.add(
            content="alpha bravo charlie delta echo",
            tier="core",
            role="observation",
            dedup_similarity=0.92,
        )
        result = mem.add(
            content="foxtrot golf hotel india juliet",
            tier="core",
            role="observation",
            dedup_similarity=0.92,
        )

        assert result["deduped"] is False
        assert len(mem.get_all(tier="core")) == 2

    def test_near_threshold_gated_by_configured_similarity(self, store):
        """Genuine ~0.8-cosine near-duplicate: the first 10 embed-text tokens
        (the shared date token + 7 leading shared words) overlap 8/10 between
        the two, 2 differ on each side — computed below via the embedder
        rather than assumed. Must NOT dedupe at the 0.92 default but MUST
        dedupe once the (config) threshold is lowered to 0.75."""
        from woven_imprint.config import get_config

        mem, _s = store
        content1 = "alpha bravo charlie delta echo foxtrot golf hotel india"
        content2 = "alpha bravo charlie delta echo foxtrot golf juliet kilo"

        with clock.override(T0):
            created_at = clock.sqlite_ts()
            text1 = MemoryStore.build_embed_text(content1, "observation", created_at, None, "Ada")
            text2 = MemoryStore.build_embed_text(content2, "observation", created_at, None, "Ada")
            v1 = mem.embedder.embed(text1)
            v2 = mem.embedder.embed(text2)
            sim = sum(a * b for a, b in zip(v1, v2))
            assert 0.7 < sim < 0.9  # genuinely near-threshold, not 0.0 or 1.0

            cfg = get_config()
            original = cfg.memory.fact_dedup_similarity
            try:
                mem.add(
                    content=content1,
                    tier="core",
                    role="observation",
                    dedup_similarity=cfg.memory.fact_dedup_similarity,
                )
                not_deduped = mem.add(
                    content=content2,
                    tier="core",
                    role="observation",
                    dedup_similarity=cfg.memory.fact_dedup_similarity,
                )
                assert not_deduped["deduped"] is False
                assert len(mem.get_all(tier="core")) == 2

                cfg.memory.fact_dedup_similarity = 0.75
                deduped = mem.add(
                    content=content2,
                    tier="core",
                    role="observation",
                    dedup_similarity=cfg.memory.fact_dedup_similarity,
                )
                assert deduped["deduped"] is True
                assert len(mem.get_all(tier="core")) == 2
            finally:
                cfg.memory.fact_dedup_similarity = original

    def test_threshold_zero_disables_dedup(self, store):
        mem, _s = store
        mem.add(
            content="identical wording every time",
            tier="core",
            role="observation",
            dedup_similarity=0,
        )
        second = mem.add(
            content="identical wording every time",
            tier="core",
            role="observation",
            dedup_similarity=0,
        )

        assert second["deduped"] is False
        assert len(mem.get_all(tier="core")) == 2

    def test_dedup_default_off_when_not_requested(self, store):
        mem, _s = store
        mem.add(content="identical wording every time", tier="core", role="observation")
        second = mem.add(content="identical wording every time", tier="core", role="observation")

        assert second["deduped"] is False
        assert len(mem.get_all(tier="core")) == 2

    def test_dedup_never_touches_buffer_tier(self, store):
        mem, _s = store
        mem.add(
            content="identical wording every time",
            tier="buffer",
            role="user",
            dedup_similarity=0.92,
        )
        second = mem.add(
            content="identical wording every time",
            tier="buffer",
            role="user",
            dedup_similarity=0.92,
        )

        assert second["deduped"] is False
        assert len(mem.get_all(tier="buffer")) == 2

    def test_require_tokens_guard_rejects_match_missing_object_token(self, store):
        """`dedup_require_tokens` (entity-delta guard): a shared 10-word preamble
        makes the FakeEmbedder's bag-of-words vector identical (the differing
        word lands past the embedder's first-10-word window, so it never
        enters the vector at all — cosine is exactly 1.0, comfortably >= the
        0.92 threshold) even though the two statements describe a different
        color. Without the token guard this would wrongly dedupe."""
        mem, _s = store
        prefix = "caroline told me a long story yesterday about her all time favorite color which turns out to be"
        mem.add(
            content=f"{prefix} blue",
            tier="core",
            role="observation",
            dedup_similarity=0.92,
            dedup_require_tokens=["blue"],
        )
        result = mem.add(
            content=f"{prefix} red",
            tier="core",
            role="observation",
            dedup_similarity=0.92,
            dedup_require_tokens=["red"],
        )

        assert result["deduped"] is False
        assert len(mem.get_all(tier="core")) == 2

    def test_require_tokens_guard_allows_match_when_object_restated(self, store):
        """Same setup as above, but the object word restated identically —
        the required token IS present in the candidate's content, so the
        guard doesn't block the (otherwise legitimate) dedup."""
        mem, _s = store
        prefix = "caroline told me a long story yesterday about her all time favorite color which turns out to be"
        mem.add(
            content=f"{prefix} blue",
            tier="core",
            role="observation",
            dedup_similarity=0.92,
            dedup_require_tokens=["blue"],
        )
        result = mem.add(
            content=f"{prefix} blue",
            tier="core",
            role="observation",
            dedup_similarity=0.92,
            dedup_require_tokens=["blue"],
        )

        assert result["deduped"] is True
        assert len(mem.get_all(tier="core")) == 1

    def test_require_tokens_none_does_not_change_behavior(self, store):
        """Omitting `dedup_require_tokens` (the default) leaves plain
        cosine-threshold dedup behavior byte-for-byte unchanged."""
        mem, _s = store
        first = mem.add(
            content="identical wording every time",
            tier="core",
            role="observation",
            dedup_similarity=0.92,
        )
        second = mem.add(
            content="identical wording every time",
            tier="core",
            role="observation",
            dedup_similarity=0.92,
        )

        assert second["deduped"] is True
        assert second["id"] == first["id"]

    def test_dedup_never_touches_bedrock_tier(self, store):
        mem, _s = store
        mem.add(
            content="identical wording every time",
            tier="bedrock",
            role=None,
            dedup_similarity=0.92,
        )
        second = mem.add(
            content="identical wording every time",
            tier="bedrock",
            role=None,
            dedup_similarity=0.92,
        )

        assert second["deduped"] is False
        assert len(mem.get_all(tier="bedrock")) == 2


# --- Wired through the fact pipeline (character.py) ------------------------------------------


def _char(llm=None):
    engine = make_test_engine()
    if llm is not None:
        engine.llm = llm
    char = engine.create_character("Ada")
    char.background = False
    char.parallel = False
    char.enforce_consistency = False
    char.unified_assessment = True
    return engine, char


def _fact_rows(char) -> list[dict]:
    """Core memories that came from fact extraction (`metadata.source ==
    "extraction"`) — excludes session-summary core rows `end_session()` also
    writes, which aren't part of what this feature dedupes."""
    return [
        m
        for m in char.memory.get_all(tier="core")
        if (m.get("metadata") or {}).get("source") == "extraction"
    ]


class TestFactPipelineDedupUnstructured:
    def test_identical_unstructured_fact_across_sessions_dedupes(self):
        """`FakeLLM`'s default bookkeeping response always extracts the same
        unstructured fact statement ("A notable fact was shared", no subject/
        predicate/object) — two sessions ingesting it produce one core row."""
        _engine, char = _char()

        char.start_session()
        char.ingest("user", "Something happened today.", user_id="toni")
        char.end_session()

        char.start_session()
        char.ingest("user", "Something happened today.", user_id="toni")
        char.end_session()

        core = _fact_rows(char)
        assert len(core) == 1
        assert core[0]["metadata"]["dup_count"] == 1
        assert char.health()["dedup_skipped"] == 1

    def test_threshold_zero_keeps_both_unstructured_facts(self):
        from woven_imprint.config import get_config

        cfg = get_config()
        original = cfg.memory.fact_dedup_similarity
        cfg.memory.fact_dedup_similarity = 0
        try:
            _engine, char = _char()
            char.start_session()
            char.ingest("user", "Something happened today.", user_id="toni")
            char.end_session()
            char.start_session()
            char.ingest("user", "Something happened today.", user_id="toni")
            char.end_session()

            core = _fact_rows(char)
            assert len(core) == 2
            assert char.health()["dedup_skipped"] == 0
        finally:
            cfg.memory.fact_dedup_similarity = original


class DuplicateStatementDifferentSubjectLLM(FakeLLM):
    """Two 'bookkeeping assistant' calls return structured facts with the SAME
    rendered statement text but a DIFFERENT (subject, predicate, object) each
    time — modeling extraction drift where the same underlying fact gets
    tagged differently. `FactStore`'s own (subject, predicate) supersession
    only fires on an exact key match, so this never reaches it — it's the
    shape `MemoryStore.add`'s semantic dedup exists to catch instead.
    """

    STATEMENT = "The visitor is considering a career in counseling and mental health."

    def __init__(self):
        super().__init__()
        self._n = 0

    def generate_json(self, messages, **kw):
        system = messages[0].get("content", "") if messages else ""
        if "bookkeeping assistant" in system.lower():
            self._n += 1
            self.call_count += 1
            subject = "user" if self._n == 1 else "the_visitor"
            return {
                "emotion": {"mood": "neutral", "intensity": 0.5, "cause": ""},
                "relationship": {"trust": 0.0},
                "beat": None,
                "facts": [
                    {
                        "statement": self.STATEMENT,
                        "subject": subject,
                        "predicate": "considering_career_in",
                        "object": "counseling and mental health",
                        "event_time": None,
                    }
                ],
            }
        return super().generate_json(messages, **kw)

    def generate_json_robust(self, messages, temperature=0.3, **kw):
        return self.generate_json(messages, temperature=temperature, **kw)


class TestFactPipelineDedupStructured:
    def test_near_duplicate_structured_facts_share_one_core_row(self):
        llm = DuplicateStatementDifferentSubjectLLM()
        _engine, char = _char(llm)

        char.start_session()
        char.ingest("user", "I might become a counselor.", user_id="toni")
        char.end_session()

        char.start_session()
        char.ingest("user", "I'm drawn to counseling and mental health work.", user_id="toni")
        char.end_session()

        core = _fact_rows(char)
        assert len(core) == 1
        assert core[0]["metadata"]["dup_count"] == 1

        # The structured `facts` table still gets both rows — its own
        # (subject, predicate) supersession is a different key per call here,
        # so it never fires; dedup only collapsed the memory side.
        assert char.facts.count() == 2
        f1 = char.facts.find_active("user", "considering_career_in")
        f2 = char.facts.find_active("the_visitor", "considering_career_in")
        assert f1 is not None and f2 is not None
        assert f1["memory_id"] == core[0]["id"]
        assert f2["memory_id"] == core[0]["id"]

    def test_dedup_match_equal_to_contradicted_old_memory_gets_fresh_row(self):
        """Same (subject, predicate), different object, near-identical statement
        text: the first 10 words are identical, the object word lands after
        position 10 so the bag-of-words embedding is identical (cosine 1.0) —
        the new fact's dedup match is exactly the old fact's memory, the very
        row (subject, predicate) supersession is about to mark `contradicted`.
        The new fact must NOT end up linked to a contradicted memory."""
        _engine, char = _char()
        prefix = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"

        with clock.override(T0):
            char._store_structured_fact(
                {
                    "statement": f"{prefix} Tampere",
                    "subject": "user",
                    "predicate": "lives_in",
                    "object": "Tampere",
                    "event_time": None,
                },
                "toni",
                None,
                0.75,
            )
            old = char.facts.find_active("user", "lives_in")
            assert old is not None
            old_mem_id = old["memory_id"]

            char._store_structured_fact(
                {
                    "statement": f"{prefix} Oulu",
                    "subject": "user",
                    "predicate": "lives_in",
                    "object": "Oulu",
                    "event_time": None,
                },
                "toni",
                None,
                0.75,
            )

        new = char.facts.find_active("user", "lives_in")
        assert new is not None and new["id"] != old["id"] and new["object"] == "Oulu"
        assert new["memory_id"] != old_mem_id

        old_mem = _engine.storage.get_memory(old_mem_id)
        assert old_mem["status"] == "contradicted"

        new_mem = _engine.storage.get_memory(new["memory_id"])
        assert new_mem is not None and new_mem["status"] == "active"

    def test_unrelated_facts_with_shared_preamble_dont_cross_link(self):
        """Two entirely unrelated structured facts (different subject/predicate,
        so `old` is None both times — no contradiction bookkeeping involved)
        share a 10-word statement preamble, differing only in the object word
        placed past the FakeEmbedder's window — bag-of-words cosine is exactly
        1.0. Without the entity-delta guard the second fact's dedup lookup
        would wrongly match the first fact's memory and link to it instead of
        getting its own row."""
        _engine, char = _char()
        prefix = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"

        with clock.override(T0):
            char._store_structured_fact(
                {
                    "statement": f"{prefix} blue",
                    "subject": "user",
                    "predicate": "favorite_color",
                    "object": "blue",
                    "event_time": None,
                },
                "toni",
                None,
                0.75,
            )
            char._store_structured_fact(
                {
                    "statement": f"{prefix} pizza",
                    "subject": "user",
                    "predicate": "favorite_food",
                    "object": "pizza",
                    "event_time": None,
                },
                "toni",
                None,
                0.75,
            )

        color = char.facts.find_active("user", "favorite_color")
        food = char.facts.find_active("user", "favorite_food")
        assert color is not None and food is not None
        assert color["memory_id"] != food["memory_id"]
        assert len(_fact_rows(char)) == 2

    def test_supersession_leaves_shared_memory_active_when_other_fact_still_references_it(self):
        """When the OLD fact's memory is shared with another still-active fact
        (semantic dedup — same identical statement — links a second, unrelated
        (subject, predicate) fact to the same core memory row), superseding the
        old fact must NOT mark that shared memory contradicted: the other fact
        still depends on it reading as active."""
        _engine, char = _char()

        with clock.override(T0):
            char._store_structured_fact(
                {
                    "statement": "The visitor enjoys board games.",
                    "subject": "user",
                    "predicate": "plays",
                    "object": "chess",
                    "event_time": None,
                },
                "toni",
                None,
                0.75,
            )
            plays = char.facts.find_active("user", "plays")
            shared_memory_id = plays["memory_id"]

            char._store_structured_fact(
                {
                    "statement": "The visitor enjoys board games.",
                    "subject": "user",
                    "predicate": "enjoys",
                    "object": "board games",
                    "event_time": None,
                },
                "toni",
                None,
                0.75,
            )
            enjoys = char.facts.find_active("user", "enjoys")
            assert enjoys is not None and enjoys["memory_id"] == shared_memory_id  # dedup confirmed

            char._store_structured_fact(
                {
                    "statement": "The visitor now plays shogi instead.",
                    "subject": "user",
                    "predicate": "plays",
                    "object": "shogi",
                    "event_time": None,
                },
                "toni",
                None,
                0.75,
            )

        old_plays = char.facts.get(plays["id"])
        assert old_plays is not None and old_plays["valid_to"] is not None  # expired

        shared_mem = _engine.storage.get_memory(shared_memory_id)
        assert shared_mem["status"] == "active"  # NOT contradicted — `enjoys` still needs it
        assert char.facts.get(enjoys["id"])["memory_id"] == shared_memory_id  # untouched

        # Control: once the sharing fact is retracted too, a THIRD supersession
        # of the same (subject, predicate) key is free to mark it contradicted
        # (nothing else references it any more). Not exercised here — covered
        # by test_facts_edit.py's shared-row tests and FactStore.retract's own
        # existing guard.


# --- Harness ingest stat -----------------------------------------------------------------------

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def _dedup_conversation() -> Conversation:
    turns = [
        Turn(speaker="Toni", role="user", text="I might become a counselor.", dia_id="d1", at=T0),
        Turn(speaker="Ada", role="assistant", text="That sounds meaningful.", dia_id="d2", at=T0),
        Turn(
            speaker="Toni",
            role="user",
            text="I'm considering counseling and mental health work.",
            dia_id="d3",
            at=T0,
        ),
        Turn(speaker="Ada", role="assistant", text="Tell me more.", dia_id="d4", at=T0),
    ]
    session = Session(session_id="s1", at=T0, turns=turns)
    return Conversation(
        conv_id="conv-dedup-1",
        user_name="Toni",
        character_name="Ada",
        sessions=[session],
        questions=[],
        transcript_text="",
    )


class TestHarnessDedupStat:
    def test_ingest_conversation_records_dedup_skipped(self):
        """`fact_extraction_interval` defaults to 3: 4 turns means bookkeeping's
        `want_facts` is true on turn_count 0 and 3 — two extractions of the
        same fixed `FakeLLM` fact statement, on the same simulated day, so the
        second dedupes into the first."""
        engine = Engine(db_path=":memory:", llm=FakeLLM(), embedding=FakeEmbedder())
        conv = _dedup_conversation()
        cfg = RunConfig(bench="locomo", mode="memory", run_id="dedup-test")

        stats = ingest_conversation(conv, engine, cfg)

        assert "dedup_skipped" in stats
        assert stats["dedup_skipped"] == 1

        char = engine.get_character(conv.conv_id)
        core = _fact_rows(char)
        assert len(core) == 1
        assert core[0]["metadata"]["dup_count"] == 1
