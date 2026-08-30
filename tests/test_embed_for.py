"""Tests for `MemoryStore.embed_for` and its use as the shared embedding seam
for every memory-writing subsystem that persists a vector outside
`MemoryStore.add()`/`edit()` themselves — consolidation summaries, belief
contradictions, growth events, and `FactStore.edit`'s linked-memory re-embed
(final code review fix wave, item 3).

Before this, three of those four call sites (`belief.py`, `growth.py`,
`FactStore.edit`'s fallback) embedded raw content with no date/speaker
context and (`belief.py`/`growth.py`) no dimension guard at all;
`consolidation.py` guarded but also embedded raw content. `embed_for` fixes
both: same contextualized text `add()` computes, same dimension guard.
"""

from __future__ import annotations

import pytest

from tests.helpers import FakeEmbedder, FakeLLM
from woven_imprint import clock
from woven_imprint.embedding.cache import CachedEmbedder
from woven_imprint.engine import Engine
from woven_imprint.memory.store import MemoryStore, build_embed_text
from woven_imprint.storage.sqlite import SQLiteStorage


class SpyEmbedder(CachedEmbedder):
    """Records every text it was asked to embed (see test_embedding_context.py
    for why this subclasses CachedEmbedder rather than wrapping a plain one)."""

    def __init__(self):
        super().__init__(FakeEmbedder())
        self.calls: list[str] = []

    def embed(self, text):
        self.calls.append(text)
        return super().embed(text)


def _char_with_spy():
    embedder = SpyEmbedder()
    engine = Engine(db_path=":memory:", llm=FakeLLM(), embedding=embedder)
    char = engine.create_character("Ada")
    char.parallel = False
    char.background = False
    return engine, char, embedder


# --- MemoryStore.embed_for (pure) -------------------------------------------


class TestEmbedFor:
    def test_matches_the_vector_add_would_compute(self):
        s = SQLiteStorage(":memory:")
        s.save_character("c1", "Ada", {})
        embedder = FakeEmbedder()
        store = MemoryStore(s, embedder, "c1", character_name="Ada")

        with clock.override(clock.parse_ts("2023-05-08 10:00:00")):
            created_at = clock.sqlite_ts()
            via_add = store.add(
                content="[User] I adopted a cat",
                tier="buffer",
                role="user",
                metadata={"user_id": "caroline"},
            )
            expected_text = build_embed_text(
                "[User] I adopted a cat", "user", created_at, {"user_id": "caroline"}, "Ada"
            )
            via_embed_for = store.embed_for(
                "[User] I adopted a cat", "user", created_at, {"user_id": "caroline"}
            )
        assert via_embed_for == embedder.embed(expected_text)
        assert via_add["embedding"] == via_embed_for
        s.close()

    def test_raises_on_dimension_mismatch(self):
        s = SQLiteStorage(":memory:")
        s.save_character("c1", "Ada", {})
        store = MemoryStore(s, FakeEmbedder(), "c1", character_name="Ada")
        store.add(content="seed row", tier="core", role="observation")

        class Mismatched:
            def embed(self, text):
                return [0.1, 0.2]  # 2-d vs FakeEmbedder's 50-d

            def embed_batch(self, texts):
                return [self.embed(t) for t in texts]

            def dimensions(self):
                return 2

        store.embedder = Mismatched()
        with pytest.raises(ValueError, match="[Ee]mbedding dimension mismatch"):
            store.embed_for("new content", "observation", clock.sqlite_ts(), None)
        s.close()


# --- Consolidation summaries -------------------------------------------------


class TestConsolidationUsesEmbedFor:
    """`ConsolidationEngine` supports an injected `embed_fn` (tested directly
    here), but `Character` does NOT wire it to `char.memory.embed_for` by
    default — see the deviation note next to `self.consolidator =
    ConsolidationEngine(...)` in `character.py` and CHANGELOG "Deviations":
    wiring it there measurably regressed two `eval/bench_longhorizon.py`
    checks via a timestamp-tie/rowid-ordering interaction, root-caused but
    not fixed. This test exercises the mechanism itself, not the (currently
    unwired) production default.
    """

    def test_summary_embedded_with_contextualized_date_prefixed_text_when_wired(self):
        from woven_imprint.memory.consolidation import ConsolidationEngine

        _engine, char, embedder = _char_with_spy()
        embedder.calls.clear()
        engine = ConsolidationEngine(
            char.storage,
            char.llm,
            embedder,
            char.id,
            threshold=10,
            embed_fn=char.memory.embed_for,
        )
        for i in range(15):
            char.memory.add(f"the lake was calm on day {i}", tier="buffer")
        embedder.calls.clear()

        result = engine.consolidate()

        assert result["created"] >= 1
        assert embedder.calls, "consolidation never called the (spy) embedder"
        # Contextualized (date-prefixed) — the summary text itself, not the
        # stored "[Consolidated] ..." tag (never part of any row's embedded
        # text, same as the User/character bracket tags — see build_embed_text).
        assert any(c.startswith("[20") for c in embedder.calls), embedder.calls

    def test_default_character_wiring_uses_raw_embed_not_embed_fn(self):
        """Documents the current (deviation) default: `char.consolidator`
        has no `embed_fn` wired, so summaries embed via the raw fallback."""
        _engine, char, _embedder = _char_with_spy()
        assert char.consolidator.embed_fn is None


# --- FactStore.edit -----------------------------------------------------------


class TestFactEditUsesEmbedFor:
    def test_edit_reembeds_linked_memory_with_contextualized_text(self):
        _engine, char, embedder = _char_with_spy()
        m = char.memory.add(
            "[User] The visitor lives in Tampere.",
            tier="core",
            role="user",
            metadata={"user_id": "caroline"},
        )
        f = char.facts.add(
            subject="user",
            predicate="lives_in",
            object="Tampere",
            statement=m["content"],
            memory_id=m["id"],
        )
        embedder.calls.clear()

        out = char.facts.edit(f["id"], object="Oulu", statement="[User] The visitor lives in Oulu.")

        assert out["object"] == "Oulu"
        assert embedder.calls, "edit() never called the (spy) embedder"
        last = embedder.calls[-1]
        # Contextualized: date-prefixed, speaker tag injected, user_id present,
        # bracket tag stripped — not the raw "[User] The visitor lives in Oulu."
        assert last.startswith("[20")
        assert "User: caroline:" in last
        assert last != "[User] The visitor lives in Oulu."
        assert engine_memory_content(_engine, m["id"]) == "[User] The visitor lives in Oulu."


def engine_memory_content(engine, memory_id: str) -> str:
    return engine.storage.get_memory(memory_id)["content"]


# --- Belief contradictions ----------------------------------------------------


class TestBeliefUsesEmbedFor:
    def test_contradict_embeds_contextualized_text(self):
        _engine, char, embedder = _char_with_spy()
        old = char.memory.add("Old belief.", tier="core", role="observation")
        embedder.calls.clear()

        new_mem = char.belief.contradict(old["id"], "New belief.", source="test")

        assert embedder.calls
        assert embedder.calls[-1].startswith("[20")
        assert "New belief." in embedder.calls[-1]
        assert new_mem["embedding"] is not None


# --- Growth events -------------------------------------------------------------


class TestGrowthUsesEmbedFor:
    def test_apply_growth_embeds_contextualized_text(self):
        from woven_imprint.persona.growth import GrowthEvent

        _engine, char, embedder = _char_with_spy()
        embedder.calls.clear()

        events = [
            GrowthEvent(
                trait="personality",
                old_value="shy",
                new_value="confident",
                reason="Built trust",
                confidence=0.8,
            )
        ]
        applied = char.growth.apply_growth(events, threshold=0.6)

        assert len(applied) == 1
        assert embedder.calls
        assert embedder.calls[-1].startswith("[20")
        assert "[Growth]" in embedder.calls[-1]
