"""Tests for contextualized embedding text (Tier 3d Task 2: date + speaker context,
`reembed`, and the rrf_k/weight_keyword defaults that were tuned for it).

Offline evidence (eval/external/runs/diagnostics/ranking/ranking_experiments_report.md,
section (c)): the winning template was `f"search_document: [{date}] {speaker}: {content}"`
where `content` already carried a `"[User] "`/`"[Name] "` bracket tag, so the speaker name
was embedded twice — flagged there as worth fixing before shipping. This implementation
strips that bracket tag before injecting the speaker, and drops the nomic `search_document:`
prefix entirely (prefixes measured harmful on raw content in the same sweep). Section (b)'s
keyword-weight sweep at the best (contextualized, rrf_k=120) cell found evidence recall@20
54.7% at weight_keyword=2.0 vs 50.8% on the pre-Tier-3d raw-content baseline (rrf_k=60,
weight_keyword=1.0) — hence the new `rrf_k`/`weight_keyword` defaults below.
"""

from __future__ import annotations

from tests.helpers import FakeEmbedder, FakeLLM
from woven_imprint.config import get_config
from woven_imprint.embedding.cache import CachedEmbedder
from woven_imprint.engine import Engine
from woven_imprint.maintenance import Budget, MaintenanceRunner
from woven_imprint.memory.store import build_embed_text


class SpyEmbedder(CachedEmbedder):
    """Records every text it was asked to embed, cache hits included.

    Subclasses `CachedEmbedder` itself (rather than wrapping a plain
    `FakeEmbedder`) because `Engine.__init__` auto-wraps any embedder that
    isn't already a `CachedEmbedder` — passing this in as-is means
    `char.memory.embedder is` this spy (no invisible extra wrapper the test
    can't see calls through), while still exercising the real caching
    behavior `reembed()` is documented to respect: re-embedding identical
    text (e.g. right after `add()` wrote it) is a cache hit and does not
    reach the inner `FakeEmbedder` — but it still shows up here, because the
    recording happens in `embed()` before the cache lookup.
    """

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


# --- build_embed_text (pure function) -------------------------------------


class TestBuildEmbedText:
    def test_user_row_with_user_id(self):
        text = build_embed_text(
            content="[User] I adopted a cat",
            role="user",
            created_at="2023-05-08 10:00:00",
            metadata={"user_id": "caroline"},
            character_name="Ada",
        )
        assert text == "[2023-05-08] User: caroline: I adopted a cat"

    def test_user_row_without_user_id(self):
        text = build_embed_text(
            content="[User] I adopted a cat",
            role="user",
            created_at="2023-05-08 10:00:00",
            metadata={},
            character_name="Ada",
        )
        assert text == "[2023-05-08] User: I adopted a cat"

    def test_user_row_metadata_none(self):
        text = build_embed_text(
            content="[User] hi",
            role="user",
            created_at="2023-05-08 10:00:00",
            metadata=None,
            character_name="Ada",
        )
        assert text == "[2023-05-08] User: hi"

    def test_character_row_strips_name_tag(self):
        text = build_embed_text(
            content="[Ada] Hey Caroline!",
            role="character",
            created_at="2023-05-08 10:00:00",
            metadata={},
            character_name="Ada",
        )
        assert text == "[2023-05-08] Ada: Hey Caroline!"

    def test_core_observation_row_unchanged_content(self):
        text = build_embed_text(
            content="Caroline attended an LGBTQ support group yesterday.",
            role="observation",
            created_at="2023-05-09 08:00:00",
            metadata={"source": "extraction"},
            character_name="Ada",
        )
        assert text == "[2023-05-09] Caroline attended an LGBTQ support group yesterday."

    def test_event_row_unchanged_content(self):
        text = build_embed_text(
            content="[Event] It rained all day",
            role="event",
            created_at="2023-05-09 08:00:00",
            metadata={},
            character_name="Ada",
        )
        assert text == "[2023-05-09] [Event] It rained all day"

    def test_missing_created_at_falls_back_to_empty_date(self):
        text = build_embed_text(
            content="[User] hi", role="user", created_at=None, metadata=None, character_name="Ada"
        )
        assert text == "[] User: hi"


# --- MemoryStore.add wires build_embed_text through embedding_context ----


class TestAddUsesEmbedContext:
    def test_user_turn_embeds_contextualized_text(self):
        _, char, embedder = _char_with_spy()
        char.memory.add(
            content="[User] I adopted a cat",
            tier="buffer",
            role="user",
            metadata={"user_id": "caroline"},
        )
        assert embedder.calls, "embedder was never called"
        last = embedder.calls[-1]
        assert last.startswith("[")
        assert "User: caroline: I adopted a cat" in last
        assert last != "[User] I adopted a cat"

    def test_user_turn_without_user_id(self):
        _, char, embedder = _char_with_spy()
        char.memory.add(content="[User] hi there", tier="buffer", role="user")
        last = embedder.calls[-1]
        assert "User: hi there" in last
        assert "User: None" not in last

    def test_character_turn_embeds_contextualized_text(self):
        _, char, embedder = _char_with_spy()
        char.memory.add(content="[Ada] Hey Caroline!", tier="buffer", role="character")
        last = embedder.calls[-1]
        assert "Ada: Hey Caroline!" in last
        assert last.count("Ada") == 1  # name not duplicated (bracket tag + injected speaker)

    def test_core_fact_row_gets_date_context_only(self):
        _, char, embedder = _char_with_spy()
        char.memory.add(
            content="Caroline has a cat named Max.",
            tier="core",
            role="observation",
            metadata={"source": "extraction"},
        )
        last = embedder.calls[-1]
        assert last.endswith("Caroline has a cat named Max.")
        assert last.startswith("[")

    def test_stored_content_is_never_the_contextualized_text(self):
        _, char, embedder = _char_with_spy()
        mem = char.memory.add(content="[User] plain content", tier="buffer", role="user")
        assert mem["content"] == "[User] plain content"

    def test_flag_off_embeds_raw_content(self):
        _, char, embedder = _char_with_spy()
        cfg = get_config()
        original = cfg.memory.embedding_context
        cfg.memory.embedding_context = False
        try:
            char.memory.add(
                content="[User] I adopted a cat",
                tier="buffer",
                role="user",
                metadata={"user_id": "caroline"},
            )
            assert embedder.calls[-1] == "[User] I adopted a cat"
        finally:
            cfg.memory.embedding_context = original


# --- reembed ---------------------------------------------------------------


class TestReembedStreaming:
    """Tier 3d final-review fix wave, item 4: `reembed()` streams pages via
    `storage.iter_memory_rows_for_reembed` instead of materializing every
    active row (embeddings included) up front via `get_memories(limit=None)`.
    """

    def test_reembeds_1200_rows_streamed(self):
        _, char, _embedder = _char_with_spy()
        for i in range(1200):
            char.memory.add(content=f"filler memory number {i}", tier="core")

        count = char.memory.reembed(batch_size=500)

        assert count == 1200

    def test_never_calls_get_memories_with_limit_none(self, monkeypatch):
        """Monkeypatch `get_memories` to fail on the old (materialize-everything)
        call shape — `reembed()` must never take that path any more."""
        _, char, _embedder = _char_with_spy()
        for i in range(5):
            char.memory.add(content=f"filler memory number {i}", tier="core")

        orig = char.memory.storage.get_memories

        def guarded(*args, **kwargs):
            limit = kwargs.get("limit", args[3] if len(args) > 3 else 1000)
            if limit is None:
                raise AssertionError("reembed() must not call get_memories(limit=None)")
            return orig(*args, **kwargs)

        monkeypatch.setattr(char.memory.storage, "get_memories", guarded)

        count = char.memory.reembed()

        assert count == 5

    def test_page_rows_carry_no_embedding_key(self):
        """`iter_memory_rows_for_reembed`'s rows are plain id/content/role/
        created_at/metadata dicts — no `embedding` key at all, not even
        `None` — so a vector is never even briefly held for a row it's about
        to recompute."""
        _, char, _embedder = _char_with_spy()
        char.memory.add(content="a memory", tier="core")

        pages = list(char.memory.storage.iter_memory_rows_for_reembed(char.id, page=500))

        assert len(pages) == 1 and len(pages[0]) == 1
        assert "embedding" not in pages[0][0]
        assert set(pages[0][0]) == {"id", "content", "role", "created_at", "metadata"}


class TestReembed:
    def test_recomputes_every_active_row(self):
        _, char, embedder = _char_with_spy()
        char.memory.add(content="[User] first message", tier="buffer", role="user")
        char.memory.add(content="[Ada] first reply", tier="buffer", role="character")
        char.memory.add(content="Caroline has a cat.", tier="core", role="observation", metadata={})
        embedder.calls.clear()

        count = char.memory.reembed(batch_size=2)

        assert count == 3
        # every active row's contextualized text was sent to the embedder
        assert len(embedder.calls) == 3
        assert any("User: first message" in c for c in embedder.calls)
        assert any("Ada: first reply" in c for c in embedder.calls)
        assert any("Caroline has a cat." in c for c in embedder.calls)

    def test_idempotent_second_call_recomputes_same_vectors(self):
        _, char, embedder = _char_with_spy()
        char.memory.add(content="[User] hello world", tier="buffer", role="user")
        first_vectors = {m["id"]: m["embedding"] for m in char.memory.get_all()}

        char.memory.reembed()
        after_first = {m["id"]: m["embedding"] for m in char.memory.get_all()}
        char.memory.reembed()
        after_second = {m["id"]: m["embedding"] for m in char.memory.get_all()}

        assert after_first == after_second
        assert first_vectors == after_first  # same builder, same content -> same vector

    def test_reembed_skips_archived_memories(self):
        _, char, embedder = _char_with_spy()
        mem = char.memory.add(content="[User] to be archived", tier="buffer", role="user")
        char.memory.archive(mem["id"])
        embedder.calls.clear()

        count = char.memory.reembed()

        assert count == 0
        assert embedder.calls == []

    def test_retrieval_finds_memory_after_reembed(self):
        _, char, _embedder = _char_with_spy()
        char.memory.add(
            content="[User] the aardvark burrowed under the greenhouse",
            tier="buffer",
            role="user",
        )
        char.memory.reembed()
        results = char.retriever.retrieve(query="aardvark greenhouse", limit=5)
        assert any("aardvark" in m["content"] for m in results)


# --- maintenance job ---------------------------------------------------------


class TestMaintenanceReembedJob:
    def test_reembed_not_in_default_jobs(self):
        assert "reembed" not in MaintenanceRunner.DEFAULT_JOBS

    def test_reembed_job_runs_when_explicitly_requested(self):
        _, char, embedder = _char_with_spy()
        char.memory.add(content="[User] a fact to reembed", tier="buffer", role="user")
        embedder.calls.clear()

        runner = MaintenanceRunner(char, budget=Budget(limit=0))
        report = runner.run(jobs=["reembed"])

        assert report["jobs"]["reembed"]["status"] == "ok"
        assert report["jobs"]["reembed"]["reembedded"] == 1
        assert embedder.calls  # the job actually called the embedder


# --- CLI smoke ---------------------------------------------------------------


class TestReembedCLI:
    def test_cmd_reembed_prints_count(self, tmp_path, capsys, monkeypatch):
        from woven_imprint import cli

        db = str(tmp_path / "characters.db")
        engine = Engine(db_path=db, llm=FakeLLM(), embedding=FakeEmbedder())
        char = engine.create_character("Ada")
        char.memory.add(content="[User] hello", tier="buffer", role="user")
        engine.close()

        monkeypatch.setattr(
            cli,
            "_get_engine",
            lambda db_path=None, model=None: Engine(
                db_path=db, llm=FakeLLM(), embedding=FakeEmbedder()
            ),
        )

        args = argparse_namespace(character="Ada", batch=64, db=db, model=None)
        cli.cmd_reembed(args)

        out = capsys.readouterr().out
        assert "1" in out
        assert "Ada" in out


def argparse_namespace(**kw):
    import argparse

    return argparse.Namespace(**kw)


def test_edit_reembeds_with_context():
    """edit() must embed the same contextualized text add() would (T2 review note)."""
    from tests.helpers import make_test_engine

    engine = make_test_engine()
    seen: list[str] = []
    inner = engine.embedder

    class Spy:
        model = "spy"

        def embed(self, text):
            seen.append(text)
            return inner.embed(text)

        def embed_batch(self, texts):
            seen.extend(texts)
            return inner.embed_batch(texts)

        def dimensions(self):
            return inner.dimensions()

    char = engine.create_character("Ada")
    char.memory.embedder = Spy()
    row = char.memory.add(
        "[User] I adopted a cat", tier="buffer", role="user", metadata={"user_id": "caroline"}
    )
    seen.clear()
    char.memory.edit(row["id"], content="[User] I adopted two cats")
    assert seen and seen[-1].startswith("[") and "User: caroline: I adopted two cats" in seen[-1]
