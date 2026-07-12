from datetime import datetime, timedelta, timezone

from tests.helpers import FakeEmbedder, FakeLLM, make_test_engine
from woven_imprint.config import get_config
from woven_imprint.engine import Engine
from woven_imprint.maintenance import Budget, MaintenanceRunner


def _age_memory(engine, memory_id, days):
    old = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    engine.storage._conn.execute(
        "UPDATE memories SET created_at = ? WHERE id = ?", (old, memory_id)
    )
    engine.storage._conn.commit()


def test_budget():
    b = Budget(2)
    assert b.take() and b.take() and not b.take()
    assert b.used == 2
    assert Budget(None).take(100)  # unlimited


def test_buffer_hygiene_archives_old_low_importance():
    engine = make_test_engine()
    char = engine.create_character("Tidy")
    fresh = char.memory.add("fresh chatter", tier="buffer")
    old_low = char.memory.add("old chatter", tier="buffer")
    old_important = char.memory.add("old but important", tier="buffer", importance=0.8)
    _age_memory(engine, old_low["id"], 30)
    _age_memory(engine, old_important["id"], 30)

    runner = MaintenanceRunner(char)
    report = runner.run(jobs=["buffer_hygiene"])
    assert report["jobs"]["buffer_hygiene"]["status"] == "ok"
    assert report["jobs"]["buffer_hygiene"]["archived"] == 1
    active_ids = {m["id"] for m in char.memory.get_all(tier="buffer")}
    assert fresh["id"] in active_ids and old_important["id"] in active_ids
    assert old_low["id"] not in active_ids


class ScoringLLM(FakeLLM):
    def generate_json(self, messages, temperature=0.3, **kw):
        system = messages[0].get("content", "")
        if "importance" in system.lower() or "score" in system.lower():
            user = messages[-1]["content"]
            n = user.count("\n1.") + sum(user.count(f"\n{i}.") for i in range(2, 40))
            return [8] * max(1, n)
        return super().generate_json(messages, temperature=temperature, **kw)

    def generate_json_robust(self, messages, temperature=0.3, **kw):
        return self.generate_json(messages, temperature=temperature, **kw)


def test_score_importance_updates_defaults_only():
    engine = Engine(db_path=":memory:", llm=ScoringLLM(), embedding=FakeEmbedder())
    char = engine.create_character("Scored")
    char.background = False
    default_mem = char.memory.add("something notable happened", tier="buffer")
    scored_mem = char.memory.add("already scored", tier="buffer", importance=0.9)

    runner = MaintenanceRunner(char)
    report = runner.run(jobs=["score_importance"])
    assert report["jobs"]["score_importance"]["scored"] >= 1
    assert engine.storage.get_memory(default_mem["id"])["importance"] == 0.8  # 8/10
    assert engine.storage.get_memory(scored_mem["id"])["importance"] == 0.9  # untouched


def test_budget_exhaustion_skips_llm_jobs():
    engine = make_test_engine()
    char = engine.create_character("Broke")
    char.memory.add("something", tier="buffer")
    runner = MaintenanceRunner(char, budget=Budget(0))
    report = runner.run(jobs=["score_importance"])
    assert report["jobs"]["score_importance"]["status"] == "skipped"
    assert report["llm_calls_used"] == 0


def test_job_consolidate_happy_path(monkeypatch):
    # Mirrors tests/test_consolidation.py's seeding style: many buffer
    # memories with near-identical wording so they cluster and force a
    # real _summarize_cluster (LLM) call.
    engine = make_test_engine()
    char = engine.create_character("Piper")
    char.consolidator.threshold = 10
    for i in range(20):
        char.memory.add(f"the lake was calm on day {i}", tier="buffer")

    captured = {}
    orig_drain = char.consolidator.drain

    def spy_drain(*a, **kw):
        result = orig_drain(*a, **kw)
        captured["result"] = result
        return result

    monkeypatch.setattr(char.consolidator, "drain", spy_drain)

    runner = MaintenanceRunner(char)
    report = runner.run(jobs=["consolidate"])
    entry = report["jobs"]["consolidate"]

    assert entry["status"] == "ok"
    assert entry["archived"] > 0
    assert entry["llm_calls"] == captured["result"]["llm_calls"]


def test_job_consolidate_skips_below_threshold():
    engine = make_test_engine()
    char = engine.create_character("Sparse")
    char.memory.add("just one note", tier="buffer")

    runner = MaintenanceRunner(char)
    report = runner.run(jobs=["consolidate"])
    assert report["jobs"]["consolidate"]["status"] == "skipped"


def test_job_consolidate_budget_cap(monkeypatch):
    # Two well-separated content groups so clustering produces 2+
    # multi-member clusters within a single low-chunk_size pass; a
    # Budget(1) should allow the first cluster's LLM call and deny the rest.
    engine = make_test_engine()
    char = engine.create_character("Capped")
    char.consolidator.threshold = 10
    monkeypatch.setattr(get_config().maintenance, "consolidate_chunk_size", 20)
    for i in range(20):
        topic = "lake" if i % 2 == 0 else "forest"
        char.memory.add(f"note {i} about the {topic}", tier="buffer")

    runner = MaintenanceRunner(char, budget=Budget(1))
    report = runner.run(jobs=["consolidate"])
    entry = report["jobs"]["consolidate"]

    assert entry.get("budget_exhausted") is True
    assert report["llm_calls_used"] <= 1


def test_unknown_job_fails_loudly():
    engine = make_test_engine()
    char = engine.create_character("Oops")
    runner = MaintenanceRunner(char)
    report = runner.run(jobs=["nonexistent"])
    assert report["jobs"]["nonexistent"]["status"] == "failed"


def test_buffer_hygiene_scale_exceeds_fetch_limit():
    """Verify buffer_hygiene sweeps ALL stale rows even when total exceeds fetch limit.

    Insert 1010 active buffer rows for one character: the 10 oldest aged to 30 days
    with importance 0.5, and 1000 fresh rows. With fetch limit=1000, if we ordered
    newest-first in SQL, the 10 oldest would be excluded entirely. Verify that
    oldest_first=True in get_memories() captures them and they all get archived.
    """
    engine = make_test_engine()
    char = engine.create_character("ScaleTest")

    # Insert 10 very old memories with low importance (will be swept)
    old_ids = []
    for i in range(10):
        mem = char.memory.add(f"very old memory {i}", tier="buffer", importance=0.5)
        old_ids.append(mem["id"])

    # Insert 1000 fresh memories (will survive)
    fresh_ids = []
    for i in range(1000):
        mem = char.memory.add(f"fresh memory {i}", tier="buffer")
        fresh_ids.append(mem["id"])

    # Age the old ones to 30 days (exceeds default TTL)
    for old_id in old_ids:
        _age_memory(engine, old_id, 30)

    # Run buffer_hygiene with fetch limit=1000
    runner = MaintenanceRunner(char)
    report = runner.run(jobs=["buffer_hygiene"])

    assert report["jobs"]["buffer_hygiene"]["status"] == "ok"
    # All 10 stale rows should be archived despite the 1000-row fetch limit
    assert report["jobs"]["buffer_hygiene"]["archived"] == 10
    assert report["jobs"]["buffer_hygiene"]["truncated"] is True

    # Verify: all old ones gone, all fresh ones still active
    active_ids = {m["id"] for m in char.memory.get_all(tier="buffer")}
    for old_id in old_ids:
        assert old_id not in active_ids, f"Stale memory {old_id} should have been archived"
    for fresh_id in fresh_ids:
        assert fresh_id in active_ids, f"Fresh memory {fresh_id} should still be active"


def test_dedup_archives_near_duplicates_and_reinforces_kept():
    engine = make_test_engine()
    char = engine.create_character("Dedup")
    a = char.memory.add(
        "the user's sister anna loves rowing", tier="core", role="observation", importance=0.8
    )
    b = char.memory.add(
        "the user's sister anna loves rowing", tier="core", role="observation", importance=0.7
    )
    unrelated = char.memory.add(
        "the moon rises over the lake", tier="core", role="observation", importance=0.7
    )
    runner = MaintenanceRunner(char)
    report = runner.run(jobs=["dedup"])
    assert report["jobs"]["dedup"]["archived"] == 1
    active = {m["id"] for m in char.memory.get_all(tier="core")}
    assert a["id"] in active and unrelated["id"] in active and b["id"] not in active
    kept = engine.storage.get_memory(a["id"])
    assert kept["certainty"] > 1.0 - 1e-9 or kept["certainty"] == 1.0  # reinforced (clamped at 1.0)


def test_dedup_idempotent():
    engine = make_test_engine()
    char = engine.create_character("Dedup2")
    char.memory.add("fact one about cats", tier="core", role="observation")
    char.memory.add("fact one about cats", tier="core", role="observation")
    runner = MaintenanceRunner(char)
    first = runner.run(jobs=["dedup"])["jobs"]["dedup"]["archived"]
    second = runner.run(jobs=["dedup"])["jobs"]["dedup"]["archived"]
    assert first == 1 and second == 0


def test_dedup_triple_cluster_no_double_count():
    """Regression test: 4 identical memories with importances [0.7, 0.5, 0.99, 0.999].

    Dedup should archive exactly 3, leaving only the highest-importance memory.
    Previously, when drop==mems[i], the loop would continue and i could be
    used as 'keep' in a later j comparison, getting archived twice.
    """
    engine = make_test_engine()
    char = engine.create_character("DedupTriple")
    # Create 4 identical memories with increasing importances
    char.memory.add("identical fact", tier="core", role="observation", importance=0.7)
    char.memory.add("identical fact", tier="core", role="observation", importance=0.5)
    char.memory.add("identical fact", tier="core", role="observation", importance=0.99)
    mem_highest = char.memory.add(
        "identical fact", tier="core", role="observation", importance=0.999
    )

    runner = MaintenanceRunner(char)
    report = runner.run(jobs=["dedup"])
    assert report["jobs"]["dedup"]["archived"] == 3

    active = {m["id"] for m in char.memory.get_all(tier="core")}
    assert mem_highest["id"] in active, "Highest importance memory should remain active"
    assert len(active) == 1, "Exactly one memory should remain active"

    kept = engine.storage.get_memory(mem_highest["id"])
    assert kept["importance"] == 0.999


def test_reinforce_strengthens_reencountered_fact():
    engine = make_test_engine()
    char = engine.create_character("Rein")
    core = char.memory.add("keeper works as a software developer", tier="core", role="observation")
    engine.storage.update_memory_certainty(core["id"], -0.4)  # certainty 0.6
    char.memory.add("keeper works as a software developer", tier="buffer", role="user")
    runner = MaintenanceRunner(char)
    report = runner.run(jobs=["reinforce"])
    assert report["jobs"]["reinforce"]["reinforced"] == 1
    assert engine.storage.get_memory(core["id"])["certainty"] > 0.6


class ContradictionLLM(FakeLLM):
    def generate_json(self, messages, temperature=0.3, **kw):
        system = messages[0].get("content", "")
        if "contradict" in system.lower():
            return {"contradictory": True, "current": "second"}
        return super().generate_json(messages, temperature=temperature, **kw)

    def generate_json_robust(self, messages, temperature=0.3, **kw):
        return self.generate_json(messages, temperature=temperature, **kw)


def test_contradiction_sweep_marks_superseded(monkeypatch):
    engine = Engine(db_path=":memory:", llm=ContradictionLLM(), embedding=FakeEmbedder())
    char = engine.create_character("Contra")
    char.background = False
    old = char.memory.add(
        "keeper lives in the city near the harbor", tier="core", role="observation"
    )
    new = char.memory.add(
        "keeper lives in the country near the lake", tier="core", role="observation"
    )
    # FakeEmbedder bag-of-words: these share enough tokens to be candidates
    runner = MaintenanceRunner(char)
    report = runner.run(jobs=["contradictions"])
    assert report["jobs"]["contradictions"]["contradicted"] == 1
    assert engine.storage.get_memory(old["id"])["status"] == "contradicted"
    assert engine.storage.get_memory(new["id"])["status"] == "active"


def test_reflect_triggers_on_importance_sum():
    engine = make_test_engine()
    char = engine.create_character("Ref")
    for i in range(20):
        char.memory.add(f"noteworthy event {i}", tier="buffer", importance=0.7)  # sum 14 > 12
    report = MaintenanceRunner(char).run(jobs=["reflect"])
    assert report["jobs"]["reflect"]["status"] == "ok"
    reflections = [
        m for m in char.memory.get_all(tier="core") if m["content"].startswith("[Reflection]")
    ]
    assert len(reflections) == 1
    # second run: importance since last reflection is now ~0 → skipped
    report2 = MaintenanceRunner(char).run(jobs=["reflect"])
    assert report2["jobs"]["reflect"]["status"] == "skipped"


def test_evolve_skips_below_min_memories():
    engine = make_test_engine()
    char = engine.create_character("Evo")
    report = MaintenanceRunner(char).run(jobs=["evolve"])
    assert report["jobs"]["evolve"]["status"] == "skipped"


def test_job_callbacks_budget_zero_skips_without_llm_call():
    """Verify that _job_callbacks with Budget(0) returns skipped without calling LLM."""
    from tests.test_callbacks import CallbackLLM

    engine = Engine(db_path=":memory:", llm=CallbackLLM(), embedding=FakeEmbedder())
    char = engine.create_character("BudgetZero")
    char.background = False
    char.parallel = False
    char.memory.add("something important", tier="core", role="observation", importance=0.9)

    llm_calls_before = char.llm.call_count
    runner = MaintenanceRunner(char, budget=Budget(0))
    report = runner.run(jobs=["callbacks"])
    llm_calls_after = char.llm.call_count

    assert report["jobs"]["callbacks"]["status"] == "skipped"
    assert report["jobs"]["callbacks"]["created"] == 0
    assert llm_calls_after == llm_calls_before  # No LLM calls made
