from datetime import datetime, timedelta, timezone

from tests.helpers import FakeEmbedder, FakeLLM, make_test_engine
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


def test_unknown_job_fails_loudly():
    engine = make_test_engine()
    char = engine.create_character("Oops")
    runner = MaintenanceRunner(char)
    report = runner.run(jobs=["nonexistent"])
    assert report["jobs"]["nonexistent"]["status"] == "failed"
