from tests.helpers import FakeEmbedder, FakeLLM
from woven_imprint.engine import Engine


class CallbackLLM(FakeLLM):
    def generate_json(self, messages, temperature=0.3, **kw):
        system = messages[0].get("content", "")
        if "conversation hooks" in system.lower():
            return [
                {
                    "hook": "You mentioned an interview — how did it go?",
                    "kind": "open_thread",
                    "sources": [1],
                },
                {
                    "hook": "Wonder how Anna's rowing season is going.",
                    "kind": "curiosity",
                    "sources": [2],
                },
            ]
        return super().generate_json(messages, temperature=temperature, **kw)

    def generate_json_robust(self, messages, temperature=0.3, **kw):
        return self.generate_json(messages, temperature=temperature, **kw)


def _engine_with_callback_llm():
    engine = Engine(db_path=":memory:", llm=CallbackLLM(), embedding=FakeEmbedder())
    orig = engine.create_character

    def _mk(*a, **kw):
        c = orig(*a, **kw)
        c.background = False
        c.parallel = False
        return c

    engine.create_character = _mk
    return engine


def test_refresh_creates_paraphrased_callbacks():
    engine = _engine_with_callback_llm()
    char = engine.create_character("Hooky")
    char.memory.add(
        "keeper has a job interview on friday", tier="core", role="observation", importance=0.8
    )
    char.memory.add(
        "keeper's sister anna loves rowing", tier="core", role="observation", importance=0.75
    )
    created = char.refresh_callbacks()
    assert created == 2
    cbs = char.get_callbacks(limit=5)
    assert len(cbs) == 2
    assert cbs[0]["kind"] in ("open_thread", "callback", "milestone", "curiosity")
    assert "freshness" in cbs[0]
    assert cbs[0]["source_memory_ids"]  # mapped from source indices


def test_get_callbacks_is_instant_db_read():
    engine = _engine_with_callback_llm()
    char = engine.create_character("Quick")
    char.memory.add("something important happened", tier="core", role="observation", importance=0.9)
    char.refresh_callbacks()
    before = char.llm.call_count
    char.get_callbacks()
    assert char.llm.call_count == before  # zero LLM on read


def test_ready_cap_expires_oldest():
    engine = _engine_with_callback_llm()
    char = engine.create_character("Capped")
    for i in range(6):
        engine.storage.save_callback(
            {
                "id": f"old-{i}",
                "character_id": char.id,
                "kind": "curiosity",
                "hook": f"old hook {i}",
                "salience": 0.1,
            }
        )
    from woven_imprint.config import get_config

    get_config().maintenance.callbacks_ready_cap = 5
    try:
        char.memory.add("fresh important fact", tier="core", role="observation", importance=0.9)
        char.refresh_callbacks()
        ready = engine.storage.get_callbacks(char.id, status="ready", limit=50)
        assert len(ready) <= 5
    finally:
        from woven_imprint.config import reload_config

        reload_config()


def test_maintenance_job_callbacks_registered():
    from woven_imprint.maintenance import MaintenanceRunner

    assert "callbacks" in MaintenanceRunner.DEFAULT_JOBS
    engine = _engine_with_callback_llm()
    char = engine.create_character("Job")
    char.memory.add(
        "keeper started learning the violin", tier="core", role="observation", importance=0.8
    )
    report = MaintenanceRunner(char).run(jobs=["callbacks"])
    assert report["jobs"]["callbacks"]["status"] == "ok"
    assert report["jobs"]["callbacks"]["created"] >= 1
