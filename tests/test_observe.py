from tests.helpers import make_test_engine


def test_observe_stores_event_memory():
    engine = make_test_engine()
    char = engine.create_character("Watcher")
    mem = char.observe("The keeper fed you moonberries", source="game_sim")
    assert mem["content"] == "[Event] The keeper fed you moonberries"
    assert mem["role"] == "event"
    assert mem["importance"] == 0.6
    stored = char.memory.get(mem["id"])
    assert stored["metadata"]["source"] == "game_sim"


def test_observe_no_dialogue_pair_pollution():
    """Events must NOT run the dialogue-pair fact extractor (empty-string pollution)."""
    engine = make_test_engine()
    char = engine.create_character("Watcher")
    before = char.llm.call_count
    char.observe("It rained all day")  # no user_id → no assessments at all
    assert char.llm.call_count == before  # zero LLM calls


def test_observe_with_user_updates_relationship():
    engine = make_test_engine()
    char = engine.create_character("Watcher")
    char.observe("Keeper stayed up all night nursing you back to health", user_id="keeper")
    char.flush()
    rel = char.get_relationship("keeper")
    assert rel is not None  # relationship row created + assessed


def test_observe_importance_override():
    engine = make_test_engine()
    char = engine.create_character("Watcher")
    mem = char.observe("You evolved into a fledgling!", importance=0.95)
    assert mem["importance"] == 0.95


def test_observe_background_path():
    """Events with background=True queue work and increment success counters."""
    engine = make_test_engine()
    char = engine.create_character("Watcher")
    char.background = True  # Enable background worker

    char.observe("Keeper gave you a gift", user_id="keeper")
    assert char.flush(timeout=10)  # Wait for background work to finish

    # Verify relationship was created
    rel = char.get_relationship("keeper")
    assert rel is not None

    # Verify worker was created
    assert char._worker is not None

    # Verify success counter was incremented
    assert char._worker.success_counts.get("observe", 0) >= 1


def test_observe_lightweight_mode():
    """Lightweight mode skips emotion but still assesses relationships."""
    engine = make_test_engine()
    char = engine.create_character("Watcher")
    char.lightweight = True
    char.background = False

    # Record initial mood
    mood_before = char.emotion.mood

    # Observe an event
    char.observe("A storm rolled in", user_id="keeper")
    char.flush()

    # Verify relationship was updated
    rel = char.get_relationship("keeper")
    assert rel is not None

    # Verify observe subsystem shows success
    health = char.health()
    assert health["subsystems"]["observe"]["success"] >= 1

    # Verify emotion mood was NOT changed (only 1 LLM call for relationship)
    assert char.emotion.mood == mood_before
    assert char.llm.call_count == 1  # Only relationship assessment, no emotion
