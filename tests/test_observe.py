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
