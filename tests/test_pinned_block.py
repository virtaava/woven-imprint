from tests.helpers import make_test_engine
from woven_imprint.config import get_config


def _char():
    engine = make_test_engine()
    char = engine.create_character("Ada")
    char.background = False
    char.parallel = False
    char.enforce_consistency = False
    return engine, char


def test_pinned_block_in_base_and_not_duplicated():
    engine, char = _char()
    m = char.memory.add(
        "The visitor's brother died in March; never joke about it.", tier="core", importance=0.9
    )
    char.memory.pin(m["id"])
    char.chat("hello", user_id="toni")
    volatile = char.last_chat_messages[1]["content"]
    assert "Things you always remember:" in volatile
    assert volatile.count("brother died in March") == 1


def test_pinned_block_survives_tiny_budget():
    engine, char = _char()
    m = char.memory.add("Always call the visitor Captain.", tier="core")
    char.memory.pin(m["id"])
    for i in range(30):
        char.memory.add(f"Filler memory number {i} about the weather and errands.", tier="core")
    get_config().context.total_tokens = 400
    try:
        char.chat("hello", user_id="toni")
    finally:
        get_config().context.total_tokens = 6000
    assert "Always call the visitor Captain." in char.last_chat_messages[1]["content"]


def test_pinned_block_gated():
    engine, char = _char()
    m = char.memory.add("Pinned thing.", tier="core")
    char.memory.pin(m["id"])
    get_config().context.pinned_block = False
    try:
        char.chat("hi", user_id="toni")
        assert "Things you always remember" not in char.last_chat_messages[1]["content"]
    finally:
        get_config().context.pinned_block = True
