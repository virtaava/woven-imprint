import pytest

from tests.helpers import make_test_engine


def test_meta_dimensions_set_on_first_embed():
    engine = make_test_engine()
    char = engine.create_character("Dima")
    char.memory.add("hello world", tier="buffer")
    assert engine.storage.meta_get("embedding_dimensions") == "50"  # FakeEmbedder dims


def test_dimension_mismatch_raises():
    engine = make_test_engine()
    char = engine.create_character("Dima")
    char.memory.add("hello", tier="buffer")
    engine.storage.meta_set("embedding_dimensions", "768")  # simulate model swap
    with pytest.raises(ValueError, match="dimension"):
        char.memory.add("world", tier="buffer")
