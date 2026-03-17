"""Tests for Engine — character lifecycle management."""

from woven_imprint import Engine


class FakeEmbedder:
    def embed(self, text):
        h = hash(text) % 1000
        return [h / 1000, (h * 7) % 1000 / 1000, (h * 13) % 1000 / 1000]

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]

    def dimensions(self):
        return 3


class FakeLLM:
    def generate(self, messages, **kw):
        return "I respond in character."

    def generate_json(self, messages, **kw):
        return []


def _engine():
    e = Engine(db_path=":memory:", llm=FakeLLM(), embedding=FakeEmbedder())
    return e


class TestCreateCharacter:
    def test_basic_create(self):
        engine = _engine()
        char = engine.create_character("Alice", persona={"personality": "witty"})
        assert char.name == "Alice"
        assert char.id is not None
        engine.close()

    def test_create_with_birthdate(self):
        engine = _engine()
        char = engine.create_character("Bob", birthdate="2000-06-15", persona={})
        assert char.persona.age is not None
        assert char.persona.age >= 25
        engine.close()

    def test_create_seeds_bedrock(self):
        engine = _engine()
        char = engine.create_character(
            "Alice",
            persona={"backstory": "A detective in London", "personality": "sharp"},
        )
        bedrock = char.memory.count(tier="bedrock")
        assert bedrock >= 2  # backstory + personality
        engine.close()

    def test_create_with_custom_id(self):
        engine = _engine()
        char = engine.create_character("Alice", character_id="my-alice", persona={})
        assert char.id == "my-alice"
        engine.close()

    def test_shorthand_fields_normalized(self):
        engine = _engine()
        char = engine.create_character(
            "Alice",
            persona={
                "personality": "witty",
                "speaking_style": "clipped",
                "occupation": "detective",
            },
        )
        assert char.persona.soft["personality"] == "witty"
        assert char.persona.soft["speaking_style"] == "clipped"
        engine.close()


class TestLoadCharacter:
    def test_load_existing(self):
        engine = _engine()
        created = engine.create_character("Alice", persona={"personality": "test"})
        loaded = engine.load_character(created.id)
        assert loaded is not None
        assert loaded.name == "Alice"
        engine.close()

    def test_load_nonexistent(self):
        engine = _engine()
        assert engine.load_character("nope") is None
        engine.close()

    def test_get_character_raises(self):
        engine = _engine()
        try:
            engine.get_character("nope")
            assert False, "Should have raised"
        except KeyError:
            pass
        engine.close()

    def test_load_restores_state(self):
        engine = _engine()
        char = engine.create_character("Alice", persona={})
        char.emotion.mood = "joyful"
        char.emotion.intensity = 0.8
        char._save_state()

        loaded = engine.load_character(char.id)
        assert loaded.emotion.mood == "joyful"
        assert loaded.emotion.intensity == 0.8
        engine.close()


class TestDeleteCharacter:
    def test_delete(self):
        engine = _engine()
        char = engine.create_character("Alice", persona={})
        engine.delete_character(char.id)
        assert engine.load_character(char.id) is None
        engine.close()

    def test_delete_cleans_memories(self):
        engine = _engine()
        char = engine.create_character("Alice", persona={"backstory": "test"})
        char_id = char.id
        engine.delete_character(char_id)
        mems = engine.storage.get_memories(char_id)
        assert len(mems) == 0
        engine.close()


class TestListCharacters:
    def test_list_empty(self):
        engine = _engine()
        assert engine.list_characters() == []
        engine.close()

    def test_list_multiple(self):
        engine = _engine()
        engine.create_character("Alice", persona={})
        engine.create_character("Bob", persona={})
        chars = engine.list_characters()
        assert len(chars) == 2
        names = {c["name"] for c in chars}
        assert names == {"Alice", "Bob"}
        engine.close()


class TestImportExport:
    def test_export_import_roundtrip(self):
        import tempfile
        from pathlib import Path

        engine = _engine()
        original = engine.create_character(
            "Alice",
            birthdate="1998-03-15",
            persona={"backstory": "A detective", "personality": "witty"},
        )
        original.memory.add("Solved the harbor case", tier="core", importance=0.8)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            original.export(f.name)
            path = f.name

        # Import into fresh engine
        engine2 = Engine(db_path=":memory:", llm=FakeLLM(), embedding=FakeEmbedder())
        imported = engine2.import_character(path)

        assert imported.name == "Alice"
        assert imported.persona.age is not None
        assert imported.memory.count(tier="core") >= 1

        Path(path).unlink()
        engine.close()
        engine2.close()


class TestContextManager:
    def test_with_statement(self):
        with Engine(db_path=":memory:", llm=FakeLLM(), embedding=FakeEmbedder()) as engine:
            char = engine.create_character("Alice", persona={})
            assert char.name == "Alice"
        # engine.close() called automatically
