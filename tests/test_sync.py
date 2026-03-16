"""Tests for sync adapters."""

import tempfile
from pathlib import Path

from woven_imprint.storage.sqlite import SQLiteStorage
from woven_imprint.character import Character
from woven_imprint.persona.model import PersonaModel
from woven_imprint.sync.claude_code import ClaudeCodeSync
from woven_imprint.sync.hermes import HermesSync
from woven_imprint.sync.openclaw import OpenClawSync
from woven_imprint.sync.generic import GenericMarkdownSync


class FakeEmbedder:
    def embed(self, text):
        return [0.1, 0.2, 0.3]

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]

    def dimensions(self):
        return 3


class FakeLLM:
    def generate(self, messages, **kw):
        return "Response"

    def generate_json(self, messages, **kw):
        return []


def _make_char():
    storage = SQLiteStorage(":memory:")
    persona = PersonaModel(
        {
            "name": "Alice",
            "hard": {"name": "Alice", "species": "human"},
            "soft": {"personality": "witty and sharp", "speaking_style": "dry humor"},
            "backstory": "A detective in London",
        }
    )
    storage.save_character("c1", "Alice", persona.to_dict())
    char = Character("c1", storage, FakeLLM(), FakeEmbedder(), persona)
    char.enforce_consistency = False
    # Add some memories
    char.memory.add("I solved the riverside case last week", tier="core", importance=0.7)
    char.memory.add("My office is above a chip shop in Brixton", tier="bedrock", importance=0.9)
    return char, storage


class TestClaudeCodeSync:
    def test_sync_creates_claude_md(self):
        char, storage = _make_char()
        with tempfile.TemporaryDirectory() as tmpdir:
            sync = ClaudeCodeSync(project_dir=tmpdir)
            sync.sync(char)

            claude_md = Path(tmpdir) / "CLAUDE.md"
            assert claude_md.exists()
            content = claude_md.read_text()
            assert "Alice" in content
            assert "WOVEN_IMPRINT_START" in content
            assert "detective" in content.lower()
        storage.close()

    def test_sync_creates_memory_file(self):
        char, storage = _make_char()
        with tempfile.TemporaryDirectory() as tmpdir:
            sync = ClaudeCodeSync(project_dir=tmpdir)
            files = sync.sync(char)

            # Should have written a memory file
            assert len(files) == 2
            memory_files = [f for f in Path(tmpdir).glob("*.md") if "woven_imprint" in f.name]
            assert len(memory_files) >= 1
        storage.close()

    def test_unsync_removes_block(self):
        char, storage = _make_char()
        with tempfile.TemporaryDirectory() as tmpdir:
            sync = ClaudeCodeSync(project_dir=tmpdir)
            sync.sync(char)

            claude_md = Path(tmpdir) / "CLAUDE.md"
            assert "WOVEN_IMPRINT_START" in claude_md.read_text()

            sync.unsync()
            content = claude_md.read_text()
            assert "WOVEN_IMPRINT_START" not in content
        storage.close()

    def test_sync_preserves_existing_claude_md(self):
        char, storage = _make_char()
        with tempfile.TemporaryDirectory() as tmpdir:
            claude_md = Path(tmpdir) / "CLAUDE.md"
            claude_md.write_text("# My Project\n\nExisting content here.\n")

            sync = ClaudeCodeSync(project_dir=tmpdir)
            sync.sync(char)

            content = claude_md.read_text()
            assert "My Project" in content
            assert "Existing content" in content
            assert "Alice" in content
        storage.close()

    def test_resync_updates_block(self):
        char, storage = _make_char()
        with tempfile.TemporaryDirectory() as tmpdir:
            sync = ClaudeCodeSync(project_dir=tmpdir)
            sync.sync(char)

            # Modify character
            char.persona.update_soft("personality", "now very cheerful")
            sync.sync(char)

            content = (Path(tmpdir) / "CLAUDE.md").read_text()
            assert "cheerful" in content
            # Should only have ONE block
            assert content.count("WOVEN_IMPRINT_START") == 1
        storage.close()


class TestGenericSync:
    def test_sync_to_new_file(self):
        char, storage = _make_char()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "character.md"
            sync = GenericMarkdownSync(path)
            sync.sync(char)

            assert path.exists()
            content = path.read_text()
            assert "Alice" in content
            assert "detective" in content.lower()
        storage.close()

    def test_unsync_removes_from_existing(self):
        char, storage = _make_char()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rules.md"
            path.write_text("# My Rules\n\nBe helpful.\n")

            sync = GenericMarkdownSync(path)
            sync.sync(char)
            assert "Alice" in path.read_text()

            sync.unsync()
            content = path.read_text()
            assert "My Rules" in content
            assert "Alice" not in content
        storage.close()


class TestHermesSync:
    def test_sync_creates_persona_file(self):
        char, storage = _make_char()
        with tempfile.TemporaryDirectory() as tmpdir:
            sync = HermesSync(hermes_dir=tmpdir)
            sync.sync(char)

            persona_path = Path(tmpdir) / "woven_imprint_persona.md"
            assert persona_path.exists()
            content = persona_path.read_text()
            assert "Alice" in content
        storage.close()

    def test_unsync_removes_file(self):
        char, storage = _make_char()
        with tempfile.TemporaryDirectory() as tmpdir:
            sync = HermesSync(hermes_dir=tmpdir)
            sync.sync(char)
            sync.unsync()
            assert not (Path(tmpdir) / "woven_imprint_persona.md").exists()
        storage.close()


class TestOpenClawSync:
    def test_sync_creates_character_file(self):
        char, storage = _make_char()
        with tempfile.TemporaryDirectory() as tmpdir:
            sync = OpenClawSync(workspace_dir=tmpdir)
            sync.sync(char)

            char_path = Path(tmpdir) / "woven_imprint_character.md"
            assert char_path.exists()
            assert "Alice" in char_path.read_text()
        storage.close()

    def test_sync_injects_into_existing_memory_md(self):
        char, storage = _make_char()
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_md = Path(tmpdir) / "MEMORY.md"
            memory_md.write_text("# Workspace Memory\n\nExisting stuff.\n")

            sync = OpenClawSync(workspace_dir=tmpdir)
            sync.sync(char)

            content = memory_md.read_text()
            assert "Existing stuff" in content
            assert "Alice" in content
            assert "WOVEN_IMPRINT_START" in content
        storage.close()
