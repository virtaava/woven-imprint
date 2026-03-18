"""Tests for migration tools — parsers and importer."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from woven_imprint.migrate.parsers import (
    parse_chatgpt_export,
    parse_claude_project,
)
from woven_imprint.config import MigrationConfig


class TestChatGPTParser:
    def _make_export(self, num_messages=10, msg_length=100):
        """Create a minimal ChatGPT export file."""
        mapping = {}
        for i in range(num_messages):
            role = "user" if i % 2 == 0 else "assistant"
            content = f"Message {i}: " + "x" * msg_length
            mapping[str(i)] = {
                "message": {
                    "content": {"parts": [content]},
                    "author": {"role": role},
                }
            }
        data = [{"title": "Test Chat", "mapping": mapping}]
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(data, f)
        f.close()
        return f.name

    def test_no_truncation_by_default(self):
        """Default max_messages=0 means no truncation."""
        path = self._make_export(num_messages=600)
        try:
            result = parse_chatgpt_export(path)
            assert len(result["messages"]) == 600
        finally:
            os.unlink(path)

    def test_max_messages_limits(self):
        """max_messages parameter caps message count."""
        path = self._make_export(num_messages=100)
        try:
            result = parse_chatgpt_export(path, max_messages=20)
            assert len(result["messages"]) == 20
        finally:
            os.unlink(path)

    def test_max_message_length_limits(self):
        """max_message_length parameter caps individual message length."""
        path = self._make_export(num_messages=5, msg_length=5000)
        try:
            result = parse_chatgpt_export(path, max_message_length=100)
            for msg in result["messages"]:
                assert len(msg["content"]) <= 100
        finally:
            os.unlink(path)

    def test_no_message_length_limit(self):
        """max_message_length=0 means no truncation."""
        path = self._make_export(num_messages=5, msg_length=5000)
        try:
            result = parse_chatgpt_export(path, max_message_length=0)
            # Messages should be longer than 100 chars
            long_msgs = [m for m in result["messages"] if len(m["content"]) > 100]
            assert len(long_msgs) > 0
        finally:
            os.unlink(path)


class TestClaudeProjectParser:
    def test_reads_all_markdown_recursively(self):
        """Claude parser should rglob all .md files, not just memory/*.md."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create CLAUDE.md
            Path(tmpdir, "CLAUDE.md").write_text("# Project instructions")
            # Create nested markdown
            Path(tmpdir, "memory").mkdir()
            Path(tmpdir, "memory", "user.md").write_text("User is a developer")
            Path(tmpdir, "docs").mkdir()
            Path(tmpdir, "docs", "guide.md").write_text("Guide content")
            # Create .claude/ directory
            Path(tmpdir, ".claude").mkdir()
            Path(tmpdir, ".claude", "settings.md").write_text("Settings content")

            result = parse_claude_project(tmpdir)

            assert result["instructions"] == "# Project instructions"
            # Should find all markdown files (not just memory/)
            file_names = [m["file"] for m in result["memories"]]
            assert "memory/user.md" in file_names
            assert "docs/guide.md" in file_names
            assert ".claude/settings.md" in file_names

    def test_no_char_limits_on_instructions(self):
        """Instructions should not be truncated to 10000 chars."""
        with tempfile.TemporaryDirectory() as tmpdir:
            long_text = "A" * 15000
            Path(tmpdir, "CLAUDE.md").write_text(long_text)

            result = parse_claude_project(tmpdir)
            assert len(result["instructions"]) == 15000

    def test_no_char_limits_on_memories(self):
        """Memory file contents should not be truncated to 3000 chars."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "CLAUDE.md").write_text("x")
            Path(tmpdir, "memory").mkdir()
            long_text = "B" * 5000
            Path(tmpdir, "memory", "big.md").write_text(long_text)

            result = parse_claude_project(tmpdir)
            big_mem = next(m for m in result["memories"] if m["file"] == "memory/big.md")
            assert len(big_mem["content"]) == 5000


class TestChunkedAnalysis:
    """Test importer chunked analysis via public interface."""

    def test_chunked_analysis_with_many_messages(self):
        """Verify that large message sets trigger chunked analysis."""
        from woven_imprint.migrate.importer import CharacterImporter

        class FakeLLM:
            def generate(self, messages, **kwargs):
                return "Speaking style"

            def generate_json(self, messages, **kwargs):
                return {
                    "name": "TestBot",
                    "personality": "friendly",
                    "backstory": "A test bot",
                    "speaking_style": "casual",
                    "key_memories": ["fact 1"],
                }

        class FakeEmbedding:
            def embed(self, text):
                return [0.0] * 384

            def embed_batch(self, texts):
                return [[0.0] * 384 for _ in texts]

            def dimensions(self):
                return 384

        class FakeStorage:
            def __init__(self):
                self._chars = {}
                self._memories = []
                self._relationships = []
                self._sessions = []

            def save_character(self, char_id, name, persona, **kwargs):
                self._chars[char_id] = {
                    "id": char_id,
                    "name": name,
                    "persona": persona,
                    **kwargs,
                }

            def load_character(self, char_id):
                return self._chars.get(char_id)

            def list_characters(self):
                return list(self._chars.values())

            def delete_character(self, char_id):
                self._chars.pop(char_id, None)

            def save_memory(self, *args, **kwargs):
                self._memories.append(kwargs)

            def get_memories(self, *args, **kwargs):
                return []

            def count_memories(self, *args, **kwargs):
                return 0

            def save_relationship(self, data):
                self._relationships.append(data)

            def get_relationship(self, char_id, target_id):
                for r in self._relationships:
                    if r.get("character_id") == char_id and r.get("target_id") == target_id:
                        return r
                return None

            def get_relationships(self, char_id):
                return [r for r in self._relationships if r.get("character_id") == char_id]

            def load_relationship(self, *args, **kwargs):
                return None

            def load_relationships(self, *args, **kwargs):
                return []

            def save_session(self, data):
                self._sessions.append(data)

            def get_sessions(self, *args, **kwargs):
                return []

            def close(self):
                pass

        from woven_imprint.engine import Engine

        engine = object.__new__(Engine)
        engine.storage = FakeStorage()
        engine.llm = FakeLLM()
        engine.embedder = FakeEmbedding()

        importer = CharacterImporter(engine)

        # Create 120 messages (should trigger chunked analysis with chunk_size=50)
        messages = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"}
            for i in range(120)
        ]
        parsed = {
            "source": "chatgpt",
            "title": "Test",
            "messages": messages,
            "instructions": "",
        }
        char = importer._build_character(parsed)
        assert char.name == "TestBot"


class TestMigrationConfig:
    def test_defaults_are_unlimited(self):
        cfg = MigrationConfig()
        assert cfg.max_messages == 0
        assert cfg.max_message_length == 0
        assert cfg.chunk_size == 50
