"""Tests for relationship model."""

import pytest

from woven_imprint.storage.sqlite import SQLiteStorage
from woven_imprint.relationship.model import RelationshipModel


@pytest.fixture
def storage():
    s = SQLiteStorage(":memory:")
    s.save_character("c1", "Alice", {})
    yield s
    s.close()


class TestRelationshipModel:
    def test_get_or_create_new(self, storage):
        rm = RelationshipModel(storage, "c1")
        rel = rm.get_or_create("user1")
        assert rel["type"] == "stranger"
        assert rel["dimensions"]["trust"] == 0.0
        assert rel["dimensions"]["familiarity"] == 0.0

    def test_get_or_create_existing(self, storage):
        rm = RelationshipModel(storage, "c1")
        rel1 = rm.get_or_create("user1")
        rel2 = rm.get_or_create("user1")
        assert rel1["id"] == rel2["id"]

    def test_update_dimensions(self, storage):
        rm = RelationshipModel(storage, "c1")
        rm.get_or_create("user1")
        rel = rm.update("user1", {"trust": 0.1, "affection": 0.05})
        assert abs(rel["dimensions"]["trust"] - 0.1) < 0.01
        assert abs(rel["dimensions"]["affection"] - 0.05) < 0.01

    def test_delta_clamping(self, storage):
        rm = RelationshipModel(storage, "c1")
        rm.get_or_create("user1")
        # Try to change trust by 0.5 — should be clamped to MAX_DELTA (0.15)
        rel = rm.update("user1", {"trust": 0.5})
        assert abs(rel["dimensions"]["trust"] - 0.15) < 0.01

    def test_familiarity_only_increases(self, storage):
        rm = RelationshipModel(storage, "c1")
        rm.get_or_create("user1")
        rel = rm.update("user1", {"familiarity": 0.1})
        assert rel["dimensions"]["familiarity"] > 0
        # Try negative — familiarity uses abs(delta)
        rel = rm.update("user1", {"familiarity": -0.05})
        assert rel["dimensions"]["familiarity"] >= 0.1  # Should not decrease

    def test_trajectory_warming(self, storage):
        rm = RelationshipModel(storage, "c1")
        rm.get_or_create("user1")
        rel = rm.update("user1", {"trust": 0.1, "affection": 0.1})
        assert rel["trajectory"] == "warming"

    def test_trajectory_cooling(self, storage):
        rm = RelationshipModel(storage, "c1")
        rm.get_or_create("user1")
        rel = rm.update("user1", {"trust": -0.1, "affection": -0.1})
        assert rel["trajectory"] == "cooling"

    def test_describe(self, storage):
        rm = RelationshipModel(storage, "c1")
        rm.get_or_create("user1")
        rm.update("user1", {"trust": 0.1, "affection": 0.1})
        desc = rm.describe("user1")
        assert "user1" in desc
        assert "trust" in desc.lower()

    def test_key_moment(self, storage):
        rm = RelationshipModel(storage, "c1")
        rm.get_or_create("user1")
        rm.add_key_moment("user1", "First met at the café")
        rel = rm.get("user1")
        assert len(rel["key_moments"]) == 1
        assert "café" in rel["key_moments"][0]
