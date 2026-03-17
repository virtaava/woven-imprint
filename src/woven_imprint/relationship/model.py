"""Relationship model — dimensional tracking of character connections."""

from __future__ import annotations

from ..storage.sqlite import SQLiteStorage
from ..utils.text import generate_id


# Default dimensions for a new relationship
DEFAULT_DIMENSIONS = {
    "trust": 0.0,
    "affection": 0.0,
    "respect": 0.0,
    "familiarity": 0.0,
    "tension": 0.0,
}


# Maximum change per interaction for any dimension
def _max_delta() -> float:
    from ..config import get_config

    return get_config().relationship.max_delta


MAX_DELTA = 0.15  # kept for backward compat; internal code uses _max_delta()


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


class RelationshipModel:
    """Manage relationships between a character and other entities."""

    def __init__(self, storage: SQLiteStorage, character_id: str):
        self.storage = storage
        self.character_id = character_id

    def get_or_create(self, target_id: str) -> dict:
        """Get existing relationship or create a new stranger relationship."""
        rel = self.storage.get_relationship(self.character_id, target_id)
        if rel:
            return rel
        rel = {
            "id": generate_id("rel-"),
            "character_id": self.character_id,
            "target_id": target_id,
            "dimensions": DEFAULT_DIMENSIONS.copy(),
            "power_balance": 0.0,
            "type": "stranger",
            "trajectory": "stable",
            "key_moments": [],
        }
        self.storage.save_relationship(rel)
        return rel

    def update(self, target_id: str, deltas: dict[str, float], new_type: str | None = None) -> dict:
        """Update relationship dimensions with bounded deltas.

        Args:
            target_id: The other entity.
            deltas: Dict of dimension_name → change value (clamped to MAX_DELTA).
            new_type: Optional new relationship type.

        Returns:
            Updated relationship dict.
        """
        rel = self.get_or_create(target_id)

        dims = rel["dimensions"]
        for key, delta in deltas.items():
            if key not in dims:
                continue
            # Clamp delta magnitude
            clamped = max(-MAX_DELTA, min(MAX_DELTA, delta))
            if key == "familiarity":
                # Familiarity is 0-1, only increases (you can't un-know someone)
                dims[key] = _clamp(dims[key] + abs(clamped), 0.0, 1.0)
            elif key == "tension":
                # Tension is 0-1
                dims[key] = _clamp(dims[key] + clamped, 0.0, 1.0)
            else:
                # trust, affection, respect are -1 to 1
                dims[key] = _clamp(dims[key] + clamped)

        if new_type:
            rel["type"] = new_type

        # Update trajectory based on clamped changes (not raw input)
        clamped_deltas = {
            k: max(-MAX_DELTA, min(MAX_DELTA, v)) for k, v in deltas.items() if k in dims
        }
        net = sum(clamped_deltas.get(d, 0) for d in ("trust", "affection", "respect"))
        if net > 0.1:
            rel["trajectory"] = "warming"
        elif net < -0.1:
            rel["trajectory"] = "cooling"
        elif abs(clamped_deltas.get("tension", 0)) > 0.1:
            rel["trajectory"] = "volatile"
        else:
            rel["trajectory"] = "stable"

        self.storage.save_relationship(rel)
        return rel

    def set_baseline(
        self,
        target_id: str,
        dimensions: dict[str, float],
        rel_type: str = "friend",
        trajectory: str = "stable",
    ) -> dict:
        """Set relationship dimensions directly, bypassing per-interaction bounds.

        Used for migration/import when the relationship baseline is known.
        Values are clamped to valid ranges but not bounded by MAX_DELTA.
        """
        rel = self.get_or_create(target_id)
        dims = rel["dimensions"]
        for key in ("trust", "affection", "respect"):
            if key in dimensions:
                dims[key] = _clamp(float(dimensions[key]), -1.0, 1.0)
        for key in ("familiarity", "tension"):
            if key in dimensions:
                dims[key] = _clamp(float(dimensions[key]), 0.0, 1.0)
        rel["type"] = rel_type
        rel["trajectory"] = trajectory
        self.storage.save_relationship(rel)
        return rel

    def get_all(self) -> list[dict]:
        """Get all relationships for this character."""
        return self.storage.get_relationships(self.character_id)

    def get(self, target_id: str) -> dict | None:
        """Get a specific relationship."""
        return self.storage.get_relationship(self.character_id, target_id)

    def add_key_moment(self, target_id: str, moment: str) -> None:
        """Record a pivotal moment in the relationship."""
        rel = self.get_or_create(target_id)
        moments = rel.get("key_moments", [])
        moments.append(moment)
        # Keep only the 20 most recent key moments
        rel["key_moments"] = moments[-20:]
        self.storage.save_relationship(rel)

    def describe(self, target_id: str) -> str:
        """Generate a natural language description of a relationship."""
        rel = self.get(target_id)
        if not rel:
            return f"No established relationship with {target_id}."

        dims = rel["dimensions"]
        parts = [f"Relationship with {target_id} ({rel['type']}):"]

        # Describe each dimension
        descriptors = {
            "trust": [
                (-1, "deeply suspicious"),
                (-0.3, "wary"),
                (0.3, "neutral"),
                (0.7, "trusting"),
                (1, "complete trust"),
            ],
            "affection": [
                (-1, "hostile"),
                (-0.3, "cold"),
                (0.3, "neutral"),
                (0.7, "warm"),
                (1, "deep affection"),
            ],
            "respect": [
                (-1, "contemptuous"),
                (-0.3, "dismissive"),
                (0.3, "neutral"),
                (0.7, "respectful"),
                (1, "deeply admiring"),
            ],
            "familiarity": [
                (0, "strangers"),
                (0.3, "acquaintances"),
                (0.6, "well-known"),
                (1, "intimate knowledge"),
            ],
            "tension": [
                (0, "calm"),
                (0.3, "some tension"),
                (0.6, "significant tension"),
                (1, "explosive"),
            ],
        }

        for dim, levels in descriptors.items():
            val = dims.get(dim, 0)
            label = levels[0][1]
            for threshold, desc in levels:
                if val >= threshold:
                    label = desc
            parts.append(f"  {dim}: {label} ({val:.2f})")

        parts.append(f"  trajectory: {rel['trajectory']}")
        return "\n".join(parts)
