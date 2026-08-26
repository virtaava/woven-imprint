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

    TIERS = ("stranger", "acquaintance", "friend", "close_friend", "adversary")

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
            "state": {},
        }
        self.storage.save_relationship(rel)
        return rel

    @staticmethod
    def tier_for(dims: dict) -> tuple[str, float]:
        """Derive a relationship tier and affinity score from dimension values."""
        aff = (
            0.5 * dims.get("trust", 0.0)
            + 0.3 * dims.get("affection", 0.0)
            + 0.2 * dims.get("respect", 0.0)
        )
        fam = dims.get("familiarity", 0.0)
        if aff <= -0.3:
            return "adversary", aff
        if aff >= 0.5 and fam >= 0.6:
            return "close_friend", aff
        if aff >= 0.25 and fam >= 0.3:
            return "friend", aff
        if fam >= 0.1:
            return "acquaintance", aff
        return "stranger", aff

    def update(
        self,
        target_id: str,
        deltas: dict[str, float],
        new_type: str | None = None,
        *,
        note: str | None = None,
    ) -> dict:
        """Update relationship dimensions with bounded deltas.

        Args:
            target_id: The other entity.
            deltas: Dict of dimension_name → change value (clamped to MAX_DELTA).
            new_type: Optional new relationship type.
            note: Optional short context string attached to any key moment recorded
                by this update.

        Returns:
            Updated relationship dict.
        """
        from ..config import get_config
        from .. import clock

        cfg = get_config().relationship
        rel = self.get_or_create(target_id)
        dims = rel["dimensions"]
        if not cfg.dynamics:
            return self._update_legacy(rel, deltas, new_type)

        state = rel.setdefault("state", {}) or {}
        state.setdefault("updates", 0)
        state.setdefault("recent", [])
        state.setdefault("damping_left", 0)
        state.setdefault("betrayals", 0)
        now = clock.now()
        last = state.get("last_update_at")
        if last:
            try:
                elapsed_days = max(0.0, (now - clock.parse_ts(last)).total_seconds() / 86400.0)
            except ValueError:
                elapsed_days = 0.0
            if elapsed_days > 0 and dims.get("tension", 0.0) > 0:
                dims["tension"] = _clamp(
                    dims["tension"] - cfg.tension_decay_per_day * elapsed_days, 0.0, 1.0
                )

        was_damped = state["damping_left"] > 0
        today = now.date().isoformat()
        applied: dict[str, float] = {}
        for key, delta in deltas.items():
            if key not in dims:
                continue
            clamped = max(-cfg.max_delta, min(cfg.max_delta, float(delta)))
            if key == "trust":
                if clamped <= cfg.betrayal_threshold:
                    state["damping_left"] = cfg.betrayal_damping_turns
                    state["betrayals"] += 1
                    self._push_moment(
                        rel,
                        f"{today}: betrayal — trust {clamped:+.2f}"
                        + (f" — {note}" if note else ""),
                        cfg,
                    )
                elif clamped > 0:
                    clamped *= cfg.trust_gain_factor
                    if was_damped:
                        clamped *= cfg.betrayal_gain_damping
            if key == "familiarity":
                dims[key] = _clamp(dims[key] + abs(clamped), 0.0, 1.0)
            elif key == "tension":
                dims[key] = _clamp(dims[key] + clamped, 0.0, 1.0)
            else:
                dims[key] = _clamp(dims[key] + clamped)
            applied[key] = clamped
            if abs(clamped) >= cfg.key_moment_threshold and not (
                key == "trust" and clamped <= cfg.betrayal_threshold
            ):
                self._push_moment(
                    rel, f"{today}: {key} {clamped:+.2f}" + (f" — {note}" if note else ""), cfg
                )
        if was_damped:
            state["damping_left"] -= 1

        net = (
            applied.get("trust", 0.0) + applied.get("affection", 0.0) + applied.get("respect", 0.0)
        )
        state["recent"] = (state["recent"] + [[net, applied.get("tension", 0.0)]])[
            -cfg.trajectory_window :
        ]
        net_sum = sum(r[0] for r in state["recent"])
        tension_sum = sum(abs(r[1]) for r in state["recent"])
        if net_sum > 0.1:
            rel["trajectory"] = "warming"
        elif net_sum < -0.1:
            rel["trajectory"] = "cooling"
        elif tension_sum > 0.1:
            rel["trajectory"] = "volatile"
        else:
            rel["trajectory"] = "stable"

        rel["type"] = new_type or self.tier_for(dims)[0]
        state["updates"] += 1
        state["last_update_at"] = clock.sqlite_ts(now)
        rel["state"] = state
        self.storage.save_relationship(rel)
        return rel

    def _update_legacy(self, rel: dict, deltas: dict[str, float], new_type: str | None) -> dict:
        """Legacy (pre-dynamics) update arithmetic, byte-identical to prior behavior."""
        dims = rel["dimensions"]
        for key, delta in deltas.items():
            if key not in dims:
                continue
            # Clamp delta magnitude
            clamped = max(-_max_delta(), min(_max_delta(), delta))
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
            k: max(-_max_delta(), min(_max_delta(), v)) for k, v in deltas.items() if k in dims
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

    def _push_moment(self, rel: dict, moment: str, cfg) -> None:
        moments = rel.get("key_moments", []) or []
        moments.append(moment)
        rel["key_moments"] = moments[-cfg.key_moments_limit :]

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
        from ..config import get_config

        rel = self.get_or_create(target_id)
        self._push_moment(rel, moment, get_config().relationship)
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

        tier, aff = self.tier_for(dims)
        parts.append(f"  tier: {rel.get('type', tier)} (affinity {aff:+.2f})")
        for m in (rel.get("key_moments") or [])[-2:]:
            parts.append(f"  recent: {m}")

        parts.append(f"  trajectory: {rel['trajectory']}")
        return "\n".join(parts)
