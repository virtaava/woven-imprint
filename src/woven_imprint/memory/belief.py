"""Belief revision system — manage certainty and contradictions."""

from __future__ import annotations

from ..storage.sqlite import SQLiteStorage
from ..utils.text import generate_id


class BeliefReviser:
    """Track certainty of memories and handle contradictions."""

    REINFORCE_DELTA = 0.15
    CONTRADICT_CERTAINTY = 0.0

    def __init__(self, storage: SQLiteStorage, character_id: str):
        self.storage = storage
        self.character_id = character_id

    def reinforce(self, memory_id: str) -> float:
        """Increase certainty of a memory. Returns new certainty value."""
        return self.storage.update_memory_certainty(memory_id, self.REINFORCE_DELTA)

    def contradict(self, old_memory_id: str, new_content: str,
                   source: str = "", session_id: str | None = None) -> dict:
        """Mark an existing memory as contradicted and store the replacement.

        The old memory is not deleted — it remains queryable for character growth
        ("I used to think X, but now I know Y").

        Returns the new memory dict.
        """
        # Mark old as contradicted with zero certainty
        self.storage.update_memory_status(
            old_memory_id, "contradicted", certainty=self.CONTRADICT_CERTAINTY
        )

        # Get the old memory's tier — new one inherits it
        old = self.storage.get_memory(old_memory_id)
        tier = old["tier"] if old else "core"

        # Create replacement memory
        new_memory = {
            "id": generate_id("mem-"),
            "character_id": self.character_id,
            "tier": tier,
            "content": new_content,
            "embedding": None,  # Caller should compute embedding
            "importance": old.get("importance", 0.5) if old else 0.5,
            "certainty": 0.8,  # New contradicting info starts slightly below max
            "status": "active",
            "source_refs": [old_memory_id],
            "session_id": session_id,
            "role": "observation",
            "metadata": {"contradicts": old_memory_id, "source": source},
        }
        self.storage.save_memory(new_memory)
        return new_memory

    def invalidate(self, memory_id: str) -> None:
        """Remove from retrieval without deletion. Preserved in archive."""
        self.storage.update_memory_status(memory_id, "archived")

    def detect_contradictions(self, new_content: str,
                              existing_memories: list[dict]) -> list[dict]:
        """Find existing memories that may contradict new content.

        This is a lightweight heuristic check. Full NLI-based contradiction
        detection requires the LLM and is handled by the consistency checker.

        Returns list of potentially contradicting memories.
        """
        candidates = []
        new_lower = new_content.lower()

        # Simple heuristic: look for negation patterns or conflicting statements
        negation_pairs = [
            ("likes", "dislikes"), ("loves", "hates"),
            ("trusts", "distrusts"), ("is a", "is not a"),
            ("always", "never"), ("can", "cannot"),
            ("friend", "enemy"), ("alive", "dead"),
        ]

        for mem in existing_memories:
            mem_lower = mem["content"].lower()
            for pos, neg in negation_pairs:
                if (pos in new_lower and neg in mem_lower) or \
                   (neg in new_lower and pos in mem_lower):
                    candidates.append(mem)
                    break

        return candidates
