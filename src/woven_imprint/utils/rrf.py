"""Reciprocal Rank Fusion — merges multiple ranked lists into one."""

from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Merge multiple ranked lists using RRF.

    Args:
        ranked_lists: Each inner list is item IDs ordered by rank (best first).
        k: RRF constant (default 60, standard in literature).

    Returns:
        List of (item_id, score) sorted by fused score descending.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
