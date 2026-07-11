"""Reciprocal Rank Fusion — merges multiple ranked lists into one."""

from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked ID lists: score(id) = sum_i weight_i / (k + rank_i + 1).

    Args:
        ranked_lists: One ranked list of IDs per strategy (best first).
        k: RRF dampening constant.
        weights: Optional per-list weights (defaults to 1.0 each).

    Returns:
        (id, score) tuples sorted by fused score, descending.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    scores: dict[str, float] = {}
    for ranked, weight in zip(ranked_lists, weights):
        for rank, item_id in enumerate(ranked):
            scores[item_id] = scores.get(item_id, 0.0) + weight / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
