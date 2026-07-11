"""Tests for Reciprocal Rank Fusion."""

from woven_imprint.utils.rrf import reciprocal_rank_fusion


class TestRRF:
    def test_single_list(self):
        result = reciprocal_rank_fusion([["a", "b", "c"]])
        ids = [x[0] for x in result]
        assert ids == ["a", "b", "c"]

    def test_two_agreeing_lists(self):
        result = reciprocal_rank_fusion(
            [
                ["a", "b", "c"],
                ["a", "b", "c"],
            ]
        )
        # Both lists agree: a should be first
        assert result[0][0] == "a"

    def test_two_disagreeing_lists(self):
        result = reciprocal_rank_fusion(
            [
                ["a", "b", "c"],
                ["c", "b", "a"],
            ]
        )
        # b is rank 2 in both lists, a is rank 1+3, c is rank 3+1
        # With k=60: b gets 2/(61+2)=2/63, a gets 1/61+1/63, c gets 1/63+1/61
        # a and c tie, b is slightly lower. All are very close.
        scores = {item: score for item, score in result}
        # a and c should have identical scores (symmetric)
        assert abs(scores["a"] - scores["c"]) < 1e-10
        # b should also be close but rank 2 in both
        assert len(result) == 3

    def test_empty_list(self):
        result = reciprocal_rank_fusion([])
        assert result == []

    def test_disjoint_lists(self):
        result = reciprocal_rank_fusion(
            [
                ["a", "b"],
                ["c", "d"],
            ]
        )
        assert len(result) == 4

    def test_scores_decrease(self):
        result = reciprocal_rank_fusion([["a", "b", "c", "d"]])
        scores = [s for _, s in result]
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


def test_rrf_weights_bias_fusion():
    from woven_imprint.utils.rrf import reciprocal_rank_fusion

    # list A prefers "x", list B prefers "y"; weighting B 3x must flip the winner
    lists = [["x", "y"], ["y", "x"]]
    unweighted = reciprocal_rank_fusion(lists)
    assert unweighted[0][1] == unweighted[1][1]  # tie without weights
    weighted = reciprocal_rank_fusion(lists, weights=[1.0, 3.0])
    assert weighted[0][0] == "y"


def test_rrf_configurable_k():
    from woven_imprint.utils.rrf import reciprocal_rank_fusion

    lists = [["a", "b"]]
    k10 = dict(reciprocal_rank_fusion(lists, k=10))
    k60 = dict(reciprocal_rank_fusion(lists, k=60))
    assert k10["a"] == 1 / 11
    assert k60["a"] == 1 / 61
