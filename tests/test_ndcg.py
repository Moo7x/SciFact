"""Acceptance tests for `ndcg_at_k` (exercise). Every expected value is worked out by hand in
the comment beside it, so a failing test tells you which part of the formula is wrong."""

from __future__ import annotations

import math

import pytest

from scifact.eval.metrics import ndcg_at_k


def test_perfect_ranking_scores_one() -> None:
    assert ndcg_at_k([1, 2, 3], {1}, k=3) == pytest.approx(1.0)


def test_gold_at_rank_two_is_discounted() -> None:
    # DCG = 1 / log2(2 + 1) = 0.6309; IDCG = 1 / log2(1 + 1) = 1
    assert ndcg_at_k([5, 1, 7], {1}, k=3) == pytest.approx(1 / math.log2(3))


def test_two_gold_items_one_misplaced() -> None:
    # DCG = 1/log2(2) + 1/log2(4) = 1.5; IDCG = 1/log2(2) + 1/log2(3) = 1.6309
    expected = 1.5 / (1 + 1 / math.log2(3))
    assert ndcg_at_k([1, 9, 2], {1, 2}, k=3) == pytest.approx(expected)


def test_gold_outside_the_cutoff_scores_zero() -> None:
    assert ndcg_at_k([4, 5, 6, 1], {1}, k=3) == 0.0


def test_ideal_uses_at_most_k_gold_items() -> None:
    # Three gold items but only two slots: a top-2 of gold items is already perfect.
    assert ndcg_at_k([1, 2], {1, 2, 3}, k=2) == pytest.approx(1.0)


def test_ranking_shorter_than_k() -> None:
    assert ndcg_at_k([8, 1], {1}, k=10) == pytest.approx(1 / math.log2(3))


def test_rank_is_rewarded_monotonically() -> None:
    scores = [ndcg_at_k([*[0] * i, 1], {1}, k=10) for i in range(10)]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 < s <= 1.0 for s in scores)


@pytest.mark.parametrize(("gold", "k"), [(set(), 3), ({1}, 0)])
def test_undefined_inputs_raise(gold: set[int], k: int) -> None:
    with pytest.raises(ValueError):
        ndcg_at_k([1, 2, 3], gold, k=k)
